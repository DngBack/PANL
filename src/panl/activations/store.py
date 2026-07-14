"""Zarr-backed activation store.

Plan section E2: "Store metadata/behavior as Parquet and activation shards as Zarr. Never
accumulate all layers on GPU or serialize a single monolithic tensor. Activation files must
record model revision, prompt hash, block IDs, dtype, shape, and checksums."

Activations are written batch by batch, straight off the GPU, so the host never holds the
full tensor. The array is indexed [row, layer, position, d_model]; rows line up 1:1 with the
behaviour Parquet, which is the join key for everything downstream.

float16 is deliberate and is the reason interventions never read from this store: it is an
analysis artifact, and a causal patch must carry the model's own bf16 values, not a value
that has been through a lossy round trip.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from panl.models.positions import POSITION_NAMES

ARRAY_NAME = "resid_post"


@dataclass(frozen=True, slots=True)
class StoreSpec:
    n_rows: int
    n_layers: int
    d_model: int
    positions: tuple[str, ...] = POSITION_NAMES

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.n_rows, self.n_layers, len(self.positions), self.d_model)


class ActivationWriter:
    """Creates the array up front, then fills it row-block by row-block."""

    def __init__(self, path: Path, spec: StoreSpec, *, metadata: dict[str, Any]) -> None:
        self.path = path
        self.spec = spec
        path.parent.mkdir(parents=True, exist_ok=True)
        # One chunk per (row-block, layer): the E2 layer scan reads a single layer across all
        # rows, and that access pattern should not have to touch every other layer's bytes.
        self._array = zarr.create_array(
            store=str(path),
            name=ARRAY_NAME,
            shape=spec.shape,
            chunks=(min(64, spec.n_rows), 1, len(spec.positions), spec.d_model),
            dtype="float16",
            overwrite=True,
        )
        self._array.attrs.update(
            {
                **metadata,
                "positions": list(spec.positions),
                "shape": list(spec.shape),
                "dtype": "float16",
                "layout": "[row, layer, position, d_model]",
            }
        )

    def write(
        self, row_indices: tuple[int, ...], activations: dict[int, dict[str, torch.Tensor]]
    ) -> None:
        """Write one batch. `activations[layer][position]` is [batch, d_model]."""
        layers = sorted(activations)
        block = np.stack(
            [
                np.stack(
                    [
                        activations[layer][position].to(torch.float16).cpu().numpy()
                        for position in self.spec.positions
                    ],
                    axis=1,
                )
                for layer in layers
            ],
            axis=1,
        )  # [batch, layer, position, d_model]
        for offset, row in enumerate(row_indices):
            self._array[row] = block[offset]

    def finalize(self) -> str:
        """Checksum the written bytes so the manifest can pin the artifact."""
        digest = hashlib.sha256()
        for chunk in sorted(self.path.rglob("*")):
            if chunk.is_file():
                digest.update(chunk.read_bytes())
        checksum = digest.hexdigest()
        self._array.attrs["sha256"] = checksum
        return checksum


def read_activations(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    array = zarr.open_array(store=str(path), path=ARRAY_NAME, mode="r")
    return np.asarray(array[:]), dict(array.attrs)
