"""Project-owned adapter over TransformerLens.

Plan section 6: "Keep a thin project-owned adapter around hook names and semantic positions so
the analysis is not coupled directly to library-internal names." Nothing outside this module
mentions `blocks.{i}.hook_resid_post`, `HookedTransformer`, or an `ActivationCache`. The
analysis speaks in layers, semantic positions, and confidence margins.

A note on the loaded weights: TransformerLens folds LayerNorm and centres the writing and
unembedding weights by default. Those rewrites leave the model's function unchanged but do
change what the residual stream literally contains, so activations captured here are only
comparable with activations captured the same way -- which is why the load flags land in the
run manifest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Self

import torch

from panl.models.batching import PromptBatch
from panl.models.confidence import ConfidenceClasses, resolve_confidence_classes
from panl.models.spec import ModelSpec
from panl.models.tokenizer import FastTokenizer, load_tokenizer

DTYPES: Final[dict[str, torch.dtype]] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """One forward pass over a batch.

    Attributes:
        confidence_margin: [batch] pre-softmax logit(high) - logit(low) at CC.
        answer_logprobs: [batch] list of teacher-forced log p(token | prefix) over the answer.
        activations: layer -> position -> [batch, d_model], only for the layers requested.
    """

    confidence_margin: torch.Tensor
    answer_logprobs: list[torch.Tensor]
    activations: dict[int, dict[str, torch.Tensor]]


class HookedModelAdapter:
    def __init__(
        self,
        model: Any,
        tokenizer: FastTokenizer,
        spec: ModelSpec,
        classes: ConfidenceClasses,
    ) -> None:
        self._model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.classes = classes

    @classmethod
    def load(cls, spec: ModelSpec, *, device: str | None = None, n_devices: int = 1) -> Self:
        from transformer_lens import HookedTransformer

        tokenizer = load_tokenizer(spec.model_id, revision=spec.revision)
        classes = resolve_confidence_classes(tokenizer)
        model = HookedTransformer.from_pretrained(
            spec.model_id,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            n_devices=n_devices,
            dtype=DTYPES[spec.dtype],
            fold_ln=spec.fold_ln,
            center_writing_weights=spec.center_writing_weights,
            center_unembed=spec.center_unembed,
        )
        model.eval()
        return cls(model, tokenizer, spec, classes)

    # -- shape ---------------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return int(self._model.cfg.n_layers)

    @property
    def d_model(self) -> int:
        return int(self._model.cfg.d_model)

    @property
    def device(self) -> torch.device:
        return torch.device(self._model.cfg.device)

    def _resid_hook(self, layer: int) -> str:
        """The one place a TransformerLens hook name is spelled out.

        Checked against the model's own hook registry rather than trusted: a hook name that
        silently does not exist would mean `run_with_hooks` patches nothing and every
        intervention quietly reports a null effect.
        """
        name = f"blocks.{layer}.hook_resid_post"
        if name not in self._model.hook_dict:
            msg = f"{name!r} is not a hook on {self.spec.model_id}"
            raise KeyError(msg)
        return name

    # -- read-out ------------------------------------------------------------------------

    def _margin_at_cc(self, logits: torch.Tensor, batch: PromptBatch) -> torch.Tensor:
        """logit(" high") - logit(" low") at CC, per row. Pre-softmax, per plan section 4.2."""
        rows = torch.arange(batch.size, device=logits.device)
        cc = batch.positions["CC"].to(logits.device)
        at_cc = logits[rows, cc, :].float()
        return at_cc[:, self.classes.high_token_id] - at_cc[:, self.classes.low_token_id]

    def _answer_logprobs(self, logits: torch.Tensor, batch: PromptBatch) -> list[torch.Tensor]:
        """Teacher-forced log p(answer token | everything before it), per row.

        The logit at index i-1 predicts the token at index i, hence the shift.
        """
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        out: list[torch.Tensor] = []
        for row in range(batch.size):
            start, end = (int(x) for x in batch.answer_spans[row])
            targets = batch.input_ids[row, start:end]
            predicted_from = logprobs[row, start - 1 : end - 1, :]
            out.append(predicted_from.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
        return out

    # -- passes --------------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self, batch: PromptBatch, *, cache_layers: Sequence[int] | None = None
    ) -> ForwardResult:
        """A clean forward pass, optionally capturing the residual stream."""
        batch = batch.to(self.device)
        layers = list(cache_layers) if cache_layers is not None else []

        if not layers:
            logits = self._model(batch.input_ids, return_type="logits")
            return ForwardResult(
                confidence_margin=self._margin_at_cc(logits, batch),
                answer_logprobs=self._answer_logprobs(logits, batch),
                activations={},
            )

        wanted = {self._resid_hook(layer): layer for layer in layers}
        logits, cache = self._model.run_with_cache(
            batch.input_ids, return_type="logits", names_filter=list(wanted)
        )
        rows = torch.arange(batch.size, device=self.device)
        activations: dict[int, dict[str, torch.Tensor]] = {}
        for name, layer in wanted.items():
            resid = cache[name]  # [batch, seq, d_model]
            activations[layer] = {
                position: resid[rows, index.to(resid.device), :]
                for position, index in batch.positions.items()
            }
        return ForwardResult(
            confidence_margin=self._margin_at_cc(logits, batch),
            answer_logprobs=self._answer_logprobs(logits, batch),
            activations=activations,
        )

    @torch.no_grad()
    def greedy_answers(
        self, prefixes: Sequence[str], *, max_new_tokens: int = 12, stop: str = "\n"
    ) -> list[str]:
        """What the model *itself* answers, decoded greedily from an "Answer:" prefix.

        This is what makes the teacher-forced design auditable. If the model would not
        naturally produce the answer we force into its mouth, then the trajectory is
        off-policy and the answer's log-probability measures surprise at our format rather
        than knowledge of the fact -- which silently voids the fluency controls.

        Prefixes are grouped by token length so no padding is introduced, exactly as in
        `make_batches`.
        """
        encoded = [
            [int(i) for i in self.tokenizer(p, add_special_tokens=False)["input_ids"]]
            for p in prefixes
        ]
        by_length: dict[int, list[int]] = {}
        for index, ids in enumerate(encoded):
            by_length.setdefault(len(ids), []).append(index)

        out: list[str] = [""] * len(prefixes)
        for length in sorted(by_length):
            rows = by_length[length]
            tokens = torch.tensor([encoded[i] for i in rows], dtype=torch.long, device=self.device)
            for _ in range(max_new_tokens):
                logits = self._model(tokens, return_type="logits")
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, nxt], dim=1)
            for offset, row in enumerate(rows):
                text = self._decode(tokens[offset, length:].tolist())
                out[row] = text.split(stop)[0].strip()
        return out

    def _decode(self, ids: list[int]) -> str:
        return str(self.tokenizer.decode(ids, skip_special_tokens=True))

    def _attn_hook(self, layer: int) -> str:
        name = f"blocks.{layer}.attn.hook_pattern"
        if name not in self._model.hook_dict:
            msg = f"{name!r} is not a hook on {self.spec.model_id}"
            raise KeyError(msg)
        return name

    def _position_mask(self, batch: PromptBatch, name: str) -> torch.Tensor:
        """[batch, seq] boolean mask of a named set of positions.

        Sets, not single indices. The first version of the knockout named one query per edge
        and assumed the only token between PANL and CC was PANL+1. It is not: Qwen splits
        "Confidence" into "Conf" + "idence", so there is an *unnamed* token that can see the
        answer and that CC reads. Every knockout leaked through it. Position sets exist so
        that a route is cut by what it *is* -- everything after PANL -- rather than by a list
        of names that a tokenizer can silently invalidate.
        """
        arange = torch.arange(batch.seq_len, device=self.device).unsqueeze(0)
        panl = batch.positions["PANL"].unsqueeze(1)
        cc = batch.positions["CC"].unsqueeze(1)

        if name == "answer":
            start = batch.answer_spans[:, 0].unsqueeze(1)
            end = batch.answer_spans[:, 1].unsqueeze(1)
            return (arange >= start) & (arange < end)
        if name == "after_answer":
            # PANL and everything after it: every query that can see the answer at all.
            return arange >= panl
        if name == "post_panl":
            # Everything strictly after PANL, up to and including CC.
            return arange > panl
        if name == "suffix_before_cc":
            # The confidence word: however many tokens the tokenizer splits it into.
            return (arange > panl) & (arange < cc)
        if name in batch.positions:
            return arange == batch.positions[name].unsqueeze(1)
        msg = f"unknown position set {name!r}"
        raise KeyError(msg)

    @torch.no_grad()
    def run_with_knockout(
        self,
        batch: PromptBatch,
        *,
        edges: Sequence[tuple[str, str]],
        layers: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Sever attention edges and read the confidence margin at CC.

        `edges` are (query position, key set) pairs, e.g. `("CC", "answer")` to stop the
        confidence colon from attending to the answer tokens at all. Attention is zeroed and
        the query's remaining weights renormalized, which is equivalent to removing those keys
        before the softmax.

        This is the one measurement in E0 that saturation cannot distort. The patching effect
        divides by the clean gap, so a step-function read-out drives it toward zero regardless
        of the mechanism. Severing a route asks a different question -- *can the information
        get there at all* -- and its answer does not depend on how steep the read-out is.
        """
        batch = batch.to(self.device)
        knock = self._make_knockout(batch, edges)
        chosen = list(layers) if layers is not None else list(range(self.n_layers))
        logits = self._model.run_with_hooks(
            batch.input_ids,
            return_type="logits",
            fwd_hooks=[(self._attn_hook(layer), knock) for layer in chosen],
        )
        return self._margin_at_cc(logits, batch)

    def _make_knockout(self, batch: PromptBatch, edges: Sequence[tuple[str, str]]) -> Any:
        """An attention hook that zeroes every (query in Q, key in K) pair and renormalizes.

        Both sides are *sets of positions*, not single indices. The first version of this named
        one query per edge and assumed PANL+1 was the only token between PANL and CC. It is
        not: Qwen splits "Confidence" into "Conf" + "idence", so an unnamed token sits at CC-1,
        it can see the answer, and CC reads it. Every knockout leaked through that token, and
        every route number computed before this fix was contaminated.

        Zeroing the weights and rescaling the affected query rows is equivalent to deleting
        those keys before the softmax. Rows that no query mask touches are left byte-identical
        rather than divided by their own sum, so an unrelated position cannot drift.
        """
        prepared = [
            (self._position_mask(batch, query), self._position_mask(batch, keys))
            for query, keys in edges
        ]

        def knock(pattern: torch.Tensor, hook: Any) -> torch.Tensor:
            del hook
            for query_mask, key_mask in prepared:
                cut = query_mask[:, None, :, None] & key_mask[:, None, None, :]
                zeroed = pattern.masked_fill(cut, 0.0)
                renormalized = zeroed / zeroed.sum(-1, keepdim=True).clamp_min(1e-9)
                pattern = torch.where(query_mask[:, None, :, None], renormalized, pattern)
            return pattern

        return knock

    @torch.no_grad()
    def run_with_patch(
        self,
        batch: PromptBatch,
        *,
        layer: int | Sequence[int],
        position: str,
        source: torch.Tensor,
        edges: Sequence[tuple[str, str]] = (),
    ) -> torch.Tensor:
        """Overwrite the residual stream at one layer and one semantic position, then read
        the confidence margin at CC.

        Args:
            layer: a single layer, or several. Patching one layer *understates* what a
                position carries whenever that position keeps reading its source: PANL still
                attends to the answer tokens at every later layer, so it re-acquires the
                information the patch overwrote. Passing a run of layers freezes the
                position's whole downstream trajectory to the source's and removes that
                leak. `source` must then be [batch, len(layer), d_model].
            source: replacement vectors -- [batch, d_model] for a single layer, or
                [batch, n_layers, d_model] for several. Typically captured from the row's
                partner cell at the same semantic position.
            edges: attention edges to sever at the same time, as in `run_with_knockout`.
                Without this, a patch at a position the model does not *need* will look weak
                no matter what that position contains: the answer tokens are still in context,
                and any redundant route simply re-supplies the information the patch removed.
                Severing the competing routes first is what turns the patch into a test of
                what this position carries.

        Returns:
            [batch] confidence margin under the patch.
        """
        batch = batch.to(self.device)
        index = batch.positions[position]
        replacement = source.to(device=self.device)
        rows = torch.arange(batch.size, device=self.device)

        layers = [layer] if isinstance(layer, int) else list(layer)
        if replacement.ndim == 2:
            replacement = replacement.unsqueeze(1).expand(-1, len(layers), -1)
        if replacement.shape[1] != len(layers):
            msg = (
                f"source has {replacement.shape[1]} layer slots but {len(layers)} layers "
                f"were requested"
            )
            raise ValueError(msg)

        def make_patch(slot: int) -> Any:
            def patch(resid: torch.Tensor, hook: Any) -> torch.Tensor:
                del hook
                resid[rows, index, :] = replacement[:, slot, :].to(resid.dtype)
                return resid

            return patch

        hooks: list[tuple[str, Any]] = [
            (self._resid_hook(layer_i), make_patch(slot)) for slot, layer_i in enumerate(layers)
        ]

        if edges:
            knock = self._make_knockout(batch, edges)
            hooks += [(self._attn_hook(layer_i), knock) for layer_i in range(self.n_layers)]

        logits = self._model.run_with_hooks(batch.input_ids, return_type="logits", fwd_hooks=hooks)
        return self._margin_at_cc(logits, batch)
