"""The model description that lands in a run manifest.

Kept free of torch and TransformerLens so that configs, the CLI, and the analysis can be
loaded and validated without importing a deep-learning stack.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from panl.models.prompts import PromptStyle


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str = "main"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    prompt_style: PromptStyle = PromptStyle.CHAT
    role: Literal["smoke", "primary", "replication"] = "primary"

    # TransformerLens weight rewrites. Recorded because they leave the model's function
    # unchanged but do change what the residual stream literally contains -- activations are
    # only comparable across runs that used the same flags.
    fold_ln: bool = True
    center_writing_weights: bool = True
    center_unembed: bool = True
