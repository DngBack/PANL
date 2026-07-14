# PANL

Experiments for identifying question-answer interaction in post-answer confidence states.

The project asks whether the post-answer newline (PANL) state contains a `Q x A`
interaction that is separable from conditional answer likelihood and commitment, and
whether that component is causally used to produce verbal confidence.

## Status

The repository is in the experiment-design and infrastructure phase. The falsifiable
research plan, experiment matrix, data contracts, compute assumptions, and delivery order
are documented in [docs/experiment-plan.md](docs/experiment-plan.md).

## Environment

Python dependencies are managed exclusively with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest
```

Commit both `pyproject.toml` and `uv.lock`. Do not commit model weights, raw datasets,
activation stores, or experiment outputs.

## Target models

- Primary/iteration: `Qwen/Qwen2.5-7B-Instruct`
- Cross-family replication: `google/gemma-3-27b-it`
- Small smoke tests only: `Qwen/Qwen2.5-0.5B-Instruct`

CUDA-capable Linux compute is required for the main mechanistic experiments. CPU runs are
only intended for data validation, unit tests, and statistical analysis.
