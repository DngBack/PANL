# PANL

Experiments for identifying question-answer interaction in post-answer confidence states.

The project asks whether the post-answer newline (PANL) state contains a `Q x A`
interaction that is separable from conditional answer likelihood and commitment, and
whether that component is causally used to produce verbal confidence.

## Status

The falsifiable research plan, experiment matrix, data contracts, compute assumptions, and
delivery order are documented in [docs/experiment-plan.md](docs/experiment-plan.md).

Implemented:

- **Data contract and Tier-1 builder.** Crossed 2x2 blocks over controlled facts, with the
  splits, labels, and run manifest of plan section 3.
- **Semantic position resolver.** AC, LAT, PANL, PANL+1 and CC resolved from real
  tokenization, with a frozen per-tokenizer snapshot (plan section 4.1).
- **Activation collection.** A project-owned adapter over TransformerLens, the residual
  stream at every layer and semantic position in a Zarr store, and the teacher-forced
  likelihood controls of plan section 4.3.
- **E0.** The confidence contrast with block-level bootstrap and permutation tests, and a
  layer x position patching sweep with preregistered gates.

Not yet implemented: E1-E4, the probes and nuisance projections, and the intervention module.

## Environment

Python dependencies are managed exclusively with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest
```

Commit both `pyproject.toml` and `uv.lock`. Do not commit model weights, raw datasets,
activation stores, or experiment outputs.

## Usage

```bash
# Build the crossed-block table (writes the Parquet file and its run manifest).
uv run panl data build --config configs/data/tier1.yaml

# Re-check every invariant of plan section 7 against an existing table.
uv run panl data validate data/processed/quadruples.parquet

# Resolve AC/LAT/PANL/PANL+1/CC for a tokenizer. Exits non-zero if any semantic
# boundary is destroyed by tokenization -- run this before booking GPU time.
uv run panl positions check --model Qwen/Qwen2.5-0.5B-Instruct

# Freeze the resolved positions so a tokenizer change fails CI instead of a run.
uv run panl positions snapshot --model Qwen/Qwen2.5-0.5B-Instruct \
  --out tests/fixtures/positions_qwen2_5_0_5b_instruct.json

# Score every fact against the model *before* building blocks: is the answer format
# on-policy, and does the model actually find any of these facts hard?
uv run panl score --config configs/experiment/e0_qwen7b.yaml --out outputs/scores.parquet

# Route ablation: how does the answer reach the confidence read-out? This is the primary
# localization evidence -- see "Redundant routes" below for why the patching sweep is not.
uv run panl e0 routes --config configs/experiment/e0_qwen7b.yaml

# The E3 controls for that result: both directions, and three wrong-content sources.
uv run panl e0 controls --config configs/experiment/e0_qwen7b.yaml --start-layer 16

# The original patching sweep. Superseded, kept because it is the evidence for the trap.
uv run panl e0 run --config configs/experiment/e0_qwen7b.yaml

# Re-render either report from saved parquet: no GPU, no model. A change to how a result is
# summarized must never cost GPU time twice.
uv run panl e0 report        outputs/<run> --config configs/experiment/e0_qwen7b.yaml
uv run panl e0 routes-report outputs/<run> --config configs/experiment/e0_qwen7b.yaml
```

Tests that need a tokenizer from the Hugging Face hub are marked; skip them with
`uv run pytest -m "not tokenizer"`.

## Design notes

**Crossed cells must be genuinely wrong.** Two facts are paired into a block only when
neither's answer is correct for the other's subject. The Tier-1 fact base is curated to be
*functional* for that reason: subjects with more than one defensible answer (the Netherlands
and Bolivia have split capitals; Bulgaria and Croatia moved to the euro; the radio and the
transistor have contested inventors) are deliberately absent.

**Splits are assigned to facts, not blocks.** Blocks are not independent when a fact is
reused across pairing rounds, so a block-level split leaks question and answer identities.
Assigning facts first, stratified per relation family, makes identities disjoint by
construction. `pairings_per_fact: 1` yields strictly independent blocks for the block
bootstrap of plan section 4.2, at the cost of halving the block count.

**Positions are never hard-coded.** Whether `"\nConfidence"` is one token or two is a
property of a tokenizer's pre-tokenizer regex. The resolver derives positions from character
offsets and *asserts* every semantic boundary; a tokenizer that merges the post-answer
newline into the following word raises rather than returning a plausible-looking index.

**Batches hold one token length, never padding.** A left-padded batch is only correct if the
attention mask and rotary offsets are threaded through exactly right, and an error there
would corrupt every activation without failing anything. Prompts differ only in the question
and the answer, so their lengths cluster tightly and grouping by length costs almost nothing.

**Interventions never read the activation store.** The Zarr store is float16, which is fine
for analysis and wrong for a causal patch: a patch must carry the values the model actually
computed. The patching sweep re-runs the model and patches from live bf16 activations.

**CC is a harness check, not evidence.** Patching the confidence colon transplants the
read-out state itself, so it *must* score near 1. It is in the sweep to catch a broken
harness, and the E0 gate fails if PANL ever beats it.

**Redundant routes make single-position patching lie.** The answer reaches the confidence
read-out two ways — directly, and through PANL — and *either alone* carries ~90% of the
confidence gap. So patching PANL in the intact model reports a null no matter what PANL
contains: whatever the patch removes, the direct route puts straight back. Localizing a
"cache" requires severing the bypass first (`panl e0 routes`), and then the same patch flips
96% of decisions. The full account is in [docs/stage-1-findings.md](docs/stage-1-findings.md).

**Sufficient is not necessary.** A patch-under-isolation result is evidence about what a
position *carries*. It is not proof that the intact model routes through it — the experiment
forces the model to use PANL and then observes that it does. The direct route carries 88% of
the gap on its own, so PANL is sufficient and redundant, and the code says so where it would
otherwise be tempting to overclaim.

## Target models

- Primary/iteration: `Qwen/Qwen2.5-7B-Instruct`
- Cross-family replication: `google/gemma-3-27b-it`
- Small smoke tests only: `Qwen/Qwen2.5-0.5B-Instruct`

CUDA-capable Linux compute is required for the main mechanistic experiments. CPU runs are
only intended for data validation, unit tests, and statistical analysis.
