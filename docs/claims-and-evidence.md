# Claims and evidence

**Model:** `Qwen/Qwen2.5-7B-Instruct` (bf16, TransformerLens, fold_ln + centred weights)
**Data:** 140 blocks / 560 cells, `train` split of `data/processed/quadruples.parquet`
**Statistics:** percentile bootstrap over **blocks** (10,000 resamples); the block, not the cell,
is the resampling unit
**Date:** 2026-07-15 · **Tests:** 198 CPU + 9 GPU · `ruff` and `mypy --strict` clean

This document is the audit trail. Every claim below is stated with the evidence that supports
it, the artifact it came from, and — the part that matters — **what would falsify it**. Claims
we cannot currently make are in §5, and they are the majority of the original research plan.

> **Provenance warning.** The system prompt was changed on 2026-07-15 after it was found to be
> off-policy (§4.1). Numbers in §2 and §3 are from the **fixed** prompt. Numbers marked
> *(pre-fix)* are from `outputs/e0-primary-train/` and are retained only as evidence for §4;
> they must not be quoted as results.

---

## 1. What the experiment measures

The prompt ends `Answer: <answer>\nConfidence:` and we read the pre-softmax margin
`z = logit(" high") - logit(" low")` at the final token. Semantic positions:

```
... Answer :        <answer tokens>      \n      Conf  idence   :
        AC                    ... LAT   PANL    PANL1   (—)     CC
```

A **block** is a 2x2: two questions, each one's gold answer, crossed.

| | A1 | A2 |
|---|---|---|
| **Q1** | matched | crossed |
| **Q2** | crossed | matched |

The **gap** is the block's matched-minus-crossed confidence margin. Note that in this design
`I_k = z11 - z12 - z21 + z22` is exactly twice the gap: the "interaction" and the
"matched-vs-crossed contrast" are the same number (see §5.1).

---

## 2. Claims we can make

### C1 — The route ablation is valid

**Claim.** Severing attention from every query that can see the answer drives the confidence gap
to zero. There is no route we have not accounted for.

| condition | gap (logits) | 95% CI | % of clean |
|---|---|---|---|
| clean | **+29.66** | [+29.10, +30.17] | 100% |
| **cut everything** (`after_answer ← answer`) | **−0.03** | [−0.11, +0.06] | **−0%** |

Also holds on the 48 length-matched blocks (0.4%), so it is not a rotary/position artefact.

**Why it matters.** This is the precondition for every other number here. The *first* version of
this ablation left a 19% residual — which we nearly wrote up as a "floor" — and that residual
was a leak, not a floor (§4.2).

**Falsified by:** a nonzero gap under `cut everything`. That would mean an unfound route.

**Artifact:** `outputs/routes-fixed/route_conditions.parquet`

---

### C2 — The answer reaches the confidence read-out by redundant routes

**Claim.** PANL and the direct edge are each *individually sufficient*. Neither is necessary.

| condition | gap | % of clean |
|---|---|---|
| **only via PANL** (`answer → PANL → … → CC`) | +28.83 | **97%** |
| **only direct** (`answer → CC`) | +21.18 | **71%** |
| cut `CC ← answer` alone | +28.83 | 97% |
| cut `PANL ← answer` alone | +23.36 | 79% |
| cut `suffix ← answer` alone | +29.99 | 101% |

Cutting any *single* route costs almost nothing. Cutting all of them costs everything.

**Why it matters.** This is why patching PANL in the intact model reports a null (§4.3): the
direct route re-supplies whatever the patch removes. **A single-position patch cannot localize a
position that has a bypass, no matter what that position contains.**

**Falsified by:** a model where cutting one route collapses the gap.

**Artifact:** `outputs/routes-fixed/route_conditions.parquet`

---

### C3 — PANL is a sufficient, bidirectional carrier of the confidence signal

**Claim.** With the bypasses severed, freezing PANL's residual trajectory from layer 16 onward
transplants the read-out's decision, in both directions, and neither control position does.

| position | restore: effect / flip | ablate: effect / flip |
|---|---|---|
| **PANL** | **+0.990 / 95%** | **+0.993 / 96%** |
| PANL+1 | −0.005 / 0% | +0.027 / 0% |
| AC | +0.001 / 0% | −0.000 / 0% |

Peak effect **1.002**, moving **+28.89 of a possible 28.9 logits** — a complete transplant.
Self-patching a row with its own activation is an **exact** no-op, so this is not an artefact of
the intervention machinery.

**Falsified by:** the control positions moving too; a one-directional effect.

**Artifacts:** `outputs/routes-fixed/isolated_summary.parquet`,
`outputs/controls-decisive/controls_summary.parquet`

---

### C4 — CC reads PANL at layers ~17–20

**Claim.** Freezing PANL from L16 onward works; from L20 onward does nothing. By L20 the read
has already happened.

| patch span | effect | flip |
|---|---|---|
| L14..end | 1.00 | 95% |
| **L16..end** | **0.99** | **95%** |
| L18..end | 0.50 | 65% |
| L20..end | −0.01 | **0%** |

**Reading the cliff correctly matters.** Every start layer *below* the read point ties at the
peak — a span from L4 contains the span from L16 — so `argmax` reports an arbitrary member of
that tie (L4) and says nothing. The informative number is the **last** start layer that still
works. (`routes.read_cliff`.)

**Falsified by:** a flat profile, or a peak that does not decay.

**Artifact:** `outputs/routes-fixed/isolated_summary.parquet`

---

### C5 — The model is confident by default; PANL carries **doubt**

**Claim.** Destroying PANL — with *anything* — makes a diffident model confident. Nothing makes
a confident model doubt except a source that actually carries doubt.

Two independent demonstrations:

**(a) Cut every route.** Both conditions converge on *confident*:

| | matched | crossed |
|---|---|---|
| clean | +19.29 | **−10.37** |
| cut everything | +15.07 | **+15.09** |

**(b) Write noise into PANL.** A norm-matched random direction:

| direction | flip rate | logits moved |
|---|---|---|
| **restore** (target = crossed) | **96%** | +28.24 |
| **ablate** (target = matched) | **1%** | −0.15 |

**Why it matters — this is the most useful thing in the document.** The **restore direction is
confounded for every destructive intervention**. A study that patches a confidence cache and
reports only the restore direction is reporting an effect that *noise reproduces*. Our own
route-ablation report did exactly that before the controls were run.

**Falsified by:** a model whose default is low confidence, or noise that ablates as well as it
restores.

**Artifacts:** `outputs/routes-fixed/route_conditions.parquet`,
`outputs/controls-decisive/controls_summary.parquet`

---

### C6 — What is cached is a **verdict**, not a **relation**

**This is the answer to the original research question, and it is a no.**

**Claim.** The signal PANL carries is not specific to the question–answer pair. A crossed cell
from an **unrelated block** — a different question, a different answer — removes confidence just
as well as the true partner.

Ablate direction (the uncontaminated one):

| source | effect | logits moved | flip |
|---|---|---|---|
| **partner** (same block, the real intervention) | +0.993 | −28.61 | **96%** |
| **crossed donor** (different block) | **+0.997** | **−28.75** | **97%** |
| gaussian (noise) | +0.005 | −0.15 | 1% |
| mean-ablation | +0.520 | −14.98 | 5% |
| matched donor (different block) | −0.013 | +0.37 | 1% |

And symmetrically, in restore: matched donor **96%** vs partner **95%**.

`random_cell` (49% ablate) is *not* evidence either way: its donor pool is 50/50 matched/crossed,
so ~50% is exactly what a binary signal predicts. Only the *type-pure* donors decide it, and they
decide it flatly.

**Interpretation.** The relational computation — *does this answer fit this question* — happens
**upstream** of PANL. What is written into PANL is only its scalar outcome. The cache stores the
verdict, not the reasoning.

**Falsified by:** a crossed donor from another block failing to ablate. **Threatened by:** §5.4.

**Artifact:** `outputs/controls-decisive/controls_summary.parquet`

---

## 3. Methodological findings (each cost us a wrong result first)

These are model-agnostic and do not depend on the dataset. Each produced a plausible,
publishable-looking number that was wrong, and **none of them crashed anything**.

### M1 — Redundant routes make single-position patching report false nulls
Patching PANL in the intact model: **+3.01 of 30 logits, 0% of decisions flipped**. It looks
like PANL is empty. It is not (C3). The direct route puts back whatever the patch removes.
*Any* patching study of a "cache" is exposed to this.

### M2 — Never define an intervention by naming tokens
Qwen splits `"Confidence"` into `"Conf"` + `"idence"`. The second token has **no name** in our
position schema, sits at CC−1, can see the answer, and CC reads it. Every knockout leaked through
it; `only via PANL` was not isolating PANL. **The "19% floor" we were about to explain away as a
rotary artefact was this leak.** Interventions now address position *sets* (`after_answer`,
`post_panl`, `suffix_before_cc`), so a tokenizer cannot silently reopen a path.

### M3 — A single-layer patch leaks when the position keeps reading its source
`ISOLATE_PANL` deliberately leaves `PANL ← answer` open — PANL must see the answer or it could
carry nothing. So a one-layer patch is re-supplied at every later layer. Freezing the trajectory
closes it: **0.42 → 1.00**, 50% → 96%.

### M4 — The normalized patching metric conflates patch strength with saturation
The same patch moves **more** in absolute logits on 7B (+3.01) than on 0.5B (+1.21), yet scores
0.10 versus 0.92 — because 7B's clean gap is 23x larger. Report absolute shift and flip rate
alongside any gap-normalized effect.

### M5 — A "LAT effect" is an artefact of answer length
Apparent LAT effect 0.395. Split by answer token count: **1 token → 1.018; 2 → 0.511; 3+ →
0.128**. LAT is not a mechanism; it measures *how much of the answer the patch replaced*. For a
one-token answer, LAT **is** the answer.

### M6 — Exploratory subsets lie
A 60-block exploratory run reported a 76% flip rate. On the full 140-block split it was 50%.

---

## 4. Data-design defects found by running the model

### 4.1 The prompt was off-policy *(fixed)*
The model does not say `Paris`; it says *"The capital of France is Paris."* Teacher-forcing our
gold was therefore an **off-policy trajectory**, and the answer's log-probability — the nuisance
variable the entire fluency control of plan §4.3 rests on — was measuring *surprise at our
formatting*, not knowledge. **Residualizing on it would have controlled for nothing.**
Fixed with a format-strict system prompt: on-policy **0% → 90%**, verified by `panl score`,
which decodes the model's own answer and compares it to gold.

### 4.2 `matched` and `correct` are perfectly confounded *(not fixed)*
Crosstab is exactly diagonal: **280/280** both ways. "The answer fits this question" and "the
answer is correct" are **the same variable**. So C6's "verdict" cannot be distinguished from a
*correctness* verdict. Plan §9's claim table requires them separable.
The lever exists and is unused: `panl score` found **24 alias cells** (right entity, surface form
the model does not itself produce).

### 4.3 The read-out is saturated *(not fixed)*
Across all 240 facts: median gold margin **+20.6 logits**; only **2%** have |margin| < 10.
Pre-fix behavioural run: AUC **1.000**, d_z **7.0**. The model knows essentially every fact cold.

---

## 5. Claims we **cannot** make

| Plan §1 requirement | Status |
|---|---|
| **Identification** — a non-additive Q×A contrast | ⚠️ **Trivial by construction.** `I_k = 2 × (matched − crossed)` by definition, and the additive null (`z = f(Q) + g(A)`) is a straw man nobody believes. |
| **Selectivity** — separable from likelihood and commitment | ❌ **Zero evidence.** The likelihood controls were broken (§4.1) and have not been re-run. Commitment has never been touched. *This is the core of the original paper.* |
| **Generalization** — held-out identities and families | ❌ **Not run.** Everything is on `train`. |
| **Causal use** of a relational component | ❌ **Refuted.** PANL is not necessary (C2), and what it carries is not relational (C6). |

**5.4 The main threat to C6.** A reviewer will say: *"Your items are trivially easy (§4.3). On
easy items the model resolves to a scalar immediately; on hard items PANL might retain more."*
**They would be right, and we cannot currently answer them.** Two responses, in order of cost:

- *Cheap (30 min, no GPU):* stratify C6 by each block's clean gap and check whether
  pair-specificity holds across strata. If it holds even in the lowest stratum, C6 is much
  stronger. If specificity *appears* at the hard end, **that is a better finding** and must be
  reported.
- *Expensive (1–2 days):* rebuild the fact base for difficulty and re-run.

---

## 6. Reproduction

```bash
uv run panl data build   --config configs/data/tier1.yaml
uv run panl score        --config configs/experiment/e0_qwen7b.yaml --out outputs/scores_qwen7b.parquet
uv run panl e0 routes    --config configs/experiment/e0_qwen7b.yaml --run-id routes-fixed
uv run panl e0 controls  --config configs/experiment/e0_qwen7b.yaml --start-layer 16 --run-id controls-decisive

# Re-render either report from saved parquet. No GPU, no model.
uv run panl e0 routes-report outputs/routes-fixed     --config configs/experiment/e0_qwen7b.yaml
```

Every run writes a manifest pinning the git commit, the `uv.lock` hash, the config hash, and a
checksum of each artifact.

| artifact | what it is |
|---|---|
| `outputs/routes-fixed/` | C1, C2, C3, C4, C5(a) — the corrected route ablation |
| `outputs/controls-decisive/` | C3, C5(b), C6 — source and direction controls |
| `outputs/scores_qwen7b.parquet` | §4.1, §4.3 — on-policy rate, item difficulty, 23 model errors, 24 aliases |
| `outputs/e0-primary-train/` | M1, M4, M5 — the superseded patching sweep, kept as evidence |

---

## 7. What a submission needs that we do not have

Ranked by whether the paper stands without it.

| gap | cost | consequence if skipped |
|---|---|---|
| **Gemma-3-27B replication** | ~4 h GPU (tokenizer verified, config ready) | **Fatal.** n = 1 model, 1 prompt. A reviewer stops here. |
| Held-out (validation split) | ~1 h | Weakens every claim; nothing is fitted, so this is cheap insurance. |
| C6 stratified by item difficulty | 30 min, no GPU | Leaves §5.4 unanswered. |
| Re-measure calibration under the fixed prompt | ~15 min GPU | §4.3's AUC is from the pre-fix prompt. |

**Honest assessment.** What we have is a *methods + negative-result* paper (C1–C6 + M1–M6), not
the mechanism paper the plan set out to write. Three of the plan's four requirements have no
data, and the fourth is refuted. That is a legitimate contribution — plan §9 anticipated it —
but it must be written as what it is, and it needs the Gemma replication to exist at all.
