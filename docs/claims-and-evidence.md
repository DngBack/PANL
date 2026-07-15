# Claims and evidence

**Models:** `Qwen/Qwen2.5-7B-Instruct` (28 layers) and `google/gemma-3-27b-it` (62 layers) —
different families, different tokenizers, different depths. Both bf16 via TransformerLens.
**Data:** 140 blocks / 560 cells, `train` split of `data/processed/quadruples.parquet`
**Statistics:** percentile bootstrap over **blocks** (10,000 resamples). The block, not the cell,
is the resampling unit.
**Date:** 2026-07-15 · **Tests:** 198 CPU + 9 GPU · `ruff` and `mypy --strict` clean

This is the audit trail. Every claim is stated with its evidence, its artifact, and **what would
falsify it**. Claims we *cannot* make are in §5, and they are most of the original research plan.

---

## 0. Summary

**Every claim below replicates on both models.** The headline:

> The post-answer confidence cache is real, causally sufficient, and **transplantable**: freezing
> PANL under route isolation moves the confidence margin the full width of the gap and flips
> 95–98% of high/low decisions, in **both** directions, while a norm-matched random vector does
> nothing (0–1% in ablate) and the control positions do nothing (≤1%).
>
> But the cache is **redundant** — the direct `answer → CC` route alone preserves ~70% of the gap
> — and what it stores is a **pair-independent verdict, not a Q×A relation**: a crossed cell from
> an *unrelated block* removes confidence as well as the true partner does (97–99% vs 96–98%).

That last sentence is the answer to the original research question, and it is **no**.

---

## 1. What is measured

The prompt ends `Answer: <answer>\nConfidence:`; we read the pre-softmax margin
`z = logit(" high") − logit(" low")` at the final token.

```
... Answer :        <answer tokens>      \n      Conf  idence   :     ← Qwen: 2 suffix tokens
        AC                    ... LAT   PANL    PANL1   (—)     CC
... Answer :        <answer tokens>      \n    Confidence        :     ← Gemma: 1 suffix token
```

A **block** is a 2×2 — two questions, each one's gold answer, crossed — and the **gap** is its
matched-minus-crossed margin. Note `I_k = z11 − z12 − z21 + z22` is exactly twice the gap: in this
design the "interaction" and the "matched-vs-crossed contrast" are the same number (§5).

---

## 2. Claims

### C1 — The route ablation is valid

Severing attention from every query that can see the answer drives the gap to **zero**. No route
is unaccounted for.

| | Qwen2.5-7B | Gemma-3-27B |
|---|---|---|
| clean gap | +29.66 [+29.10, +30.17] | +34.80 [+34.48, +35.13] |
| **cut everything** | **−0.03** [−0.11, +0.06] | **−0.05** [−0.13, +0.03] |

Also 0% on the 48 length-matched blocks, so it is not a rotary/position artefact.

**Why it matters.** This is the precondition for everything else. The *first* version of this
ablation left a 19% residual that we nearly wrote up as a "floor". It was a leak (M2).

**Falsified by:** a nonzero gap here.

---

### C2 — The answer reaches the read-out by **redundant** routes

PANL and the direct edge are each individually sufficient. **Neither is necessary.**

| condition | Qwen | Gemma |
|---|---|---|
| **only via PANL** (`answer → PANL → … → CC`) | **97%** | **93%** |
| **only direct** (`answer → CC`) | **71%** | **68%** |
| cut `PANL ← answer` alone | 79% | 74% |

Cutting any single route costs almost nothing. Cutting all of them costs everything.

**Why it matters.** This is why patching PANL in the intact model reports a null (M1): the direct
route re-supplies whatever the patch removes. **A single-position patch cannot localize a
position that has a bypass, whatever that position contains.**

---

### C3 — PANL is a sufficient, bidirectional carrier

With the bypasses severed, freezing PANL's residual trajectory transplants the decision — both
ways — and neither control position does.

| | Qwen (L16..end) | Gemma (L24..end) |
|---|---|---|
| **PANL** peak effect | **+1.002** | **+1.001** |
| logits moved (of the gap) | **+28.89 / 28.9** | **+32.31 / 32.3** |
| flip: restore / ablate | **95% / 96%** | **98% / 98%** |
| PANL+1 control (max flip) | ≤1% | ≤1% |
| AC control (max flip) | 0% | 0% |

A **complete** transplant. Self-patching a row with its own activation is an **exact** no-op, so
none of this is an artefact of the intervention machinery.

---

### C4 — CC reads PANL at a consistent *relative* depth

Freezing PANL from the cliff layer onward works; a few layers later does nothing, because by then
the read has already happened.

| | cliff | depth |
|---|---|---|
| Qwen (28 layers) | **L16** | 57% |
| Gemma (62 layers) | **L24** | 39% |

**Reading the cliff correctly matters.** With a cumulative patch, every start layer *below* the
read point ties at the peak — a span from L4 contains the span from L16 — so `argmax` reports an
arbitrary member of the tie (L4) and says nothing. The informative number is the **last** start
layer that still works (`routes.read_cliff`).

---

### C5 — The model is confident **by default**; PANL carries **doubt**

**(a) Cut every route** — both conditions converge on *confident* (Qwen):

| | matched | crossed |
|---|---|---|
| clean | +19.29 | **−10.37** |
| cut everything | +15.07 | **+15.09** |

**(b) Write noise into PANL** — a norm-matched random direction:

| | restore (target = crossed) | ablate (target = matched) |
|---|---|---|
| **Qwen** | **96%** flip | **1%** flip |
| **Gemma** | **98%** flip | **0%** flip |

**Why this is the most useful finding in the document.** The **restore direction is confounded
for every destructive intervention**. A study that patches a confidence cache and reports only
the restore direction is reporting an effect that *noise reproduces*. **Our own route report did
exactly that** until the controls were run.

---

### C6 — What is cached is a **verdict**, not a **relation**

**This is the answer to the original research question, and it is a no.**

The signal PANL carries is not specific to the question–answer pair. A crossed cell from an
**unrelated block** — different question, different answer — removes confidence as well as the
true partner. Ablate direction (the only uncontaminated one):

| source | Qwen: effect / flip | Gemma: effect / flip |
|---|---|---|
| **partner** (same block — the real intervention) | +0.993 / **96%** | +0.999 / **98%** |
| **crossed donor** (different block) | **+0.997 / 97%** | **+1.004 / 99%** |
| gaussian (noise) | +0.005 / 1% | +0.094 / 0% |
| mean-ablation | +0.520 / 5% | +0.276 / 0% |
| matched donor (different block) | −0.013 / 1% | −0.003 / 0% |

Symmetrically in restore: matched donor 96% (Qwen) / 98% (Gemma) vs partner 95% / 98%.

`random_cell` (≈49% ablate on both) decides nothing: its donor pool is 50/50 matched/crossed, so
~50% is exactly what a binary signal predicts. Only the **type-pure** donors decide it — and they
decide it flatly, on both models.

**Interpretation.** The relational computation — *does this answer fit this question* — happens
**upstream** of PANL. What is written into PANL is only its scalar outcome. **The cache stores
the verdict, not the reasoning.**

**Robustness to item difficulty.** Stratifying blocks into difficulty terciles by clean gap, the
crossed donor tracks the true partner in **every** stratum (Qwen):

| stratum | blocks | clean gap | partner | **crossed donor** | gaussian |
|---|---|---|---|---|---|
| hard | 48 | 25.4 | 88% | **94%** | 4% |
| medium | 46 | 29.7 | 100% | **98%** | 0% |
| easy | 46 | 31.5 | 100% | **99%** | 0% |

So C6 is not an artefact of the easy end of *our* data. **It does not fully answer §5.4:** even
the "hard" tercile has a 25-logit gap. We have no items the model is genuinely unsure about, and
that is a limitation of the stimuli, not of the analysis.

**Falsified by:** a crossed donor from another block failing to ablate.

---

## 3. Methodological findings — each cost us a wrong result first

Model-agnostic, dataset-independent, and **none of them crashed anything**. Each produced a
plausible, publishable-looking number that was wrong.

### M1 — Redundant routes make single-position patching report **false nulls**
Patching PANL in the intact model: **+3.01 of 30 logits, 0% of decisions flipped.** It looks like
PANL is empty. It is not (C3): the same patch, with the bypass severed, flips **96%**.

### M2 — Never define an intervention by **naming tokens**
Qwen splits `"Confidence"` into `"Conf"` + `"idence"`. The second token has **no name** in our
position schema, sits at CC−1, sees the answer, and CC reads it. Every knockout leaked through it;
`only via PANL` was not isolating PANL. **The "19% floor" we were about to explain away as a
rotary artefact was this leak.**

> **The controlled demonstration:** Gemma tokenizes `"Confidence"` as **one** token. The *same
> name-based code* is correct on Gemma and leaks on Qwen — silently, with no error, producing a
> plausible number on both. Interventions must address position *sets* (`after_answer`,
> `post_panl`, `suffix_before_cc`), never token names.

### M3 — A single-layer patch leaks when the position keeps reading its source
`ISOLATE_PANL` deliberately leaves `PANL ← answer` open — PANL must see the answer or it could
carry nothing. So a one-layer patch is re-supplied at every later layer. Freezing the trajectory
closes it: **0.42 → 1.00**, 50% → 96%.

### M4 — The gap-normalized metric conflates patch strength with **saturation**
The same patch moves **more** in absolute logits on 7B (+3.01) than on 0.5B (+1.21), yet scores
0.10 vs 0.92 — because 7B's clean gap is 23× larger. Report absolute shift and flip rate
alongside any normalized effect.

### M5 — Reporting one direction gives **false positives**
See C5. Noise flips 96–98% of *restore* decisions on both models.

### M6 — Exploratory subsets lie
A 60-block exploratory run reported a 76% flip rate. On the full 140-block split: 50%.

---

## 4. Data-design defects found by running the model

### 4.1 The prompt was off-policy *(fixed)*
The model does not say `Paris`; it says *"The capital of France is Paris."* Teacher-forcing our
gold was an **off-policy trajectory**, so the answer's log-probability — the nuisance variable the
entire fluency control of plan §4.3 rests on — was measuring *surprise at our formatting*, not
knowledge. **Residualizing on it would have controlled for nothing.** Fixed with a format-strict
system prompt: on-policy **0% → 90%**, verified by `panl score`, which decodes the model's own
answer and compares it to gold.

### 4.2 `matched` and `correct` are perfectly confounded *(not fixed)*
Crosstab exactly diagonal: **280/280** both ways. "The answer fits this question" and "the answer
is correct" are **the same variable**, so C6's "verdict" cannot be distinguished from a
*correctness* verdict. Plan §9's claim table requires them separable. The lever exists and is
unused: `panl score` found **24 alias cells** (right entity, surface form the model does not
itself produce).

### 4.3 The read-out is saturated *(not fixed)* — **the main limitation**
Re-measured under the **fixed** prompt (Qwen, 140 train blocks):

| | |
|---|---|
| interaction `I_k` | **+59.32** [+58.19, +60.34], sign 100%, permutation *p* = 0.0001 |
| **calibration AUC** | **1.0000** |
| paired effect size `d_z` | **9.01** |
| mean margin | matched **+19.29**, crossed **−10.37** |
| cells with \|margin\| < 5 logits | **9 / 560** |

Across all 240 facts: median gold margin **+20.6**; only **2%** have \|margin\| < 10. The model
knows essentially every fact cold and separates right from wrong **perfectly**.

---

## 5. Claims we **cannot** make

| Plan §1 requirement | Status |
|---|---|
| **Identification** — a non-additive Q×A contrast | ⚠️ **Trivial by construction.** `I_k = 2 × (matched − crossed)` *by definition*, and the additive null (`z = f(Q) + g(A)`) is a straw man. |
| **Selectivity** — separable from likelihood and commitment | ❌ **Zero evidence.** The likelihood controls were broken (§4.1) and have not been re-run. Commitment has never been touched. *This is the core of the original paper.* |
| **Generalization** — held-out identities and families | ❌ **Not run.** Everything is on `train`. |
| **Causal use of a relational component** | ❌ **Refuted.** PANL is not necessary (C2), and what it carries is not relational (C6). |

**5.4 The standing threat to C6.** *"Your items are trivially easy (§4.3). On easy items the model
resolves to a scalar immediately; on hard items PANL might retain more."* The difficulty
stratification (C6) answers this **within our range** and not beyond it, because our range contains
no genuinely uncertain items. Honest response: state it. Fixing it means rebuilding the fact base
for difficulty (1–2 days) and re-running.

---

## 6. Artifacts and reproduction

```bash
uv run panl data build   --config configs/data/tier1.yaml
uv run panl score        --config configs/experiment/e0_qwen7b.yaml  --out outputs/scores_qwen7b.parquet
uv run panl e0 routes    --config configs/experiment/e0_qwen7b.yaml   --run-id routes-fixed
uv run panl e0 controls  --config configs/experiment/e0_qwen7b.yaml   --start-layer 16 --run-id controls-qwen
uv run panl e0 routes    --config configs/experiment/e0_gemma27b.yaml --run-id gemma-routes  --layer-step 4
uv run panl e0 controls  --config configs/experiment/e0_gemma27b.yaml --start-layer 24 --run-id gemma-controls

# Re-render any report from saved parquet. No GPU, no model.
uv run panl e0 routes-report outputs/routes-fixed --config configs/experiment/e0_qwen7b.yaml
```

Every run writes a manifest pinning the git commit, the `uv.lock` hash, the config hash, and a
checksum of each artifact.

| artifact | supports |
|---|---|
| `outputs/routes-fixed/`, `outputs/gemma-routes/` | C1, C2, C3, C4, C5(a) |
| `outputs/controls-qwen/`, `outputs/gemma-controls/` | C3, C5(b), C6 |
| `outputs/scores_qwen7b.parquet` | §4.1, §4.3 — on-policy rate, difficulty, 23 model errors, 24 aliases |
| `outputs/behavior_fixed_prompt.parquet` | §4.3 — calibration under the fixed prompt |
| `outputs/e0-primary-train/` | M1, M4, M5 — the superseded patching sweep, kept as evidence |

---

## 7. Assessment for AAAI-27

**What this is:** a **methods + negative-result** paper. It is *not* the mechanism paper the plan
set out to write — three of the plan's four requirements have no data and the fourth is refuted.
Plan §9 anticipated exactly this outcome and licenses it.

**What makes it submittable:**
- **n = 2 models**, different families, tokenizers, and depths. Every claim replicates.
- **Six methodological traps**, three with hard controlled evidence: M1 (same patch, 0% → 96%),
  M2 (the Qwen/Gemma tokenizer contrast), M5 (noise flips 96–98% in one direction).
- A clean positive result (C1–C5) and a clean negative one (C6), both replicated.

**What must be in Limitations, not buried:** AUC = 1.0000 (§4.3); `matched ≡ correct` (§4.2);
train split only; and the fact that a patch-under-isolation result is evidence about what a
position *carries*, not proof that the intact model *routes through it* — we forced the model to
use PANL and then observed that it does.

**Still missing (cheap):** held-out validation split (~1 h).
