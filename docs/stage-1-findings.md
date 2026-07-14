# Stage 1: infrastructure, E0, and what E0 changed

**Dates:** 2026-07-14 to 2026-07-15
**Models:** `Qwen/Qwen2.5-0.5B-Instruct` (smoke), `Qwen/Qwen2.5-7B-Instruct` (primary)
**Hardware:** 2x NVIDIA H200 (143 GB each)
**Status:** infrastructure complete; E0 run; the experimental design changed as a result.

This document records what was built, what was measured, and — most importantly — the three
things the original design got wrong. Those errors are the substance of this stage. Each was
found only by running the experiment, and each would have silently corrupted E1–E3.

---

## 1. Headline

The research premise **survives**, but the claim it supports is not the one the plan set out
to make.

> On Qwen2.5-7B, PANL genuinely carries the question–answer information that the verbal
> confidence read-out consumes. When PANL is made the only route from the answer to the
> confidence colon, freezing it from layer 16 onward moves the confidence margin by **+26.0 of
> a possible 28.9 logits (0.90)** and flips **96%** of high/low decisions, while the PANL+1
> control flips **0%**.
>
> But PANL is **not necessary**. The answer also reaches the read-out by a direct route, and
> that route alone preserves **88%** of the confidence gap. The cache exists; it is not a
> bottleneck.

All numbers below are on 140 train blocks (560 cells) with block-level bootstrap CIs.

"PANL is a sufficient but redundant carrier" is a sharper and more defensible claim than
"PANL is a cache the model uses", and it directly refines Kumaran et al. It is also the only
claim the evidence actually licenses (§5).

---

## 2. What was built

| Component | Module | Purpose |
|---|---|---|
| Data contract | `panl.data.schema` | Frozen Arrow schema for the crossed-block table |
| Tier-1 fact base | `panl.data.facts` | 240 functional facts across 4 relation families |
| Block builder | `panl.data.blocks` | 2x2 crossed blocks; crossed cells provably wrong |
| Splits | `panl.data.splits` | Fact-level, family-stratified, identity-disjoint |
| Validator | `panl.data.validate` | Every invariant of plan §7, each with a fault-injection test |
| Position resolver | `panl.models.positions` | AC/LAT/PANL/PANL+1/CC from real tokenization |
| Model adapter | `panl.models.adapter` | Thin wrapper over TransformerLens; patching + attention knockout |
| Batching | `panl.models.batching` | Length-grouped; **no padding is ever introduced** |
| Activation store | `panl.activations.store` | Zarr, `[row, layer, position, d_model]`, float16 |
| Statistics | `panl.analysis.stats` | Block-level bootstrap, sign-flip permutation, AUC |
| Scoring | `panl.experiments.scoring` | On-policy check + item difficulty, before any block is built |
| E0 | `panl.experiments.e0` | Confidence contrast + patching sweep + gates |
| Route ablation | `panl.experiments.routes` | The corrected localization experiment |

181 tests (172 CPU, 9 GPU). `ruff` clean, `mypy --strict` clean across 51 files.

### CLI

```bash
uv run panl data build --config configs/data/tier1.yaml
uv run panl data validate data/processed/quadruples.parquet
uv run panl positions check --model Qwen/Qwen2.5-7B-Instruct
uv run panl score --config configs/experiment/e0_qwen7b.yaml --out outputs/scores_qwen7b.parquet
uv run panl e0 run --config configs/experiment/e0_qwen7b.yaml       # patching sweep (superseded)
uv run panl e0 routes --config configs/experiment/e0_qwen7b.yaml    # route ablation (primary)

# Re-render either report from saved parquet. No GPU, no model: a change to how a result is
# summarized must never cost GPU time twice.
uv run panl e0 report        outputs/<run> --config configs/experiment/e0_qwen7b.yaml
uv run panl e0 routes-report outputs/<run> --config configs/experiment/e0_qwen7b.yaml
```

Artifacts: `outputs/routes-primary-v2/` (the result above), `outputs/e0-primary-train/`
(the superseded patching sweep, kept because it is the evidence for §4.3),
`outputs/scores_qwen7b.parquet`.

---

## 3. Verified facts about the setup

These are load-bearing and were each confirmed empirically, not assumed.

- **PANL exists as a token.** Qwen2's pre-tokenizer splits `"\n"` from the following word, so
  the post-answer newline is its own token (`Ċ`). The resolver asserts this rather than
  assuming it; a tokenizer that merged them would raise.
- **Length-grouped batching is semantically exact.** In float32, batched and single-prompt runs
  agree to 5e-5. No padding is ever introduced, so no attention-mask or rotary-offset error is
  possible. *In bfloat16 — the compute dtype for every real run — they differ by up to 0.125
  logits.* That is not a masking error: the batch size steers cuBLAS to a different GEMM
  kernel whose accumulation order rounds differently in 8 mantissa bits, and float32 shows the
  same comparison at 5e-5. It is bounded and it is three orders of magnitude below the smallest
  effect reported here (gaps are 1–30 logits), but it is not zero, and an earlier draft of this
  document wrongly said it was.
- **The patch machinery is correct.** Patching a row with its own activation is an exact
  no-op (delta = 0.0e+00), including under attention knockout. Patching CC — the read-out
  position — restores the source margin at 1.000 with CI [1.000, 1.000].
- **Gemma-3-27B is supported by TransformerLens.** The cross-family replication is viable.
- **The GPU assumption in the plan is stale.** §2 assumed no CUDA and 16 GB RAM. There are two
  H200s. 7B inference peaks at ~19 GB. The "AAAI-27 is high risk without GPU" caveat no longer
  applies for the compute reason (it still applies for the calendar reason — §7).

---

## 4. The three errors E0 found

### 4.1 The prompt was off-policy

The original system prompt did not constrain the answer format. Asked a question, the model
does not say `Paris` — it says:

```
"What is the capital of France?"  →  " The capital of France is Paris."
```

Teacher-forcing `" Paris"` onto that is an **off-policy trajectory**. The consequence is not
cosmetic: the answer's log-probability — the nuisance variable the entire fluency control of
plan §4.3 rests on — was measuring *surprise at our formatting*, not knowledge of the fact.
Median gold NLL was 3.24/token, and 236 of 280 gold answers looked "unlikely" to the model.
Residualizing confidence on that quantity would have controlled for nothing.

**Fixed.** A format-strict system prompt (`panl.models.prompts.DEFAULT_SYSTEM`) makes the
bare entity what greedy decoding already emits. On-policy rate went from **0% → 90%**.

The fix is verified by `panl score`, which decodes the model's own answer and compares it to
gold. Any future change to the prompt must be re-checked the same way.

### 4.2 `matched` and `correct` are perfectly confounded

Verified on the built table: the crosstab is exactly diagonal — every matched cell is correct,
every crossed cell is incorrect, 280/280 both ways.

So "the answer relationally fits this question" and "the answer is correct" are **the same
variable** in this dataset. No analysis can separate a relational-fit component from a
correctness detector, and plan §9's claim-selection table requires exactly that separation.

**Not yet fixed.** The lever exists: `panl score` found items where the model produces the
right entity in a different surface form (`Japanese Yen`, `Reykjavik`). Those are `alias`
cells — matched but not exactly gold — and the schema already carries `answer_source` to hold
them. See §6.

### 4.3 The patching sweep measured the wrong thing

This is the important one.

E0's design patched PANL while the answer tokens were still fully visible to CC. On 7B the
result looked like a clean null: the patch moved 3 of 30 logits and flipped **0%** of
decisions, versus 0.92 normalized effect on 0.5B.

Two explanations were on the table, and both were wrong in an instructive way.

**First read — saturation.** The clean matched-vs-crossed gap on 7B is 29.9 logits (AUC 1.000,
d_z = 7.0); on 0.5B it is 1.3. In absolute terms the *same* patch moves **more** on 7B (+3.01
logits) than on 0.5B (+1.21). The normalized metric divides by the gap, so it conflates *how
hard the patch pushes* with *how steep the read-out is*. That is real, and the metric was
duly fixed — `mean_moved` and `flip_rate` are now reported alongside `effect`, and a
`SATURATED READ-OUT` warning fires above a 10-logit gap.

**But saturation was not the main cause.** Route ablation shows the answer reaches CC by **two
redundant routes** — directly, and through PANL — and *either one alone* carries ~90% of the
confidence gap (table below).

A single-position patch at PANL was therefore **never going to show an effect**, no matter
what PANL contained: whatever the patch removed, the direct route put straight back. The
measurement was confounded by **redundancy**, and it would have reported a null for any
position on any model with a bypass path.

**Route ablation** (140 blocks, block bootstrap):

| condition | gap (logits) | 95% CI | share of clean |
|---|---|---|---|
| clean | +29.66 | [+29.10, +30.17] | 100% |
| cut `CC ← answer` | +28.83 | [+28.23, +29.38] | 97% |
| cut `CC ← PANL` | +29.02 | [+28.43, +29.60] | 98% |
| cut `PANL ← answer` | +23.36 | [+22.76, +23.93] | 79% |
| **only via PANL** | +28.86 | [+28.27, +29.39] | **97%** |
| **only direct** | +26.14 | [+25.50, +26.74] | **88%** |
| cut everything | +5.70 | [+5.49, +5.91] | 19% (floor) |

**Then patch PANL under isolation.** One further subtlety, and it cost a wrong number before
it was caught: `ISOLATE_PANL` deliberately leaves `PANL ← answer` open — PANL must be able to
read the answer, or it could carry nothing. But that means a **single-layer patch leaks**: at
every later layer PANL re-attends to the *target's* answer and re-acquires what the patch
overwrote. Freezing PANL's whole trajectory from layer L onward closes the leak:

| patch span | effect | logits moved | flip rate |
|---|---|---|---|
| PANL, L16..end | **0.902** | **+26.03** | **96%** |
| PANL, L18..end | 0.42 | +12.14 | 50% |
| PANL, L20..end | −0.01 | −0.37 | 1% |
| PANL, single layer (peak, L18) | 0.415 | +11.98 | 50% |
| PANL+1 control | 0.000 | +0.00 | **0%** |
| AC control | 0.000 | +0.00 | 1% |

Self-patch under the same knockout is an exact no-op, so none of this is an artefact of the
intervention machinery.

**The cliff is the localization, and reading it needs care.** With a cumulative patch every
start layer *below* the read point ties at the peak — a span from L4 contains the span from
L16 — so `argmax` reports an arbitrary member of that tie (L4) and says nothing. The
informative number is the far edge: the *last* start layer that still works. That is what
`routes.read_cliff` computes.

```
L4..end  … L16..end   0.90   flip=96%   ██████████████████████████████
L18..end              0.42   flip=50%   ██████████████
L20..end             -0.01   flip= 1%
```

Freezing PANL from L16 onward transplants the decision; starting at L20 does nothing, because
by then the read has already happened. **CC reads PANL at roughly layers 17–20.**

> **A caution recorded so it is not forgotten.** The first exploratory run of this experiment
> used 60 blocks and reported a 76% flip rate for the single-layer patch. On the full 140-block
> train split it is 50%. The subset was not representative. Every number in this document is
> from the full split.

**A fourth, smaller error.** The apparent LAT effect (0.395) is an artefact of answer length.
Split by answer token count: 1 token → **1.018**, 2 tokens → 0.511, 3+ tokens → 0.128. LAT is
not a mechanism; it measures *how much of the answer the patch happened to replace*. For a
single-token answer, LAT **is** the whole answer. Any LAT claim must stratify by answer length.

---

## 5. What the evidence licenses — and what it does not

**Licensed:** PANL carries the Q×A information the confidence read-out consumes. Under
isolation, patching it drives the read-out; matched controls do not. This is a *sufficiency*
result and it is solid.

**Not licensed:** that the intact model routes confidence through PANL. The patch-under-
isolation experiment *forces* the model to use PANL and then observes that it does. Inferring
necessity from that would be inferring necessity from a sufficiency experiment. The direct
route carries 88% of the gap on its own — so the honest statement is that PANL is **sufficient
but redundant**.

This distinction is written into the module docstring of `panl.experiments.routes` so it
cannot quietly erode.

**Also unresolved:** the 19% residual gap under `cut everything`. Matched and crossed prompts
differ in answer token count, so CC sits at a different absolute position — a RoPE/length
confound. It sets the floor against which every other condition should be read. Worth an
explicit length-matched control before publication.

---

## 6. Open problem: the fact base is too easy

`panl score` on 7B over all 240 facts:

| | |
|---|---|
| on-policy (greedy answer = gold entity) | 90% |
| model errors (a different entity) | 23 |
| median confidence margin on the *gold* answer | **+20.6 logits** |
| facts with \|margin\| < 10 logits | **2%** |

The model knows essentially every fact in the base cold. This does **not** threaten the
mechanistic results — route ablation never divides by the clean gap, so saturation cannot
distort it. It **does** threaten the behavioural claims:

- E1 will find an enormous interaction that survives any likelihood control trivially, and a
  reviewer will correctly say the task was too easy to be evidence against `H_fluency`.
- The stress subsets of plan §3.2 (high-LP-wrong, low-LP-correct) cannot be built from items
  the model finds uniformly trivial.

Pairing "near-neighbour" distractors will **not** fix this — the model still knows perfectly
well that Bratislava is not the capital of Austria. What lowers the gap is **item difficulty**:
facts the model genuinely does not know.

---

## 7. Calendar

AAAI-27: abstract **2026-07-21**, paper **2026-07-28**, supplementary **2026-07-31**.

The infrastructure is done, which was the largest chunk. But the full E1→E2→E3 chain plus a
Gemma replication plus writing, solo, in 13 days — after a data redesign — is not realistic.
The abstract deadline is cheap and withdrawable; it is the natural go/no-go point.

---

## 7. Next steps

Ordered, and each is a gate: if it fails, stop and say so rather than working around it.

### Step 1 — Harden the mechanistic result *(≈1 day, needs no new data)*

This is the publishable core and it does **not** depend on fixing the fact base, because route
ablation never divides by the clean gap and so is immune to the saturation problem of §6.

- **Length-matched control for the 19% floor.** Matched and crossed prompts differ in answer
  token count, so CC sits at a different absolute position. Re-run `cut everything` on blocks
  whose two answers tokenize to the same length. If the floor drops to ~0, the residual was
  a RoPE/length artefact; if it does not, there is another route we have not found — and that
  would be the most important open question in the paper.
- **Both directions.** Currently only `restore` (matched source → crossed target) is reported.
  Plan §E3 requires bidirectional evidence; `ablate` is already computed and needs summarizing.
- **Random-subspace and orthogonal-complement controls** at PANL, matched on norm (plan §E3).
  Without them, "patching PANL moves confidence" does not exclude "patching *anything* of that
  magnitude moves confidence".
- **Held-out blocks.** Every number here is on `train`. Nothing has been fitted, so this is
  not leakage — but the test split must be opened once, at the end, not now.
- **Gemma-3-27B replication** of the route ablation only. TL supports it and it fits on one
  H200. If the redundant-route structure does not replicate, the finding is Qwen-specific and
  the paper must say so.

### Step 2 — Rebuild the fact base for difficulty *(1–2 days)*

- Expand with genuinely obscure entities. The current base is 240 facts the model knows cold.
- Promote the **23 model errors** `panl score` already found into high-LP-wrong distractors.
- Add **`alias` cells** to break the `matched` ≡ `correct` confound (§4.2). `panl score` found
  these too (`Japanese Yen`, `Reykjavik`): the right entity in a surface form the model does
  not itself produce. They are matched-but-not-gold, which is exactly the cell type needed.
- Re-run `panl score`; select blocks stratified across the gold-margin range.

**Gate:** at least ~30% of blocks with a clean gap under 10 logits. **If the fact base cannot
produce them, the behavioural arm (E1) is not viable and should be dropped** — say so and ship
the mechanistic arm, which stands on its own.

### Step 3 — Re-run E0 + routes on the difficulty-stratified set *(½ day)*

### Step 4 — Only then E1 → E2 → E3

Every intervention run under route isolation. The E3 "selective interaction-subspace
intervention" in particular must be done with the bypass severed, or it will report a null for
the same reason E0 did.

---

## 8. Recommended reframing

Retarget the paper from *"PANL contains a Q×A component causally used for confidence"* to:

> **The post-answer confidence cache is real but redundant.** PANL carries the question–answer
> information the verbal-confidence read-out consumes, and freezing it transplants the
> read-out's decision in 96% of cases. But the read-out also reads the answer tokens directly,
> and that route alone preserves 88% of the confidence gap — so the cache is not a bottleneck.
> Localizing it requires severing the bypass first; single-position patching reports a null
> regardless of what the position contains.

Three reasons this is the better paper:

1. **It is what the evidence shows.** The original claim asserts causal use by the intact
   model; we have not shown that and, given the redundancy, probably cannot.
2. **It refines rather than repeats** Kumaran et al.: the cache they identify exists, and is
   not load-bearing at 7B scale.
3. **The methodological point is a contribution in itself.** A redundant bypass makes
   single-position activation patching report a null for a position that in fact carries the
   entire signal. This is a trap any patching study of a "cache" will fall into — and this one
   did, twice (once via the bypass, once via the single-layer leak). Both are worth writing up.

## 9. Calendar reality

AAAI-27: abstract **2026-07-21**, paper **2026-07-28**, supplementary **2026-07-31**.

Step 1 alone is a coherent submission under the reframing above and is reachable by the
abstract deadline. The full E1→E2→E3 chain plus a Gemma replication plus writing, solo, in 13
days — after a data rebuild — is not. Treat the abstract deadline as the go/no-go: it is cheap
and withdrawable.

Recommendation: **do Step 1, decide on 07-21 with those results in hand.** If Step 1 is clean,
submit the mechanistic paper and leave E1–E3 for the follow-up. If it is not, skip AAAI-27; the
infrastructure keeps.
