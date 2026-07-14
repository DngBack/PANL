"""The adapter against a real model.

These are the checks that cannot be faked: that a hook actually fires, that a patch lands on
the token we think it lands on, and that batching does not perturb a single logit. Run on the
0.5B smoke model, which is what it exists for.
"""

from __future__ import annotations

import pytest
import torch

from panl.models.adapter import HookedModelAdapter
from panl.models.batching import make_batches
from panl.models.positions import POSITION_NAMES, ResolvedPositions, resolve_positions
from panl.models.prompts import PromptRenderer, PromptStyle, PromptTemplate
from panl.models.spec import ModelSpec

pytestmark = [pytest.mark.tokenizer, pytest.mark.gpu]

CASES = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of France?", "Tokyo"),
    ("What is the official currency of Japan?", "Japanese yen"),
    ("Who wrote Nineteen Eighty-Four?", "George Orwell"),
]


@pytest.fixture(scope="module")
def model() -> HookedModelAdapter:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        return HookedModelAdapter.load(
            ModelSpec(model_id="Qwen/Qwen2.5-0.5B-Instruct", role="smoke")
        )
    except Exception as exc:
        pytest.skip(f"could not load the smoke model: {exc}")


@pytest.fixture(scope="module")
def resolved(model: HookedModelAdapter) -> list[ResolvedPositions]:
    renderer = PromptRenderer(model.tokenizer, template=PromptTemplate(), style=PromptStyle.CHAT)
    return [resolve_positions(model.tokenizer, renderer.render(q, a)) for q, a in CASES]


def _batch(resolved: list[ResolvedPositions], rows: list[int]):  # type: ignore[no-untyped-def]
    subset = [resolved[i] for i in rows]
    batches = list(make_batches(subset, max_batch_size=len(rows)))
    assert len(batches) == 1, "test rows must share a token length"
    return batches[0]


class TestForward:
    def test_margins_are_finite(
        self, model: HookedModelAdapter, resolved: list[ResolvedPositions]
    ) -> None:
        for batch in make_batches(resolved, max_batch_size=4):
            margins = model.run(batch).confidence_margin
            assert torch.isfinite(margins).all()

    def test_matched_beats_crossed_on_a_known_fact(
        self, model: HookedModelAdapter, resolved: list[ResolvedPositions]
    ) -> None:
        """France/Paris versus France/Tokyo. If this ordering fails there is no confidence
        signal to study and E0 has nothing to reproduce."""
        margins = {}
        for batch in make_batches(resolved[:2], max_batch_size=1):
            margins[batch.row_indices[0]] = float(model.run(batch).confidence_margin[0])
        assert margins[0] > margins[1]

    def test_batching_is_semantically_exact(self) -> None:
        """The justification for length-grouped batches: batching must not change the
        computation, only how it is scheduled.

        Checked in float32, because that is where the claim is actually testable. In bfloat16
        the batched and solo runs differ by ~0.1 logits -- not from any masking or position
        error but because the batch size steers cuBLAS to a different GEMM kernel, whose
        accumulation order rounds differently in 8 mantissa bits. float32 has the headroom to
        show that difference for what it is: 5e-5, i.e. nothing.
        """
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        precise = HookedModelAdapter.load(
            ModelSpec(model_id="Qwen/Qwen2.5-0.5B-Instruct", role="smoke", dtype="float32")
        )
        renderer = PromptRenderer(
            precise.tokenizer, template=PromptTemplate(), style=PromptStyle.CHAT
        )
        items = [resolve_positions(precise.tokenizer, renderer.render(q, a)) for q, a in CASES]

        solo = {}
        for index, item in enumerate(items):
            batch = next(iter(make_batches([item], max_batch_size=1)))
            solo[index] = float(precise.run(batch).confidence_margin[0])

        for batch in make_batches(items, max_batch_size=4):
            margins = precise.run(batch).confidence_margin
            for offset, row in enumerate(batch.row_indices):
                assert abs(float(margins[offset]) - solo[row]) < 1e-3

    def test_bf16_batching_drift_is_far_below_any_real_effect(
        self, model: HookedModelAdapter, resolved: list[ResolvedPositions]
    ) -> None:
        """bfloat16 is the compute dtype for every real run, so bound its drift explicitly.

        The threshold is deliberately far below the smallest effect we would ever report: the
        confidence gaps in these experiments are 1-30 logits. A masking or index bug would
        move the margin by that much, not by a rounding step.
        """
        solo = {}
        for index, item in enumerate(resolved):
            batch = _batch(resolved, [index])
            solo[index] = float(model.run(batch).confidence_margin[0])

        for batch in make_batches(resolved, max_batch_size=4):
            margins = model.run(batch).confidence_margin
            for offset, row in enumerate(batch.row_indices):
                assert abs(float(margins[offset]) - solo[row]) < 0.5

    def test_activations_have_the_expected_shape(
        self, model: HookedModelAdapter, resolved: list[ResolvedPositions]
    ) -> None:
        batch = _batch(resolved, [0])
        out = model.run(batch, cache_layers=[0, 5])
        assert set(out.activations) == {0, 5}
        for layer in (0, 5):
            assert set(out.activations[layer]) == set(POSITION_NAMES)
            assert out.activations[layer]["PANL"].shape == (1, model.d_model)

    def test_answer_logprobs_cover_exactly_the_answer_tokens(
        self, model: HookedModelAdapter, resolved: list[ResolvedPositions]
    ) -> None:
        batch = _batch(resolved, [2])  # "Japanese yen", a multi-token answer
        out = model.run(batch)
        assert len(out.answer_logprobs[0]) == resolved[2].n_answer_tokens
        assert (out.answer_logprobs[0] <= 0).all()


class TestPatching:
    def test_patching_a_run_with_its_own_activations_changes_nothing(
        self, model: HookedModelAdapter, resolved: list[ResolvedPositions]
    ) -> None:
        """The strongest available check on the patch machinery. If the index, the hook, or
        the batch alignment were wrong, writing a row's own value back would still perturb
        the output. It must be exactly a no-op."""
        batch = _batch(resolved, [0])
        layer = model.n_layers // 2
        clean = model.run(batch, cache_layers=[layer])

        for position in POSITION_NAMES:
            source = clean.activations[layer][position]
            patched = model.run_with_patch(batch, layer=layer, position=position, source=source)
            assert float(patched[0]) == float(clean.confidence_margin[0]), position

    def test_patching_cc_transplants_the_confidence(
        self, model: HookedModelAdapter, resolved: list[ResolvedPositions]
    ) -> None:
        """CC at the last layer is the read-out state itself, so moving it across must carry
        the source's margin almost exactly. This is the harness check, not a finding."""
        source_batch = _batch(resolved, [0])  # Paris, confident
        target_batch = _batch(resolved, [1])  # Tokyo, not confident
        layer = model.n_layers - 1

        source = model.run(source_batch, cache_layers=[layer])
        target_clean = float(model.run(target_batch).confidence_margin[0])
        source_clean = float(source.confidence_margin[0])

        patched = float(
            model.run_with_patch(
                target_batch,
                layer=layer,
                position="CC",
                source=source.activations[layer]["CC"],
            )[0]
        )
        moved = (patched - target_clean) / (source_clean - target_clean)
        assert 0.9 <= moved <= 1.1

    def test_patching_panl_moves_confidence_toward_the_source(
        self, model: HookedModelAdapter, resolved: list[ResolvedPositions]
    ) -> None:
        source_batch = _batch(resolved, [0])
        target_batch = _batch(resolved, [1])
        layer = model.n_layers // 2

        source = model.run(source_batch, cache_layers=[layer])
        target_clean = float(model.run(target_batch).confidence_margin[0])
        patched = float(
            model.run_with_patch(
                target_batch,
                layer=layer,
                position="PANL",
                source=source.activations[layer]["PANL"],
            )[0]
        )
        assert patched > target_clean

    def test_an_unknown_layer_is_refused(
        self, model: HookedModelAdapter, resolved: list[ResolvedPositions]
    ) -> None:
        """A hook name that does not exist would patch nothing and report a null effect."""
        batch = _batch(resolved, [0])
        with pytest.raises(KeyError, match="is not a hook"):
            model.run_with_patch(
                batch,
                layer=model.n_layers + 5,
                position="PANL",
                source=torch.zeros(1, model.d_model),
            )
