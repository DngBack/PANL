"""Position resolution against the real Qwen tokenizer, plus the frozen snapshot.

The unit tests prove the resolver's logic with a double. These prove the *prompt* survives
the tokenizer we actually plan to run -- which is the Jul 14-15 kill gate. If Qwen2's
pre-tokenizer merged the post-answer newline into the following word, PANL would not exist
as a position and the experiment design would need changing before any GPU time is booked.

The snapshot then freezes what we found, so a transformers upgrade or a tokenizer revision
bump that shifts tokenization fails here on CPU instead of silently moving PANL under a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from panl.models.confidence import resolve_confidence_classes
from panl.models.positions import POSITION_NAMES, resolve_positions
from panl.models.prompts import PromptRenderer, PromptStyle, PromptTemplate
from panl.models.snapshot import SNAPSHOT_CASES, build_snapshot
from panl.models.tokenizer import FastTokenizer, load_tokenizer

pytestmark = pytest.mark.tokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SNAPSHOT_PATH = Path(__file__).parent.parent / "fixtures" / "positions_qwen2_5_0_5b_instruct.json"


@pytest.fixture(scope="module")
def tokenizer() -> FastTokenizer:
    try:
        return load_tokenizer(MODEL_ID)
    except Exception as exc:
        pytest.skip(f"{MODEL_ID} tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def renderer(tokenizer: FastTokenizer) -> PromptRenderer:
    return PromptRenderer(tokenizer, template=PromptTemplate(), style=PromptStyle.CHAT)


class TestRealTokenizer:
    @pytest.mark.parametrize(("question", "answer"), SNAPSHOT_CASES)
    def test_every_case_resolves(
        self, tokenizer: FastTokenizer, renderer: PromptRenderer, question: str, answer: str
    ) -> None:
        prompt = renderer.render(question, answer)
        resolved = resolve_positions(tokenizer, prompt)
        assert set(resolved.indices) == set(POSITION_NAMES)

    def test_panl_is_a_bare_newline_token(
        self, tokenizer: FastTokenizer, renderer: PromptRenderer
    ) -> None:
        """The single assumption the whole design rests on."""
        prompt = renderer.render("What is the capital of France?", "Paris")
        resolved = resolve_positions(tokenizer, prompt)
        panl_id = resolved.token_ids["PANL"]
        assert tokenizer.convert_ids_to_tokens([panl_id]) == ["Ċ"]  # byte-level BPE for "\n"

    def test_cc_is_the_last_token(self, tokenizer: FastTokenizer, renderer: PromptRenderer) -> None:
        prompt = renderer.render("What is the capital of France?", "Paris")
        resolved = resolve_positions(tokenizer, prompt)
        assert resolved.indices["CC"] == resolved.n_tokens - 1

    def test_an_accented_answer_does_not_break_lat(
        self, tokenizer: FastTokenizer, renderer: PromptRenderer
    ) -> None:
        """ "Reykjavík" splits into byte-level pieces; LAT must still be the final one."""
        prompt = renderer.render("Name the capital city of Iceland.", "Reykjavík")
        resolved = resolve_positions(tokenizer, prompt)
        assert resolved.indices["PANL"] == resolved.indices["LAT"] + 1

    def test_crossed_and_matched_cells_share_a_position_layout(
        self, tokenizer: FastTokenizer, renderer: PromptRenderer
    ) -> None:
        """A crossed cell must differ from its matched twin only in the answer, so any
        confidence difference cannot be a byproduct of a different prompt shape."""
        matched = resolve_positions(
            tokenizer, renderer.render("What is the capital of France?", "Paris")
        )
        crossed = resolve_positions(
            tokenizer, renderer.render("What is the capital of France?", "Tokyo")
        )
        assert matched.indices == crossed.indices

    def test_confidence_classes_are_single_tokens(self, tokenizer: FastTokenizer) -> None:
        classes = resolve_confidence_classes(tokenizer)
        assert classes.high_token_id != classes.low_token_id

    def test_the_confidence_word_is_more_than_one_token_here(
        self, tokenizer: FastTokenizer, renderer: PromptRenderer
    ) -> None:
        """Documents the fact that broke the route ablation, so nobody re-assumes otherwise.

        Qwen tokenizes "Confidence" as "Conf" + "idence". So the span between PANL and CC is
        *two* tokens, only the first of which (PANL+1) has a name. Any intervention that
        enumerates the post-answer tokens by name will miss the other one -- it can see the
        answer, CC reads it, and the route stays open. Interventions must address the span,
        not the names: see `_position_mask("suffix_before_cc")`.
        """
        prompt = renderer.render("What is the capital of France?", "Paris")
        resolved = resolve_positions(tokenizer, prompt)
        span = resolved.indices["CC"] - resolved.indices["PANL"] - 1
        assert span >= 1
        if span == 1:
            pytest.skip("this tokenizer emits a single confidence-word token; nothing to guard")
        unnamed = set(range(resolved.indices["PANL"] + 1, resolved.indices["CC"])) - {
            resolved.indices["PANL1"]
        }
        assert unnamed, "expected at least one unnamed token between PANL+1 and CC"


@pytest.fixture(scope="module")
def stored() -> dict[str, Any]:
    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"no frozen snapshot at {SNAPSHOT_PATH}. Regenerate it with:\n"
            f"  uv run panl positions snapshot --model {MODEL_ID} --out {SNAPSHOT_PATH}"
        )
    data: dict[str, Any] = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return data


class TestFrozenSnapshot:
    def test_snapshot_still_matches(self, tokenizer: FastTokenizer, stored: dict[str, Any]) -> None:
        current = build_snapshot(tokenizer, model_id=MODEL_ID, style=PromptStyle.CHAT)
        assert current["template_hash"] == stored["template_hash"], (
            "the prompt template changed; positions and prompt hashes are no longer comparable "
            "with anything measured before this change"
        )
        assert current["confidence_classes"] == stored["confidence_classes"]
        assert current["cases"] == stored["cases"], (
            "tokenization drifted: the resolved positions or prompt hashes no longer match the "
            "frozen snapshot. Check the transformers/tokenizer version before trusting any run."
        )
