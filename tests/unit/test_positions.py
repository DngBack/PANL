"""Semantic position resolution, including the tokenizations that must be refused."""

from __future__ import annotations

import pytest

from panl.models.confidence import ConfidenceTokenError, resolve_confidence_classes
from panl.models.positions import (
    POSITION_NAMES,
    PositionResolutionError,
    resolve_positions,
)
from panl.models.prompts import PromptRenderer, PromptStyle, PromptTemplate
from panl.models.tokenizer import FastTokenizer
from tests.conftest import FakeTokenizer


def _render(tokenizer: FastTokenizer, style: PromptStyle = PromptStyle.CHAT):  # type: ignore[no-untyped-def]
    return PromptRenderer(tokenizer, template=PromptTemplate(), style=style)


class TestPromptRendering:
    def test_anchors_point_at_the_characters_they_name(self, fake_tokenizer: FastTokenizer) -> None:
        prompt = _render(fake_tokenizer).render("What is the capital of France?", "Paris")
        text, anchors = prompt.text, prompt.anchors

        assert text[anchors.answer_colon] == ":"
        assert text[anchors.answer_start : anchors.answer_end] == "Paris"
        assert text[anchors.newline] == "\n"
        assert text[anchors.confidence_colon] == ":"

    def test_the_prompt_ends_at_the_confidence_colon(self, fake_tokenizer: FastTokenizer) -> None:
        prompt = _render(fake_tokenizer).render("Q?", "A")
        assert prompt.text.endswith("Confidence:")
        assert prompt.anchors.confidence_colon == len(prompt.text) - 1

    def test_raw_style_needs_no_tokenizer(self) -> None:
        prompt = PromptRenderer(None, style=PromptStyle.RAW).render("Q?", "A")
        assert "<|im_start|>" not in prompt.text
        assert prompt.text.endswith("Confidence:")

    def test_chat_style_without_a_tokenizer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs a tokenizer"):
            PromptRenderer(None, style=PromptStyle.CHAT)

    def test_a_template_without_the_anchors_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must end with ':'"):
            PromptTemplate(answer_prefix="Answer")
        with pytest.raises(ValueError, match="must end with ':'"):
            PromptTemplate(confidence_prefix="Confidence")
        with pytest.raises(ValueError, match="single"):
            PromptTemplate(newline="\n\n")

    def test_prompt_hash_tracks_the_text(self, fake_tokenizer: FastTokenizer) -> None:
        renderer = _render(fake_tokenizer)
        a = renderer.render("Q?", "Paris")
        b = renderer.render("Q?", "Tokyo")
        assert a.prompt_sha256 != b.prompt_sha256
        assert a.template_hash == b.template_hash


class TestResolution:
    @pytest.mark.parametrize("style", [PromptStyle.CHAT, PromptStyle.RAW])
    def test_all_five_positions_resolve(
        self, fake_tokenizer: FastTokenizer, style: PromptStyle
    ) -> None:
        prompt = _render(fake_tokenizer, style).render("What is the capital of France?", "Paris")
        resolved = resolve_positions(fake_tokenizer, prompt)
        assert set(resolved.indices) == set(POSITION_NAMES)

    def test_positions_are_strictly_ordered(self, fake_tokenizer: FastTokenizer) -> None:
        prompt = _render(fake_tokenizer).render("What is the capital of France?", "Paris")
        r = resolve_positions(fake_tokenizer, prompt)
        assert (
            r.indices["AC"]
            < r.indices["LAT"]
            < r.indices["PANL"]
            < r.indices["PANL1"]
            < r.indices["CC"]
        )

    def test_the_tokens_at_each_position_are_the_expected_ones(
        self, fake_tokenizer: FastTokenizer
    ) -> None:
        prompt = _render(fake_tokenizer).render("What is the capital of France?", "Paris")
        r = resolve_positions(fake_tokenizer, prompt)
        assert r.tokens["AC"] == ":"
        assert r.tokens["LAT"] == " Paris"
        assert r.tokens["PANL"] == "\n"
        assert r.tokens["PANL1"] == "Confidence"
        assert r.tokens["CC"] == ":"

    def test_cc_is_the_final_token(self, fake_tokenizer: FastTokenizer) -> None:
        prompt = _render(fake_tokenizer).render("Q?", "Paris")
        r = resolve_positions(fake_tokenizer, prompt)
        assert r.indices["CC"] == r.n_tokens - 1

    def test_lat_is_the_last_token_of_a_multi_token_answer(
        self, fake_tokenizer: FastTokenizer
    ) -> None:
        prompt = _render(fake_tokenizer).render("Q?", "Japanese yen")
        r = resolve_positions(fake_tokenizer, prompt)
        assert r.tokens["LAT"] == " yen"
        # PANL still sits immediately after the final answer token.
        assert r.indices["PANL"] == r.indices["LAT"] + 1

    def test_panl_plus_one_is_the_token_after_panl(self, fake_tokenizer: FastTokenizer) -> None:
        prompt = _render(fake_tokenizer).render("Q?", "Paris")
        r = resolve_positions(fake_tokenizer, prompt)
        assert r.indices["PANL1"] == r.indices["PANL"] + 1

    def test_the_record_form_carries_indices_and_token_ids(
        self, fake_tokenizer: FastTokenizer
    ) -> None:
        prompt = _render(fake_tokenizer).render("Q?", "Paris")
        record = resolve_positions(fake_tokenizer, prompt).as_record()
        for name in POSITION_NAMES:
            assert isinstance(record[f"pos_{name}"], int)
            assert isinstance(record[f"tok_{name}"], int)

    def test_special_tokens_with_zero_width_offsets_do_not_confuse_the_map(self) -> None:
        """HF reports (0, 0) for added tokens; treating that as covering char 0 would
        misattribute an anchor to `<|im_start|>`."""
        tokenizer = FakeTokenizer(zero_width_special_offsets=True)
        prompt = _render(tokenizer).render("Q?", "Paris")
        r = resolve_positions(tokenizer, prompt)
        assert r.tokens["PANL"] == "\n"


class TestHostileTokenizations:
    """The resolver must refuse a tokenizer that destroys a boundary, not guess an index."""

    def test_newline_merged_into_the_next_word_destroys_panl(self) -> None:
        tokenizer = FakeTokenizer(merge_newline_into_next=True)
        prompt = _render(tokenizer).render("Q?", "Paris")
        with pytest.raises(PositionResolutionError, match="not its own token"):
            resolve_positions(tokenizer, prompt)

    def test_answer_merged_into_the_newline_destroys_lat(self) -> None:
        tokenizer = FakeTokenizer(merge_answer_into_newline=True)
        prompt = _render(tokenizer).render("Q?", "Paris")
        with pytest.raises(PositionResolutionError, match="LAT"):
            resolve_positions(tokenizer, prompt)

    def test_a_single_confidence_token_collapses_the_control_position(self) -> None:
        """If "Confidence:" is one token then PANL+1 *is* CC, and the "effect at PANL but
        not at PANL+1" control would be comparing the read-out position with itself."""
        tokenizer = FakeTokenizer(merge_confidence_prefix=True)
        prompt = _render(tokenizer).render("Q?", "Paris")
        with pytest.raises(PositionResolutionError, match=r"CC|strictly ordered"):
            resolve_positions(tokenizer, prompt)


class TestConfidenceClasses:
    def test_classes_resolve_to_single_tokens(self, fake_tokenizer: FastTokenizer) -> None:
        classes = resolve_confidence_classes(fake_tokenizer)
        assert classes.high_token == " high"
        assert classes.low_token == " low"
        assert classes.high_token_id != classes.low_token_id

    def test_margin_is_the_pre_softmax_difference(self, fake_tokenizer: FastTokenizer) -> None:
        classes = resolve_confidence_classes(fake_tokenizer)
        logits = [0.0] * 64
        logits[classes.high_token_id] = 2.5
        logits[classes.low_token_id] = -1.5
        assert classes.margin(logits) == pytest.approx(4.0)

    def test_a_multi_token_class_word_is_rejected(self, fake_tokenizer: FastTokenizer) -> None:
        with pytest.raises(ConfidenceTokenError, match="tokenizes to"):
            resolve_confidence_classes(fake_tokenizer, high="very sure", low="low")

    def test_identical_classes_are_rejected(self, fake_tokenizer: FastTokenizer) -> None:
        with pytest.raises(ConfidenceTokenError, match="same token id"):
            resolve_confidence_classes(fake_tokenizer, high="high", low="high")
