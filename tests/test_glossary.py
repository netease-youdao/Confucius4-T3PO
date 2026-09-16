"""Terminology selection, rendering, and its effect on the request."""

import asyncio

from inference.glossary import (
    MAX_TERMS,
    build_glossary_block,
    normalize_terms,
    terms_in_segment,
)
from inference.prompts import STREAMING_PROMPTS, build_user_message
from inference.translation import EngineConfig, TranslationEngine


def test_normalize_accepts_both_wire_shapes_and_dedupes() -> None:
    terms = normalize_terms(
        [
            {"src": "破产隔离", "trg": "bankruptcy remote"},
            ["尽职调查", "due diligence"],
            {"source": "对赌协议", "target": "valuation adjustment mechanism"},
            {"src": "破产隔离", "trg": "bankruptcy remote"},  # duplicate
            {"src": "", "trg": "empty source"},
            {"src": "no target", "trg": ""},
            "malformed",
        ]
    )
    assert terms == (
        ("破产隔离", "bankruptcy remote"),
        ("尽职调查", "due diligence"),
        ("对赌协议", "valuation adjustment mechanism"),
    )


def test_normalize_rejects_non_lists_and_caps_length() -> None:
    assert normalize_terms(None) == ()
    assert normalize_terms("terms") == ()
    assert normalize_terms({"src": "a", "trg": "b"}) == ()
    many = [{"src": f"term{i}", "trg": f"t{i}"} for i in range(MAX_TERMS + 50)]
    assert len(normalize_terms(many)) == MAX_TERMS


def test_only_terms_present_in_the_segment_are_selected() -> None:
    terms = normalize_terms(
        [{"src": "破产隔离", "trg": "bankruptcy remote"}, {"src": "尽职调查", "trg": "due diligence"}]
    )
    assert terms_in_segment(terms, "这个方法的核心是破产隔离") == (
        ("破产隔离", "bankruptcy remote"),
    )
    assert terms_in_segment(terms, "今天天气不错") == ()
    assert terms_in_segment((), "破产隔离") == ()


def test_long_terms_match_across_mangled_spacing() -> None:
    terms = normalize_terms([{"src": "bankruptcy remote", "trg": "破产隔离"}])
    # ASR dropped the space; the space-stripped comparison still matches.
    assert terms_in_segment(terms, "a bankruptcyremote structure") == (
        ("bankruptcy remote", "破产隔离"),
    )


def test_short_terms_require_a_literal_match() -> None:
    # A two-character term must not fire on a space-stripped coincidence.
    terms = normalize_terms([{"src": "AI", "trg": "人工智能"}])
    assert terms_in_segment(terms, "AI 很有用") == (("AI", "人工智能"),)
    assert terms_in_segment(terms, "A I 分开写") == ()


def test_block_is_empty_without_hits_and_labelled_as_reference() -> None:
    assert build_glossary_block((), "zh2en") == ""
    block = build_glossary_block((("破产隔离", "bankruptcy remote"),), "zh2en")
    assert "- 破产隔离 -> bankruptcy remote" in block
    # The framing matters: the model must not translate the list itself.
    assert "reference only" in block
    assert "NOT source text to translate" in block


def test_no_glossary_leaves_the_request_byte_identical() -> None:
    prompt = STREAMING_PROMPTS["zh2en"]
    baseline = f"{prompt}\n\n<STREAMING_HISTORY>\nH\n\n<CURRENT_INPUT>\nC"
    assert build_user_message("zh2en", "H", "C") == baseline
    assert build_user_message("zh2en", "H", "C", glossary="") == baseline


def test_glossary_sits_between_history_and_current_input() -> None:
    message = build_user_message("zh2en", "H", "C", glossary="GLOSSARY")
    history_at = message.index("<STREAMING_HISTORY>")
    glossary_at = message.index("GLOSSARY")
    current_at = message.index("<CURRENT_INPUT>")
    # Prefix caching depends on this ordering.
    assert history_at < glossary_at < current_at


def test_engine_injects_only_matching_terms_per_segment() -> None:
    seen: list[str] = []

    def fake_complete(message: str, force: bool = False) -> str:
        seen.append(message)
        return "out"

    engine = TranslationEngine(
        EngineConfig(
            direction="zh2en",
            force_break_threshold=1,
            terms=normalize_terms([{"src": "破产隔离", "trg": "bankruptcy remote"}]),
        ),
        fake_complete,
    )
    asyncio.run(engine.feed("破产隔离"))
    asyncio.run(engine.feed("今天天气"))

    assert "bankruptcy remote" in seen[0]
    # The second segment has no hit, so it carries no terminology block at all.
    assert "Terminology" not in seen[1]
