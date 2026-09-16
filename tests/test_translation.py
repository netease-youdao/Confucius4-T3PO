import asyncio

from inference.prompts import build_user_message, parse_model_response
from inference.translation import (
    EngineConfig,
    OpenAICompatibleGenerator,
    TranslationEngine,
    join_translation_segments,
    source_units,
    split_source,
)


def test_source_units_match_model_card_protocol() -> None:
    zh_units = split_source("你好 prioritization。", "zh2en")
    assert source_units(zh_units, "zh2en") == 4
    assert split_source("Good morning, everyone.", "en2zh") == [
        "Good",
        "morning,",
        "everyone.",
    ]


def test_prompt_contains_interleaved_history_and_current_input() -> None:
    prompt = build_user_message("zh2en", "你好¦Hello§", "世界")
    assert "<STREAMING_HISTORY>" in prompt
    assert "你好¦Hello§" in prompt
    assert "<CURRENT_INPUT>\n世界" in prompt


def test_eos_response_parser() -> None:
    assert parse_model_response("") == ("WAIT", "")
    assert parse_model_response("<WAIT>") == ("WAIT", "")
    assert parse_model_response("TRANS Hello") == ("TRANS", "Hello")
    assert parse_model_response("Translation starts here.") == (
        "TRANS",
        "Translation starts here.",
    )
    assert parse_model_response("hello¦world") == ("TRANS", "hello｜world")


def test_engine_waits_then_commits_and_flushes() -> None:
    calls: list[tuple[str, bool]] = []

    def complete(message: str, force: bool) -> str:
        calls.append((message, force))
        if force:
            return "Hello world."
        return "" if len(calls) == 1 else "Hello"

    engine = TranslationEngine(
        EngineConfig(direction="en2zh", force_break_threshold=20), complete
    )
    assert asyncio.run(engine.feed("Hello"))[0]["type"] == "wait"
    events = asyncio.run(engine.feed("world."))
    assert events[0]["type"] == "translation"
    assert events[0]["text"] == "Hello"
    assert engine.pending_source == ""
    assert asyncio.run(engine.flush()) == []
    assert calls


def test_engine_does_not_expose_internal_force_flag() -> None:
    async def complete(_message: str, _force: bool) -> str:
        return "你好"

    engine = TranslationEngine(
        EngineConfig(direction="zh2en", force_break_threshold=1), complete
    )
    event = asyncio.run(engine.feed("你"))[0]
    assert event == {"type": "translation", "source": "你", "text": "你好", "source_units": 1}
    assert "forced" not in event


def test_join_translation_segments_repairs_english_spacing() -> None:
    assert join_translation_segments(["Hello", "world", "!"], "zh2en") == "Hello world!"
    assert join_translation_segments(["大家", "好。"], "en2zh") == "大家好。"


def test_openai_compatible_generator_builds_vllm_request() -> None:
    recorded = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Hello"}}]}

    class Client:
        def post(self, url, *, headers, json):
            recorded.update(url=url, headers=headers, payload=json)
            return Response()

    generator = OpenAICompatibleGenerator(
        "http://127.0.0.1:8001/v1",
        "bilingual-simt",
        api_key="secret",
    )
    generator._client = Client()
    assert generator("prompt", True) == "Hello"
    assert recorded["url"].endswith("/v1/chat/completions")
    assert recorded["payload"]["min_tokens"] == 1
    assert recorded["headers"]["authorization"] == "Bearer secret"
