"""Backend selection for the text_inference CLI and Python API."""

import asyncio

import pytest

from inference.text_inference import StreamingTextTranslator, build_parser, run
from inference.translation import OpenAICompatibleGenerator


def _args(**overrides):
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_auto_backend_uses_openai_when_base_url_is_set(monkeypatch) -> None:
    captured = {}

    class Recorder:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def __call__(self, message, force=False):
            return ""

    monkeypatch.setattr("inference.text_inference.OpenAICompatibleGenerator", Recorder)
    args = _args(
        backend="auto",
        base_url="http://127.0.0.1:8010/v1",
        model_id="Confucius4-T3PO",
        text="你好。",
    )
    assert asyncio.run(run(args)) == 0
    assert captured["args"] == ("http://127.0.0.1:8010/v1", "Confucius4-T3PO")


def test_auto_backend_falls_back_to_hf_without_base_url(monkeypatch) -> None:
    calls = {}

    class FakeHF:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model_id"] = model_id
            return lambda message, force=False: ""

    monkeypatch.setattr("inference.text_inference.HFTextGenerator", FakeHF)
    args = _args(backend="auto", base_url="", model_id="org/model", text="你好。")
    assert asyncio.run(run(args)) == 0
    assert calls["model_id"] == "org/model"


def test_openai_backend_requires_base_url() -> None:
    args = _args(backend="openai", base_url="", model_id="Confucius4-T3PO")
    with pytest.raises(SystemExit, match="base-url"):
        asyncio.run(run(args))


def test_openai_backend_falls_back_to_vllm_model_env(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MODEL", "served-name")
    captured = {}

    class Recorder:
        def __init__(self, base_url, model, **kwargs):
            captured["model"] = model

        def __call__(self, message, force=False):
            return ""

    monkeypatch.setattr("inference.text_inference.OpenAICompatibleGenerator", Recorder)
    args = _args(
        backend="openai", base_url="http://127.0.0.1:8010/v1", model_id="", text="你好。"
    )
    assert asyncio.run(run(args)) == 0
    assert captured["model"] == "served-name"


def test_translator_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        StreamingTextTranslator("m", "zh2en", backend="sglang-native")


def test_translator_accepts_openai_backend_without_loading_weights() -> None:
    translator = StreamingTextTranslator(
        "Confucius4-T3PO",
        "zh2en",
        backend="openai",
        base_url="http://127.0.0.1:8010",
        api_key="secret",
    )
    assert isinstance(translator.generator, OpenAICompatibleGenerator)
    # base_url without /v1 is normalized to the chat-completions route.
    assert translator.generator.url == "http://127.0.0.1:8010/v1/chat/completions"
