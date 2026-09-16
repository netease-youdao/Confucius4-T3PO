import asyncio

import inference.server as server


class FakeGenerator:
    def __call__(self, _message: str, _force: bool = False) -> str:
        return "Hello"


def test_static_demo_and_health_are_available_without_loading_models() -> None:
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "Confucius4-T3PO" in html
    # The demo ships its own controls; a missing one means the page and the
    # client script have drifted apart.
    for element in ("latencyMode", "glossaryButton", "textInput", "direction"):
        assert f'id="{element}"' in html
    health = asyncio.run(server.health())
    assert health["status"] == "ok"


def test_translation_session_uses_shared_generator() -> None:
    session = server.TranslationSession("zh2en", FakeGenerator())
    payload = asyncio.run(session.feed("你好"))
    assert payload[0]["text"] == "Hello"
    assert asyncio.run(session.flush()) == []


class _RecordingWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def test_speech_session_flushes_tail_on_asr_reset() -> None:
    from inference.asr import ASRUpdate

    async def scenario() -> list[dict]:
        session = server.SpeechSession(_RecordingWebSocket(), "zh2en")
        session._translation = server.TranslationSession("zh2en", FakeGenerator())
        worker = asyncio.create_task(session._translate_worker())
        await session._on_asr_update(ASRUpdate(text="你好", reset=False))
        await session._on_asr_update(ASRUpdate(text="", reset=True))
        await session._translation_queue.join()
        await session._translation_queue.put(None)
        await worker
        return session.websocket.sent

    events = asyncio.run(scenario())
    asr_events = [e for e in events if e["type"] == "asr"]
    assert asr_events == [{"type": "asr", "text": "你好", "reset": False}]
    # The reset with no new text still queues a flush, which yields a
    # translation because FakeGenerator always returns non-empty text.
    translations = [e for e in events if e["type"] == "translation"]
    assert len(translations) == 1
    assert translations[0]["text"] == "Hello"


def test_translation_session_honors_per_session_segmentation_overrides() -> None:
    default = server.TranslationSession("zh2en", FakeGenerator())
    tuned = server.TranslationSession(
        "zh2en", FakeGenerator(), force_break_threshold=15, punct_force=True
    )
    assert tuned.engine.config.force_break_threshold == 15
    assert tuned.engine.config.punct_force is True
    # Omitting them keeps the process-wide settings.
    assert default.engine.config.force_break_threshold == server.settings.force_break_threshold
    assert default.engine.config.punct_force == server.settings.punct_force


def test_idle_timer_flushes_tail_when_asr_goes_quiet() -> None:
    from inference.asr import ASRUpdate

    async def scenario() -> list[dict]:
        session = server.SpeechSession(
            _RecordingWebSocket(), "zh2en", idle_force_seconds=0.05
        )
        session._translation = server.TranslationSession("zh2en", FakeGenerator())
        worker = asyncio.create_task(session._translate_worker())
        # Text arrives, then ASR stays silent past the idle window.
        await session._on_asr_update(ASRUpdate(text="这个方法的", reset=False))
        await asyncio.sleep(0.2)
        await session._translation_queue.join()
        session._cancel_idle_timer()
        await session._translation_queue.put(None)
        await worker
        return session.websocket.sent

    events = asyncio.run(scenario())
    # The silence triggered a flush, so the buffered tail was translated.
    assert [e["text"] for e in events if e["type"] == "translation"] == ["Hello"]


def test_idle_timer_disabled_by_default() -> None:
    async def scenario() -> object:
        from inference.asr import ASRUpdate

        session = server.SpeechSession(_RecordingWebSocket(), "zh2en")
        session._translation = server.TranslationSession("zh2en", FakeGenerator())
        await session._on_asr_update(ASRUpdate(text="这个", reset=False))
        return session._idle_task

    assert server.settings.idle_force_seconds == 0.0
    assert asyncio.run(scenario()) is None
