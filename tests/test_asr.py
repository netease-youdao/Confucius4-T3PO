import asyncio
import json

import pytest

from inference.asr import (
    ASRError,
    ASRUpdate,
    R2T2Config,
    R2T2StreamingASR,
    TranscriptAccumulator,
    parse_r2t2_message,
)


def test_r2t2_config_metadata_fills_direction_language() -> None:
    config = R2T2Config(url="ws://localhost:8093/asr_stream_api_v1", secret_key="k")
    meta = config.metadata("req-1", direction="zh2en")
    assert meta["requestId"] == "req-1"
    assert meta["secret_key"] == "k"
    assert meta["language"] == "Chinese"
    assert meta["mode"] == "slow"


def test_r2t2_config_explicit_language_overrides_direction() -> None:
    config = R2T2Config(url="ws://x", language="zhen")
    assert config.metadata("r", direction="en2zh")["language"] == "zhen"


def test_parse_r2t2_message_extracts_increment_and_reset() -> None:
    raw = json.dumps({"status": "success", "msg": {"text": "你好", "reset": False}})
    update = parse_r2t2_message(raw)
    assert update == ASRUpdate(text="你好", reset=False, asr_cost_ms=None, total_cost_ms=None)


def test_parse_r2t2_message_ignores_connected_and_keepalive() -> None:
    assert parse_r2t2_message(json.dumps({"status": "connected", "requestId": "r"})) is None
    assert parse_r2t2_message("{}") is None


def test_parse_r2t2_message_raises_asr_error_on_error_status() -> None:
    raw = json.dumps({"status": "error", "msg": "invalid secret_key"})
    with pytest.raises(ASRError, match="invalid secret_key"):
        parse_r2t2_message(raw)


def test_parse_r2t2_message_rejects_malformed_json() -> None:
    with pytest.raises(ASRError):
        parse_r2t2_message("not json")


def test_transcript_accumulator_joins_increments_and_tracks_utterances() -> None:
    acc = TranscriptAccumulator()
    acc.add(ASRUpdate(text="你好", reset=False))
    acc.add(ASRUpdate(text="今天天气不错", reset=False))
    assert acc.text == "你好今天天气不错"
    acc.add(ASRUpdate(text="。", reset=True))
    assert acc.text == ""
    assert acc.utterances == ["你好今天天气不错。"]
    assert acc.full_text == "你好今天天气不错。"


def test_streaming_asr_requires_url() -> None:
    with pytest.raises(ASRError, match="ASR_WS_URL"):
        R2T2StreamingASR(R2T2Config(url=""))


class _FakeWebSocket:
    """Enough of the ``websockets`` client surface for R2T2StreamingASR."""

    def __init__(self, replies: list[str]) -> None:
        self.sent: list[object] = []
        self._replies = list(replies)
        self.closed = False

    async def send(self, message) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if not self._replies:
            raise ConnectionResetError("no more replies")
        return self._replies.pop(0)

    async def close(self) -> None:
        self.closed = True


def test_streaming_asr_end_to_end_with_fake_socket(monkeypatch) -> None:
    replies = [
        json.dumps({"status": "connected", "requestId": "r"}),
        json.dumps({"status": "success", "msg": {"text": "你好", "reset": False}}),
        json.dumps({"status": "success", "msg": {"text": "", "reset": False}}),
        json.dumps({"status": "success", "msg": {"text": "。", "reset": True}}),
    ]
    fake_ws = _FakeWebSocket(replies)

    class FakeWebsocketsModule:
        @staticmethod
        async def connect(url, ping_interval=None):
            return fake_ws

    import sys

    monkeypatch.setitem(sys.modules, "websockets", FakeWebsocketsModule)

    updates: list[ASRUpdate] = []

    async def scenario() -> None:
        asr = R2T2StreamingASR(
            R2T2Config(url="ws://localhost:8093/asr_stream_api_v1", secret_key="k"),
            direction="zh2en",
            on_update=lambda u: updates.append(u) or asyncio.sleep(0),
            request_id="req-1",
        )
        await asr.connect()
        await asr.send_audio(b"\x00\x00\x01\x00")
        await asr.finish()

    asyncio.run(scenario())

    metadata = json.loads(fake_ws.sent[0])
    assert metadata == {
        "requestId": "req-1",
        "channels": 1,
        "sample_rate": 16000,
        "language": "Chinese",
        "use_vad": False,
        "secret_key": "k",
        "mode": "slow",
        "smooth": False,
    }
    assert fake_ws.sent[1] == b"\x00\x00\x01\x00"
    assert fake_ws.sent[2] == "YOUDAO_ONETIME_ASR_STREAM_EOS"
    # The empty-text/no-reset frame is skipped; the other two reach on_update.
    assert [u.text for u in updates] == ["你好", "。"]
    assert updates[-1].reset is True
    assert fake_ws.closed is True


def test_streaming_asr_rejects_odd_length_audio(monkeypatch) -> None:
    fake_ws = _FakeWebSocket([json.dumps({"status": "connected"})])

    class FakeWebsocketsModule:
        @staticmethod
        async def connect(url, ping_interval=None):
            return fake_ws

    import sys

    monkeypatch.setitem(sys.modules, "websockets", FakeWebsocketsModule)

    async def scenario() -> None:
        asr = R2T2StreamingASR(R2T2Config(url="ws://x", secret_key="k"))
        await asr.connect()
        with pytest.raises(ASRError, match="complete PCM16 samples"):
            await asr.send_audio(b"\x00")
        await asr.close()

    asyncio.run(scenario())
