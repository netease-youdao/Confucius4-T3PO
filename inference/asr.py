"""WebSocket client for the separately released Confucius4-R2T2 streaming ASR.

This project does not run ASR inference itself.  R2T2 is deployed as its own
service (see its repository for ``run_start_server.sh``) and this module only
speaks its WebSocket protocol, so ASR model updates never require a change
here.

Protocol summary (``/asr_stream_api_v1``):

* Send one JSON metadata frame first: ``requestId`` is required, and
  ``secret_key`` is validated by the server, which closes with code 4401 when
  it does not match.  The server answers ``{"status": "connected", ...}``.
* Then stream raw PCM16LE frames as binary messages.
* Send the literal string ``YOUDAO_ONETIME_ASR_STREAM_EOS`` to finish; the
  server replies with the final result and closes the connection.
* Downstream messages are ``{"status": "success", "msg": {"text": ...,
  "reset": ...}}``.  ``text`` carries only the newly fixed increment (it may be
  an empty string when a chunk produced nothing), so no prefix reconciliation
  is needed on our side.  ``reset`` marks the end of an ASR utterance, either
  from server-side VAD or from the final EOS response.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

# Sentinel understood by the R2T2 server to end a streaming request.
EOS_MESSAGE = "YOUDAO_ONETIME_ASR_STREAM_EOS"

# R2T2 accepts "zhen" (mixed Chinese/English), "Chinese", or "English".
# Our zh2en/en2zh directions describe the *source* language.
DIRECTION_TO_LANGUAGE = {"zh2en": "Chinese", "en2zh": "English"}


class ASRError(RuntimeError):
    """The ASR service could not be reached or returned an error."""


@dataclass(frozen=True)
class ASRUpdate:
    """One increment of recognized text.

    ``text`` is the newly fixed fragment only.  ``reset`` is True when the
    service considers the current utterance finished, which is the cue to flush
    the translation tail.
    """

    text: str
    reset: bool = False
    asr_cost_ms: float | None = None
    total_cost_ms: float | None = None


@dataclass(frozen=True)
class R2T2Config:
    """Connection settings for the R2T2 WebSocket service."""

    url: str
    secret_key: str = ""
    sample_rate: int = 16_000
    channels: int = 1
    language: str = ""
    use_vad: bool = False
    mode: str = "slow"
    smooth: bool = False
    connect_timeout_seconds: float = 10.0
    recv_timeout_seconds: float = 30.0

    def metadata(self, request_id: str, direction: str | None = None) -> dict[str, Any]:
        """Build the JSON header frame for one request."""

        language = self.language.strip()
        if not language and direction:
            language = DIRECTION_TO_LANGUAGE.get(direction, "")
        return {
            "requestId": request_id,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "language": language or "zhen",
            "use_vad": self.use_vad,
            "secret_key": self.secret_key,
            "mode": self.mode,
            "smooth": self.smooth,
        }


def parse_r2t2_message(raw: Any) -> ASRUpdate | None:
    """Translate one downstream message into an ``ASRUpdate``.

    Returns ``None`` for frames that carry no transcript update: the empty
    ``{}`` keep-alive sent while the server buffers its first window, and
    messages without a recognizable payload.  Raises ``ASRError`` when the
    service reports an error so the caller can surface it to the browser.
    """

    if isinstance(raw, (bytes, bytearray)):
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ASRError("the ASR service returned a malformed message") from error
    if not isinstance(payload, dict):
        return None

    status = payload.get("status")
    if status == "error":
        message = payload.get("msg")
        if not isinstance(message, str) or not message:
            message = "the ASR service reported an error"
        raise ASRError(message)
    if status != "success":
        # "connected" and the empty {} warm-up frame carry no transcript.
        return None

    message = payload.get("msg")
    if not isinstance(message, dict):
        return None
    return ASRUpdate(
        text=str(message.get("text") or ""),
        reset=bool(message.get("reset", False)),
        asr_cost_ms=message.get("asr_cost_ms"),
        total_cost_ms=message.get("total_cost_ms"),
    )


class R2T2StreamingASR:
    """One streaming ASR request against the R2T2 service.

    The instance owns a single WebSocket connection and is not reusable after
    :meth:`close`, mirroring the one-request-per-connection design of the
    service (it closes the socket right after the final result).
    """

    def __init__(
        self,
        config: R2T2Config,
        *,
        direction: str | None = None,
        on_update: Callable[[ASRUpdate], Awaitable[None]] | None = None,
        request_id: str | None = None,
    ) -> None:
        if not str(config.url or "").strip():
            raise ASRError("ASR_WS_URL is not configured for the audio demo")
        self.config = config
        self.direction = direction
        self.on_update = on_update
        self.request_id = request_id or str(uuid.uuid4())
        self._ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._closed = False

    async def connect(self) -> None:
        try:
            import websockets
        except ImportError as error:  # pragma: no cover - dependency environment
            raise ASRError("the websockets package is required for audio input") from error

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(self.config.url, ping_interval=None),
                timeout=self.config.connect_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise ASRError("timed out connecting to the ASR service") from error
        except Exception as error:
            raise ASRError("unable to connect to the ASR service") from error

        metadata = self.config.metadata(self.request_id, self.direction)
        try:
            await self._ws.send(json.dumps(metadata))
        except Exception as error:
            await self.close()
            raise ASRError("unable to initialize the ASR request") from error
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            while True:
                raw = await self._ws.recv()
                update = parse_r2t2_message(raw)
                if update is None:
                    continue
                # Empty text with reset=False means "no new words yet"; skip it
                # so callers are not woken for every silent chunk.
                if not update.text and not update.reset:
                    continue
                if self.on_update is not None:
                    await self.on_update(update)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A closed connection ends the stream; the caller observes this
            # through finish()/close() rather than an exception here.
            return

    async def send_audio(self, pcm16le: bytes) -> None:
        """Forward one raw PCM16LE frame to the service."""

        if self._closed or self._ws is None:
            raise ASRError("the ASR connection is not open")
        if len(pcm16le) % 2:
            raise ASRError("audio chunks must contain complete PCM16 samples")
        if not pcm16le:
            return
        try:
            await self._ws.send(pcm16le)
        except Exception as error:
            raise ASRError("the ASR connection was closed while sending audio") from error

    async def finish(self) -> None:
        """Send EOS and wait for the service to deliver the final result."""

        if self._closed or self._ws is None:
            return
        try:
            await self._ws.send(EOS_MESSAGE)
        except Exception:
            # Already disconnected; nothing further to collect.
            await self.close()
            return
        if self._reader is not None:
            try:
                # The server closes the socket after the final message, which
                # ends the reader task on its own.
                await asyncio.wait_for(self._reader, timeout=self.config.recv_timeout_seconds)
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass
        await self.close()

    async def close(self) -> None:
        self._closed = True
        reader, self._reader = self._reader, None
        if reader is not None and not reader.done():
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, Exception):
                pass
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass


@dataclass
class TranscriptAccumulator:
    """Join R2T2 increments into a display transcript.

    The service already emits monotonic, non-revising increments, so this only
    concatenates them and tracks utterance boundaries for the UI.
    """

    text: str = ""
    utterances: list[str] = field(default_factory=list)

    def add(self, update: ASRUpdate) -> str:
        if update.text:
            self.text += update.text
        if update.reset:
            if self.text:
                self.utterances.append(self.text)
            self.text = ""
        return self.text

    @property
    def full_text(self) -> str:
        parts = [*self.utterances]
        if self.text:
            parts.append(self.text)
        return "".join(parts)
