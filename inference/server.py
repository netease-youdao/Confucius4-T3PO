"""FastAPI service and modern browser demo for local HF inference."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .asr import ASRError, ASRUpdate, R2T2Config, R2T2StreamingASR
from .config import Settings
from .glossary import MAX_TERMS, normalize_terms
from .latency import (
    DEFAULT_LATENCY_MODE,
    LATENCY_MODES,
    build_logit_bias,
    describe_modes,
    resolve_latency_mode,
)
from .runtime import run_blocking
from .translation import (
    Direction,
    EngineConfig,
    HFTextGenerator,
    OpenAICompatibleGenerator,
    TranslationEngine,
)

# Sentinel queued to flush the translation tail without stopping the worker.
_FLUSH = object()

LOGGER = logging.getLogger("streaming_translation.inference")
logging.basicConfig(
    level=getattr(logging, __import__("os").environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

settings = Settings.from_environment()


class InferenceError(RuntimeError):
    """A configured local model could not complete inference."""


class ModelStore:
    """Process-wide lazy translation cache.

    Translation weights (or the vLLM client) are shared by all text/audio
    sessions.  ASR is not cached here because it runs in the separately
    deployed R2T2 service; each speech session opens its own connection.
    """

    def __init__(self) -> None:
        # The openai backend needs one client per latency mode because the mode
        # changes the request payload. The hf backend holds the weights, so it
        # is cached once under the native key.
        self.translation: dict[str, Any] = {}
        self.translation_lock = asyncio.Lock()

    async def get_translation(self, latency_mode: str = DEFAULT_LATENCY_MODE) -> Any:
        mode = resolve_latency_mode(latency_mode)
        if settings.translation_backend == "hf" and mode.tau != 0.0:
            raise InferenceError(
                "a non-native latency mode requires the vLLM (openai) backend; "
                "the in-process Hugging Face backend does not apply the WAIT bias"
            )
        key = mode.name if settings.translation_backend == "openai" else "native"
        existing = self.translation.get(key)
        if existing is not None:
            return existing
        if not settings.translation_configured:
            required = (
                "VLLM_BASE_URL and VLLM_MODEL"
                if settings.translation_backend == "openai"
                else "TRANSLATION_MODEL_ID"
            )
            raise InferenceError(
                f"{required} is not configured for the {settings.translation_backend} "
                "translation backend"
            )
        async with self.translation_lock:
            if self.translation.get(key) is None:
                try:
                    if settings.translation_backend == "openai":
                        self.translation[key] = OpenAICompatibleGenerator(
                            settings.vllm_base_url,
                            settings.vllm_model,
                            api_key=settings.vllm_api_key,
                            timeout_seconds=settings.vllm_timeout_seconds,
                            max_new_tokens=settings.translation_max_new_tokens,
                            force_max_new_tokens=settings.translation_force_max_new_tokens,
                            min_tokens_on_force=settings.translation_min_new_tokens_on_force,
                            logit_bias=build_logit_bias(
                                mode.tau, mode.stop_token_ids, mode.bias_scales
                            ),
                            repetition_penalty=mode.repetition_penalty,
                        )
                    else:
                        self.translation[key] = await run_blocking(
                            HFTextGenerator.from_pretrained,
                            settings.translation_model_id,
                            revision=settings.translation_revision,
                            cache_dir=settings.hf_cache_dir,
                            token=settings.hf_token,
                            local_files_only=settings.local_files_only,
                            trust_remote_code=settings.trust_remote_code,
                            device=settings.translation_device,
                            dtype=settings.translation_dtype,
                            max_new_tokens=settings.translation_max_new_tokens,
                            force_max_new_tokens=settings.translation_force_max_new_tokens,
                            min_new_tokens_on_force=settings.translation_min_new_tokens_on_force,
                        )
                except Exception as error:
                    raise InferenceError(
                        "unable to load the translation model; check the model id, "
                        "Hugging Face access, and available device memory"
                    ) from error
        return self.translation[key]

    async def close(self) -> None:
        generators, self.translation = self.translation, {}
        for generator in generators.values():
            if generator is not None and callable(getattr(generator, "close", None)):
                await run_blocking(generator.close)


model_store = ModelStore()


class TranslateRequest(BaseModel):
    session_id: str | None = None
    direction: Direction | None = None
    text: str = Field(default="", max_length=20_000)
    end: bool = False
    # Only honored when creating a session; ignored on continuation requests.
    latency_mode: str | None = None


class TranslationSession:
    def __init__(
        self,
        direction: Direction,
        generator: Any,
        *,
        force_break_threshold: int | None = None,
        punct_force: bool | None = None,
        terms: tuple[tuple[str, str], ...] = (),
    ) -> None:
        async def complete(message: str, force: bool) -> str:
            try:
                return str(await run_blocking(generator, message, force) or "")
            except Exception as error:
                raise InferenceError("translation inference failed") from error

        self.engine = TranslationEngine(
            EngineConfig(
                direction=direction,
                force_break_threshold=(
                    settings.force_break_threshold
                    if force_break_threshold is None
                    else force_break_threshold
                ),
                max_buffer_units=settings.max_buffer_units,
                punct_force=(
                    settings.punct_force if punct_force is None else punct_force
                ),
                history_window=settings.history_window,
                terms=terms,
            ),
            complete,
        )
        self.lock = asyncio.Lock()
        self.last_access = time.monotonic()

    async def feed(self, text: str) -> list[dict[str, Any]]:
        async with self.lock:
            result = await self.engine.feed(text)
            self.last_access = time.monotonic()
            return result

    async def flush(self) -> list[dict[str, Any]]:
        async with self.lock:
            result = await self.engine.flush()
            self.last_access = time.monotonic()
            return result


sessions: dict[str, TranslationSession] = {}
sessions_lock = asyncio.Lock()
active_speech_sessions = 0


async def create_translation_session(
    direction: Direction, latency_mode: str | None = None
) -> tuple[str, TranslationSession]:
    async with sessions_lock:
        if len(sessions) >= settings.max_active_sessions:
            raise HTTPException(status_code=429, detail="too many active sessions")
    generator = await model_store.get_translation(latency_mode or settings.latency_mode)
    async with sessions_lock:
        if len(sessions) >= settings.max_active_sessions:
            raise HTTPException(status_code=429, detail="too many active sessions")
        session_id = str(uuid.uuid4())
        session = TranslationSession(direction, generator)
        sessions[session_id] = session
        return session_id, session


async def pop_translation_session(session_id: str) -> None:
    async with sessions_lock:
        sessions.pop(session_id, None)


async def cleanup_sessions() -> None:
    while True:
        await asyncio.sleep(30)
        now = time.monotonic()
        async with sessions_lock:
            expired = [
                session_id
                for session_id, session in sessions.items()
                if not session.lock.locked()
                and now - session.last_access > settings.session_ttl_seconds
            ]
            for session_id in expired:
                sessions.pop(session_id, None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(cleanup_sessions())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        sessions.clear()
        await model_store.close()


app = FastAPI(
    title="Confucius4-T3PO",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/api/v1/translate")
async def translate(request: TranslateRequest) -> dict[str, Any]:
    if request.session_id:
        async with sessions_lock:
            session = sessions.get(request.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found or expired")
        session_id = request.session_id
    else:
        if request.direction is None:
            raise HTTPException(status_code=400, detail="direction is required for a new session")
        try:
            session_id, session = await create_translation_session(
                request.direction, request.latency_mode
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except InferenceError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    try:
        events = await session.feed(request.text) if request.text else []
        if request.end:
            events.extend(await session.flush())
    except InferenceError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("text inference failed")
        raise HTTPException(status_code=502, detail="local translation inference failed") from error
    if request.end:
        await pop_translation_session(session_id)
    return {"session_id": session_id, "events": events}


@app.get("/api/v1/session/{session_id}/stats")
async def session_stats(session_id: str) -> dict[str, Any]:
    async with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found or expired")
    async with session.lock:
        session.last_access = time.monotonic()
        return {"session_id": session_id, **session.engine.stats}


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "translation_backend": settings.translation_backend,
        "translation_model": settings.translation_name or None,
        "translation_loaded": bool(model_store.translation),
        "latency_modes": describe_modes(),
        "max_terms": MAX_TERMS,
        "default_latency_mode": settings.latency_mode,
        "asr_service": settings.asr_ws_url or None,
        "asr_configured": settings.asr_configured,
        "active_sessions": len(sessions),
        "active_speech_sessions": active_speech_sessions,
        "sample_rate": settings.asr_sample_rate,
    }


@app.websocket("/ws/simul-demo")
async def simul_demo_websocket(websocket: WebSocket) -> None:
    global active_speech_sessions
    await websocket.accept()
    if active_speech_sessions >= settings.max_active_sessions:
        await websocket.send_json({"type": "error", "message": "too many active sessions"})
        await websocket.close(code=1013)
        return
    active_speech_sessions += 1
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=15)
        init = json.loads(raw)
        if init.get("type") != "init" or init.get("direction") not in {"zh2en", "en2zh"}:
            await websocket.send_json({"type": "error", "message": "a valid init message is required"})
            await websocket.close(code=1008)
            return
        await SpeechSession(
            websocket,
            init["direction"],
            latency_mode=init.get("latency_mode"),
            force_break_threshold=init.get("force_break_threshold"),
            punct_force=init.get("punct_force"),
            idle_force_seconds=init.get("idle_force_seconds"),
            terms=init.get("terms"),
        ).run()
    except WebSocketDisconnect:
        LOGGER.info("browser speech session disconnected")
    except (ASRError, InferenceError, json.JSONDecodeError, ValueError) as error:
        LOGGER.warning("speech session failed: %s", error)
        try:
            await websocket.send_json({"type": "error", "message": str(error)})
        except Exception:
            pass
    except Exception:
        LOGGER.exception("unexpected speech session failure")
        try:
            await websocket.send_json({"type": "error", "message": "unexpected server error"})
        except Exception:
            pass
    finally:
        active_speech_sessions -= 1
        try:
            await websocket.close()
        except Exception:
            pass


class SpeechSession:
    """One browser session: PCM16LE -> R2T2 ASR service -> local/vLLM MT."""

    def __init__(
        self,
        websocket: WebSocket,
        direction: Direction,
        *,
        latency_mode: str | None = None,
        force_break_threshold: int | None = None,
        punct_force: bool | None = None,
        idle_force_seconds: float | None = None,
        terms: object = None,
    ) -> None:
        self.websocket = websocket
        self.direction = direction
        # Validate here so a bad value fails before any model work starts.
        self.latency_mode = resolve_latency_mode(
            latency_mode or settings.latency_mode
        ).name
        self.force_break_threshold = force_break_threshold
        self.punct_force = punct_force
        # Translate a buffered tail when ASR goes quiet, so the last clause of
        # an utterance is not left on screen untranslated. 0 disables it.
        self.idle_force_seconds = (
            settings.idle_force_seconds
            if idle_force_seconds is None
            else max(0.0, float(idle_force_seconds))
        )
        self._idle_task: asyncio.Task[None] | None = None
        # Pinned once per session: the glossary is part of the prompt, so
        # changing it mid-stream would invalidate the prefix cache.
        self.terms = normalize_terms(terms)
        self._send_lock = asyncio.Lock()
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        # Items: a source-text increment to feed, _FLUSH to flush the tail
        # without stopping, or None to drain and stop the worker.
        self._translation_queue: asyncio.Queue[str | object | None] = asyncio.Queue()
        self._debug_source = ""
        self._paused = False
        self._ending = False
        self._audio_bytes = 0
        self._started_at = time.monotonic()
        self._asr: R2T2StreamingASR | None = None
        self._translation: TranslationSession | None = None

    async def send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_json(payload)

    def _cancel_idle_timer(self) -> None:
        task, self._idle_task = self._idle_task, None
        if task is not None and not task.done():
            task.cancel()

    def _arm_idle_timer(self) -> None:
        """Flush the buffered tail if no new ASR text arrives in time."""

        if not self.idle_force_seconds:
            return
        self._cancel_idle_timer()

        async def wait_then_flush() -> None:
            try:
                await asyncio.sleep(self.idle_force_seconds)
            except asyncio.CancelledError:
                return
            self._translation_queue.put_nowait(_FLUSH)

        self._idle_task = asyncio.create_task(wait_then_flush())

    async def _on_asr_update(self, update: ASRUpdate) -> None:
        # R2T2 sends only the newly fixed increment; it never revises text it
        # already emitted, so there is no prefix reconciliation to do here.
        if update.text:
            await self.send({"type": "asr", "text": update.text, "reset": update.reset})
            self._translation_queue.put_nowait(update.text)
            # Restart the countdown: the speaker is still going.
            self._arm_idle_timer()
        if update.reset:
            # The service considers the current utterance finished (VAD or the
            # final EOS reply). Flush the tail so it is not held indefinitely.
            self._cancel_idle_timer()
            self._translation_queue.put_nowait(_FLUSH)

    async def _translate_worker(self) -> None:
        assert self._translation is not None
        while True:
            item = await self._translation_queue.get()
            try:
                if item is None:
                    return
                if item is _FLUSH:
                    events = await self._translation.flush()
                else:
                    events = await self._translation.feed(item)
                for event in events:
                    if event.get("type") == "translation":
                        await self.send(event)
                await self.send({"type": "metrics", **self._translation.engine.stats})
            finally:
                self._translation_queue.task_done()

    async def _send_text_increment(self, text: str) -> None:
        """Handle the optional text-debug control without requiring ASR."""

        if not self._translation or not text.strip():
            return
        separator = "" if self.direction == "zh2en" else (" " if self._debug_source else "")
        self._debug_source = f"{self._debug_source}{separator}{text.strip()}"
        await self.send({"type": "asr", "text": text.strip(), "reset": False})
        events = await self._translation.feed(text)
        for event in events:
            if event.get("type") == "translation":
                await self.send(event)
        await self.send({"type": "metrics", **self._translation.engine.stats})

    async def _asr_worker(self) -> None:
        """Forward queued audio frames to the R2T2 connection in order."""

        assert self._asr is not None
        while True:
            audio = await self._audio_queue.get()
            try:
                if audio is None:
                    await self._asr.finish()
                    return
                await self._asr.send_audio(audio)
            finally:
                self._audio_queue.task_done()

    async def run(self) -> None:
        await self.send({"type": "loading", "component": "translation"})
        generator = await model_store.get_translation(self.latency_mode)
        self._translation = TranslationSession(
            self.direction,
            generator,
            force_break_threshold=self.force_break_threshold,
            punct_force=self.punct_force,
            terms=self.terms,
        )
        # Text-debug mode remains useful even without ASR configured. Audio
        # frames receive a clear error in that case, while a normal deployment
        # opens a connection to the separately released R2T2 service here.
        if settings.asr_configured:
            await self.send({"type": "loading", "component": "asr"})
            self._asr = R2T2StreamingASR(
                R2T2Config(
                    url=settings.asr_ws_url,
                    secret_key=settings.asr_secret_key,
                    sample_rate=settings.asr_sample_rate,
                    language=settings.asr_language,
                    use_vad=settings.asr_use_vad,
                    mode=settings.asr_mode,
                    smooth=settings.asr_smooth,
                    connect_timeout_seconds=settings.asr_connect_timeout_seconds,
                    recv_timeout_seconds=settings.asr_recv_timeout_seconds,
                ),
                direction=self.direction,
                on_update=self._on_asr_update,
            )
            await self._asr.connect()
        await self.send(
            {
                "type": "init_ok",
                "direction": self.direction,
                "latency_mode": self.latency_mode,
                "terms": len(self.terms),
                "sample_rate": settings.asr_sample_rate,
                "translation_backend": settings.translation_backend,
                "translation_model": settings.translation_name,
                "asr_service": settings.asr_ws_url or None,
                "asr_configured": self._asr is not None,
            }
        )
        asr_task = asyncio.create_task(self._asr_worker()) if self._asr is not None else None
        translation_task = asyncio.create_task(self._translate_worker())
        try:
            while True:
                browser_task = asyncio.create_task(self.websocket.receive())
                workers = {translation_task}
                if asr_task is not None:
                    workers.add(asr_task)
                done, _ = await asyncio.wait(
                    {browser_task, *workers},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                stopped_worker = next((task for task in workers if task in done), None)
                if stopped_worker is not None:
                    if not browser_task.done():
                        browser_task.cancel()
                        await asyncio.gather(browser_task, return_exceptions=True)
                    stopped_worker.result()
                    raise RuntimeError("an inference worker stopped unexpectedly")
                message = browser_task.result()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(code=message.get("code", 1000))
                audio = message.get("bytes")
                if audio is not None:
                    if asr_task is None:
                        raise ASRError(
                            "ASR_WS_URL is not configured; audio input is unavailable"
                        )
                    if self._paused:
                        continue
                    if len(audio) > settings.max_audio_chunk_bytes:
                        raise ASRError("audio chunk exceeds MAX_AUDIO_CHUNK_BYTES")
                    if len(audio) % 2:
                        raise ASRError("audio chunks must contain complete PCM16 samples")
                    self._audio_bytes += len(audio)
                    if self._audio_bytes > settings.max_session_audio_bytes:
                        raise ASRError("audio session exceeds MAX_SESSION_AUDIO_BYTES")
                    self._audio_queue.put_nowait(audio)
                    continue
                text = message.get("text")
                if text is None:
                    continue
                command = json.loads(text)
                command_type = command.get("type")
                if command_type == "pause":
                    self._paused = True
                    await self.send({"type": "pause_ok"})
                elif command_type == "resume":
                    self._paused = False
                    await self.send({"type": "resume_ok"})
                elif command_type == "text":
                    await self._send_text_increment(str(command.get("text") or ""))
                elif command_type == "end":
                    self._ending = True
                    if asr_task is not None:
                        await self._audio_queue.put(None)
                    break
            if asr_task is not None:
                await asr_task
            await self._translation_queue.put(None)
            await translation_task
            if self._translation is not None:
                for event in await self._translation.flush():
                    if event.get("type") == "translation":
                        await self.send(event)
                await self.send({"type": "metrics", **self._translation.engine.stats})
            await self.send(
                {
                    "type": "ended",
                    "elapsed_ms": round((time.monotonic() - self._started_at) * 1000),
                    "audio_bytes": self._audio_bytes,
                    "stats": self._translation.engine.stats if self._translation else {},
                }
            )
        finally:
            for task in (asr_task, translation_task):
                if task is None:
                    continue
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            self._cancel_idle_timer()
            if self._asr is not None:
                await self._asr.close()


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/modern", include_in_schema=False)
@app.get("/modern/", include_in_schema=False)
async def modern_redirect() -> RedirectResponse:
    return RedirectResponse(url="/")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Any:
    path = STATIC_DIR / "favicon.ico"
    if path.exists():
        return FileResponse(path)
    raise HTTPException(status_code=404)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="modern")


def main() -> None:
    """Console entry point installed as ``streaming-translate-web``."""

    import uvicorn

    uvicorn.run("inference.server:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
