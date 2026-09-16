"""Environment-only configuration for the public inference package."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    Model identifiers intentionally have no private default.  A release
    consumer must opt in to the public translation repository (or provide a
    local path), which prevents an example deployment from silently downloading
    an unrelated checkpoint.

    ASR is not run in this process: audio is forwarded to the separately
    released Confucius4-R2T2 WebSocket service at ``ASR_WS_URL``.
    """

    translation_backend: str
    translation_model_id: str
    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str | None
    vllm_timeout_seconds: float
    latency_mode: str
    idle_force_seconds: float
    asr_ws_url: str
    asr_secret_key: str
    asr_use_vad: bool
    asr_mode: str
    asr_smooth: bool
    asr_connect_timeout_seconds: float
    asr_recv_timeout_seconds: float
    translation_revision: str | None
    hf_cache_dir: str | None
    hf_token: str | None
    local_files_only: bool
    trust_remote_code: bool
    translation_device: str
    translation_dtype: str
    translation_max_new_tokens: int
    translation_force_max_new_tokens: int
    translation_min_new_tokens_on_force: int
    asr_sample_rate: int
    asr_language: str
    max_audio_chunk_bytes: int
    max_session_audio_bytes: int
    force_break_threshold: int
    max_buffer_units: int
    punct_force: bool
    history_window: int
    session_ttl_seconds: int
    max_active_sessions: int
    host: str
    port: int

    @classmethod
    def from_environment(cls) -> "Settings":
        # MODEL_ID is accepted as a short alias for command-line examples.
        translation_model_id = _env(
            "TRANSLATION_MODEL_ID",
            _env("MODEL_ID", ""),
        )
        vllm_base_url = _env("VLLM_BASE_URL", "")
        translation_backend = _env("TRANSLATION_BACKEND", "auto").lower() or "auto"
        if translation_backend == "auto":
            translation_backend = "openai" if vllm_base_url else "hf"
        if translation_backend not in {"hf", "openai"}:
            raise ValueError("TRANSLATION_BACKEND must be auto, hf, or openai")
        # HF_HOME is honored directly by huggingface_hub/transformers.  Only
        # pass cache_dir when the caller provided the more specific override.
        cache_dir = _env("HF_CACHE_DIR", "")
        cache_dir = cache_dir or None
        hf_token = _env("HF_TOKEN", "") or None
        return cls(
            translation_backend=translation_backend,
            translation_model_id=translation_model_id,
            vllm_base_url=vllm_base_url,
            vllm_model=_env("VLLM_MODEL", ""),
            vllm_api_key=_env("VLLM_API_KEY", "") or None,
            vllm_timeout_seconds=_float("VLLM_TIMEOUT_SECONDS", 120.0, 1.0),
            latency_mode=_env("LATENCY_MODE", "native").lower() or "native",
            idle_force_seconds=_float("IDLE_FORCE_SECONDS", 0.0, 0.0),
            asr_ws_url=_env("ASR_WS_URL", ""),
            asr_secret_key=_env("ASR_SECRET_KEY", ""),
            asr_use_vad=_bool("ASR_USE_VAD", False),
            asr_mode=_env("ASR_MODE", "slow") or "slow",
            asr_smooth=_bool("ASR_SMOOTH", False),
            asr_connect_timeout_seconds=_float("ASR_CONNECT_TIMEOUT_SECONDS", 10.0, 1.0),
            asr_recv_timeout_seconds=_float("ASR_RECV_TIMEOUT_SECONDS", 30.0, 1.0),
            translation_revision=_env("TRANSLATION_REVISION", "") or None,
            hf_cache_dir=cache_dir,
            hf_token=hf_token,
            local_files_only=_bool("HF_LOCAL_FILES_ONLY", False),
            trust_remote_code=_bool("HF_TRUST_REMOTE_CODE", False),
            translation_device=_env("TRANSLATION_DEVICE", "auto") or "auto",
            translation_dtype=_env("TRANSLATION_DTYPE", "auto") or "auto",
            translation_max_new_tokens=_int("MAX_NEW_TOKENS", 128, 1),
            translation_force_max_new_tokens=_int(
                "FORCE_MAX_NEW_TOKENS", 128, 1
            ),
            translation_min_new_tokens_on_force=_int(
                "FORCE_MIN_NEW_TOKENS", 1, 0
            ),
            asr_sample_rate=_int("ASR_SAMPLE_RATE", 16_000, 8_000),
            # Empty lets the direction pick Chinese/English; "zhen" forces the
            # service's mixed-language mode.
            asr_language=_env("ASR_LANGUAGE", ""),
            max_audio_chunk_bytes=_int("MAX_AUDIO_CHUNK_BYTES", 262_144, 1_024),
            max_session_audio_bytes=_int(
                "MAX_SESSION_AUDIO_BYTES", 134_217_728, 1_024
            ),
            force_break_threshold=_int("FORCE_BREAK_THRESHOLD", 20, 1),
            max_buffer_units=_int("MAX_BUFFER_UNITS", 200, 1),
            punct_force=_bool("PUNCT_FORCE", False),
            history_window=_int("HISTORY_WINDOW", 30, 0),
            session_ttl_seconds=_int("SESSION_TTL_SECONDS", 300, 1),
            max_active_sessions=_int("MAX_ACTIVE_SESSIONS", 16, 1),
            host=_env("HOST", "127.0.0.1") or "127.0.0.1",
            port=_int("PORT", 8000, 1),
        )

    @property
    def translation_configured(self) -> bool:
        if self.translation_backend == "openai":
            return bool(self.vllm_base_url and self.vllm_model)
        return bool(self.translation_model_id)

    @property
    def asr_configured(self) -> bool:
        return bool(self.asr_ws_url)

    @property
    def translation_name(self) -> str:
        return self.vllm_model if self.translation_backend == "openai" else self.translation_model_id
