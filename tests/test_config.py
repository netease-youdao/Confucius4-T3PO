def test_settings_can_be_loaded_without_model_ids(monkeypatch):
    from inference.config import Settings

    for name in (
        "TRANSLATION_BACKEND",
        "TRANSLATION_MODEL_ID",
        "MODEL_ID",
        "VLLM_BASE_URL",
        "VLLM_MODEL",
        "ASR_WS_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_environment()
    assert settings.translation_backend == "hf"
    assert settings.translation_configured is False
    assert settings.asr_configured is False
    assert settings.asr_sample_rate == 16_000


def test_asr_configured_reflects_ws_url(monkeypatch):
    from inference.config import Settings

    monkeypatch.delenv("ASR_WS_URL", raising=False)
    assert Settings.from_environment().asr_configured is False

    monkeypatch.setenv("ASR_WS_URL", "ws://localhost:8093/asr_stream_api_v1")
    settings = Settings.from_environment()
    assert settings.asr_configured is True
    assert settings.asr_ws_url == "ws://localhost:8093/asr_stream_api_v1"


def test_asr_settings_default_and_override(monkeypatch):
    from inference.config import Settings

    for name in ("ASR_SECRET_KEY", "ASR_USE_VAD", "ASR_MODE", "ASR_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    defaults = Settings.from_environment()
    assert defaults.asr_secret_key == ""
    assert defaults.asr_use_vad is False
    assert defaults.asr_mode == "slow"
    assert defaults.asr_language == ""

    monkeypatch.setenv("ASR_SECRET_KEY", "test0102")
    monkeypatch.setenv("ASR_USE_VAD", "true")
    monkeypatch.setenv("ASR_MODE", "fast")
    monkeypatch.setenv("ASR_LANGUAGE", "zhen")
    overridden = Settings.from_environment()
    assert overridden.asr_secret_key == "test0102"
    assert overridden.asr_use_vad is True
    assert overridden.asr_mode == "fast"
    assert overridden.asr_language == "zhen"
