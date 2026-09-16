"""Quality-latency operating points and how they reach the request payload."""

import pytest

from inference.latency import (
    DEFAULT_REPETITION_PENALTY,
    LATENCY_MODES,
    build_logit_bias,
    describe_modes,
    normalize_tau,
    resolve_latency_mode,
)
from inference.text_inference import StreamingTextTranslator
from inference.translation import OpenAICompatibleGenerator


def test_reported_operating_points_keep_their_calibrated_values() -> None:
    # These are the points in the published comparison; changing them silently
    # would make the release disagree with the report.
    assert LATENCY_MODES["low"].tau == 0.9375009536743164
    assert LATENCY_MODES["native"].tau == 0.0
    assert LATENCY_MODES["high"].tau == -0.39
    for mode in LATENCY_MODES.values():
        assert mode.stop_token_ids == (151643, 151645)
        assert mode.bias_scales == {151643: 1.0, 151645: 1.05}
        assert mode.repetition_penalty == DEFAULT_REPETITION_PENALTY


def test_native_sends_no_bias_at_all() -> None:
    assert build_logit_bias(0.0) is None


def test_positive_tau_lowers_terminal_logits_and_negative_raises_them() -> None:
    low = build_logit_bias(LATENCY_MODES["low"].tau)
    high = build_logit_bias(LATENCY_MODES["high"].tau)
    assert low is not None and high is not None
    # Positive tau suppresses the terminal token (fewer WAITs, lower latency).
    assert all(value < 0 for value in low.values())
    # Negative tau favors it (more WAITs, higher latency).
    assert all(value > 0 for value in high.values())
    # The 1.05 scale compensates for vLLM applying repetition penalty after bias.
    assert low[151645] == pytest.approx(low[151643] * 1.05)
    assert high[151645] == pytest.approx(high[151643] * 1.05)


def test_resolve_latency_mode_is_case_insensitive_and_validates() -> None:
    assert resolve_latency_mode("HIGH").name == "high"
    assert resolve_latency_mode(None).name == "native"
    with pytest.raises(ValueError, match="unknown latency mode"):
        resolve_latency_mode("turbo")


def test_normalize_tau_accepts_negative_and_rejects_nonsense() -> None:
    assert normalize_tau(-0.39) == -0.39
    assert normalize_tau("1.5") == 1.5
    for bad in (float("nan"), float("inf"), 25.0, -25.0, True, "abc", None):
        with pytest.raises(ValueError):
            normalize_tau(bad)


def test_describe_modes_is_ordered_low_native_high() -> None:
    assert [m["name"] for m in describe_modes()] == ["low", "native", "high"]


def test_generator_omits_bias_on_force_flush() -> None:
    """The final force call must never be allowed to resolve to WAIT."""

    generator = OpenAICompatibleGenerator(
        "http://127.0.0.1:8010/v1",
        "m",
        logit_bias=build_logit_bias(LATENCY_MODES["low"].tau),
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
    )
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class Client:
        def post(self, url, *, headers, json):
            seen[json.get("min_tokens") is not None] = json
            return Response()

    generator._client = Client()
    generator("probe", False)
    generator("force", True)

    probe_payload = seen[False]
    force_payload = seen[True]
    assert "logit_bias" in probe_payload
    assert probe_payload["repetition_penalty"] == DEFAULT_REPETITION_PENALTY
    assert "logit_bias" not in force_payload
    assert "repetition_penalty" not in force_payload


def test_native_generator_payload_has_no_wait_controls() -> None:
    generator = OpenAICompatibleGenerator(
        "http://127.0.0.1:8010/v1", "m", logit_bias=None, repetition_penalty=1.05
    )
    assert generator.logit_bias is None
    # Repetition penalty is coupled to the bias; without a bias it is dropped so
    # the native request stays byte-for-byte unchanged.
    assert generator.repetition_penalty is None


def test_translator_maps_mode_to_generator_bias() -> None:
    translator = StreamingTextTranslator(
        "m", "zh2en", backend="openai", base_url="http://x/v1", latency_mode="high"
    )
    assert translator.latency_mode == "high"
    assert translator.wait_margin_threshold == -0.39
    assert translator.generator.logit_bias == {151643: 0.39, 151645: pytest.approx(0.4095)}


def test_manual_tau_overrides_the_named_mode() -> None:
    translator = StreamingTextTranslator(
        "m",
        "zh2en",
        backend="openai",
        base_url="http://x/v1",
        latency_mode="native",
        wait_margin_threshold=2.0,
    )
    assert translator.wait_margin_threshold == 2.0
    assert translator.generator.logit_bias == {151643: -2.0, 151645: pytest.approx(-2.1)}


def test_non_native_mode_rejected_on_local_hf_backend() -> None:
    with pytest.raises(ValueError, match="requires backend='openai'"):
        StreamingTextTranslator("m", "zh2en", backend="hf", latency_mode="low")
