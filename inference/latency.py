"""Quality-latency operating points for the streaming WAIT/TRANS decision.

The model treats an empty (EOS-only) completion as ``WAIT`` and a non-empty one
as ``TRANS``.  A single scalar ``tau`` shifts that first-token decision:

    WAIT iff max(stop logits) - max(non-stop logits) >= tau

``tau`` is applied as an OpenAI-compatible ``logit_bias`` of ``-tau * scale`` on
the terminal tokens, which implements the threshold exactly under greedy
decoding.  The sign is what selects the operating point:

* ``tau > 0`` pushes the terminal logits down, so the model waits less and
  commits earlier: lower latency, slightly lower quality.
* ``tau == 0`` sends no bias at all and reproduces the native policy.
* ``tau < 0`` pushes the terminal logits up, so the model waits more: higher
  latency, slightly higher quality.

The three named modes below are the operating points reported for
``Confucius4-T3PO``. ``low`` is the p25 quantile of the observed WAIT-margin
distribution; ``high`` was calibrated to raise the open-loop WAIT rate by about
five percentage points. Both were measured with the scales and repetition
penalty in this module, so changing those invalidates the calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Qwen pad/eos ids. Both act as the terminal (WAIT) token for this checkpoint.
DEFAULT_STOP_TOKEN_IDS: tuple[int, ...] = (151643, 151645)

# vLLM applies its repetition penalty *after* logit_bias, so the second
# terminal token needs a slightly larger bias to keep the effective margin
# aligned. These values are part of the published calibration.
DEFAULT_STOP_TOKEN_BIAS_SCALES: dict[int, float] = {151643: 1.0, 151645: 1.05}
DEFAULT_REPETITION_PENALTY = 1.05

MAX_ABS_TAU = 20.0


@dataclass(frozen=True)
class LatencyMode:
    """One named quality-latency operating point."""

    name: str
    tau: float
    description: str
    stop_token_ids: tuple[int, ...] = DEFAULT_STOP_TOKEN_IDS
    bias_scales: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_STOP_TOKEN_BIAS_SCALES)
    )
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY


# Calibrated for Confucius4-T3PO. These are the points in the reported
# quality-latency comparison.
LATENCY_MODES: dict[str, LatencyMode] = {
    "low": LatencyMode(
        name="low",
        tau=0.9375009536743164,
        description="commits earlier; lowest latency, slightly lower quality",
    ),
    "native": LatencyMode(
        name="native",
        tau=0.0,
        description="the model's own policy, no logit bias applied",
    ),
    "high": LatencyMode(
        name="high",
        tau=-0.39,
        description="waits longer; highest quality, higher latency",
    ),
}

DEFAULT_LATENCY_MODE = "native"


def resolve_latency_mode(name: str | None) -> LatencyMode:
    """Look up a named operating point, case-insensitively."""

    key = str(name or DEFAULT_LATENCY_MODE).strip().lower()
    if not key:
        key = DEFAULT_LATENCY_MODE
    try:
        return LATENCY_MODES[key]
    except KeyError:
        raise ValueError(
            f"unknown latency mode {name!r}; expected one of "
            f"{', '.join(sorted(LATENCY_MODES))}"
        ) from None


def normalize_tau(value: object) -> float:
    """Validate a manual tau override.

    Negative values are allowed on purpose: they are how the high-latency
    operating point is expressed.
    """

    if isinstance(value, bool):
        raise ValueError("wait_margin_threshold must be a finite number")
    try:
        tau = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("wait_margin_threshold must be a finite number") from error
    if not math.isfinite(tau):
        raise ValueError("wait_margin_threshold must be a finite number")
    if abs(tau) > MAX_ABS_TAU:
        raise ValueError(
            f"wait_margin_threshold must be within +/-{MAX_ABS_TAU:g}"
        )
    return tau


def build_logit_bias(
    tau: float,
    stop_token_ids: tuple[int, ...] = DEFAULT_STOP_TOKEN_IDS,
    bias_scales: dict[int, float] | None = None,
) -> dict[int, float] | None:
    """Turn a tau into the ``logit_bias`` map, or ``None`` for native.

    Returning ``None`` at ``tau == 0`` keeps the native request byte-for-byte
    identical to a run without this feature.
    """

    tau = normalize_tau(tau)
    if tau == 0.0:
        return None
    scales = DEFAULT_STOP_TOKEN_BIAS_SCALES if bias_scales is None else bias_scales
    return {
        int(token_id): -tau * float(scales.get(int(token_id), 1.0))
        for token_id in stop_token_ids
    }


def describe_modes() -> list[dict[str, object]]:
    """Serializable summary for the API/UI."""

    return [
        {
            "name": mode.name,
            "tau": mode.tau,
            "description": mode.description,
            "logit_bias": build_logit_bias(mode.tau, mode.stop_token_ids, mode.bias_scales),
        }
        for mode in (LATENCY_MODES[key] for key in ("low", "native", "high"))
    ]
