"""Prompt and response helpers used by both inference entry points.

Keep these strings in one module: the model card's interleaved-history
protocol is part of the model interface, and changing it independently in the
CLI and web server is an easy source of hard-to-debug quality regressions.
"""

from __future__ import annotations

import re
from typing import Literal

Direction = Literal["zh2en", "en2zh"]

SYSTEM_PROMPT = "You are a helpful assistant."

STREAMING_PROMPTS: dict[str, str] = {
    "zh2en": """### Role
You are a professional Chinese-to-English simultaneous interpreter for live streaming and ASR speech translation, with strict requirements for low latency, high coherence, and natural fluency.

### Context Format
- The conversation history is provided in <STREAMING_HISTORY>, structured as:
  source_text¦translated_text§source_text¦translated_text§...
- The last segment of <STREAMING_HISTORY> is the latest input awaiting translation.

### Input
- The latest chunk from a live ASR speech stream.
- ASR artifacts (fillers, stutters, repetitions) should be ignored.

### Rules
- Output nothing if the available context is still ambiguous.
- Otherwise, output the translation of what has become sufficiently clear.
  Do not assume linear or word-by-word correspondence — reorder and restructure
  as needed for a natural output.
- The new translation must read smoothly as a continuation of the preceding
  translated text.
- Output the translation directly, with no prefix, suffix, or extra markers.""",
    "en2zh": """### Role
You are a professional English-to-Chinese simultaneous interpreter for live streaming and ASR speech translation, with strict requirements for low latency, high coherence, and natural fluency.

### Context Format
- The conversation history is provided in <STREAMING_HISTORY>, structured as:
  source_text¦translated_text§source_text¦translated_text§...
- The last segment of <STREAMING_HISTORY> is the latest input awaiting translation.

### Input
- The latest chunk from a live ASR speech stream.
- ASR artifacts (fillers, stutters, repetitions) should be ignored.

### Rules
- Output nothing if the available context is still ambiguous.
- Otherwise, output the translation of what has become sufficiently clear.
  Do not assume linear or word-by-word correspondence — reorder and restructure
  as needed for a natural output.
- The new translation must read smoothly as a continuation of the preceding
  translated text.
- Output the translation directly, with no prefix, suffix, or extra markers.""",
}


def build_user_message(
    direction: Direction,
    history: str,
    current_input: str,
    *,
    glossary: str = "",
) -> str:
    """Build the exact user-side interleaved-history message.

    The same prompt is used whether or not the caller is forcing a segment. A
    forced segment is produced by requiring at least one new token
    (``min_tokens`` / ``min_new_tokens``), which makes an empty WAIT response
    impossible at the sampling level; a second prompt telling the model not to
    wait measured no better and cost the prefix cache, since it diverged from
    the streaming prompt after 152 characters.

    ``glossary`` is an optional terminology block. It is placed after the
    history so everything before ``<STREAMING_HISTORY>`` stays byte-identical
    across requests, which keeps the vLLM prefix cache usable. An empty block
    leaves the message byte-for-byte the same as a session without a glossary.
    """

    block = f"\n{glossary}\n" if glossary else ""
    return (
        f"{STREAMING_PROMPTS[direction]}\n\n<STREAMING_HISTORY>\n{history}\n{block}\n"
        f"<CURRENT_INPUT>\n{current_input}"
    )


def sanitize_history(text: object) -> str:
    """Prevent a generated segment from corrupting the interleaved protocol."""

    return str(text or "").replace("¦", "｜").replace("§", "；").strip()


_WAIT_ONLY = re.compile(r"^<?\s*WAIT\s*>?$", re.IGNORECASE)
_TRANS_ONLY = re.compile(r"^<?\s*TRANS\s*>?$", re.IGNORECASE)
_TRANS_PREFIX = re.compile(
    r"^(?:<\s*TRANS\s*>\s*|TRANS(?:\s*[:：]\s*|\s+))",
    re.IGNORECASE,
)


def parse_model_response(raw: object) -> tuple[str, str]:
    """Return ``(WAIT|TRANS, text)`` for EOS-style model output.

    The released protocol uses an empty/EOS-only response for WAIT.  A small
    amount of tolerance for an accidental ``WAIT``/``TRANS`` wrapper is useful
    with manually converted checkpoints, but arbitrary explanatory text is
    retained as a translation rather than silently discarded.
    """

    text = str(raw or "").strip()
    text = text.replace("<|im_end|>", "").strip()
    if not text or _WAIT_ONLY.fullmatch(text):
        return "WAIT", ""
    if _TRANS_ONLY.fullmatch(text):
        return "WAIT", ""
    text = _TRANS_PREFIX.sub("", text, count=1)
    text = sanitize_history(text)
    return ("TRANS", text) if text else ("WAIT", "")


def prompt_text(direction: Direction) -> str:
    """Expose the prompt for diagnostics and reproducibility."""

    return STREAMING_PROMPTS[direction]
