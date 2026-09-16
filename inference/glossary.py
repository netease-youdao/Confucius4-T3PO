"""Optional terminology support for the streaming translation prompt.

A caller-supplied glossary pins how known terms are rendered. Three decisions
matter for quality and cost, and all three are load-bearing:

* The block goes AFTER ``<STREAMING_HISTORY>`` and before ``<CURRENT_INPUT>``,
  so everything earlier in the request is byte-identical across calls and vLLM
  can still reuse the prefix cache.
* Only terms that actually occur in the current source segment are rendered.
  When nothing matches, the block is omitted entirely and the request is
  byte-for-byte identical to a session without a glossary.
* The prompt frames the list as a reference, never as source text to translate
  or as already-delivered history, and warns against applying a term to an
  unrelated substring.
"""

from __future__ import annotations

from .prompts import Direction

# Cap the number of pinned terms. A long list crowds out the actual input and
# the model starts pattern-matching the glossary instead of translating.
MAX_TERMS = 200

# Terms shorter than this (after removing spaces) are matched literally. The
# space-stripped match lets "bankruptcy remote" still hit a segment whose
# spacing ASR mangled, but for very short terms it would fire on unrelated
# substrings far too often.
_NOSPACE_MIN = 3

_HEAD_COMMON = (
    "### Terminology (reference only)\n"
    "If any of the following terms occurs in the CURRENT source input, render it with the "
    "specified translation for consistency. This list is a glossary reference, NOT source "
    "text to translate and NOT already-delivered history.\n"
)
_HEAD_TAIL = (
    "⚠️ Caution: A term's source form may coincidentally appear as a substring of a longer "
    "word or phrase. Only apply the specified translation when the term is used independently "
    "with its intended meaning — do NOT force-apply it to unrelated substrings or different "
    "senses:"
)
_HEAD_EN2ZH = (
    _HEAD_COMMON
    + "Use a term ONLY where it genuinely occurs in the input you are translating, and keep the "
    "rest of the sentence in natural Chinese word order. This list only chooses the wording "
    "of an existing term — never let it add, drop, repeat, or restructure content, never let "
    "it change the meaning of the surrounding sentence, and never let it replace a non-term "
    "phrase that happens to contain the term's wording.\n"
    + _HEAD_TAIL
)
_HEAD_ZH2EN = (
    _HEAD_COMMON
    + "Use a term ONLY where it genuinely occurs in the input you are translating, and keep the "
    "rest of the sentence in its original natural English. This list only chooses the wording "
    "of an existing term — never let it add, drop, repeat, or restructure content, and never "
    "let it replace a non-term phrase that happens to contain the term's wording.\n"
    + _HEAD_TAIL
)


def normalize_terms(raw: object) -> tuple[tuple[str, str], ...]:
    """Return de-duplicated ``(source, target)`` pairs from caller input.

    Accepts a list of ``{"src", "trg"}`` mappings or of two-item sequences, so
    the wire format stays forgiving without letting malformed entries through.
    Entries beyond :data:`MAX_TERMS` are dropped.
    """

    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if isinstance(item, dict):
            source = str(item.get("src") or item.get("source") or "").strip()
            target = str(item.get("trg") or item.get("target") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            source, target = (str(item[0] or "").strip(), str(item[1] or "").strip())
        else:
            continue
        if not source or not target or (source, target) in seen:
            continue
        seen.add((source, target))
        out.append((source, target))
        if len(out) >= MAX_TERMS:
            break
    return tuple(out)


def terms_in_segment(
    terms: tuple[tuple[str, str], ...], segment: str
) -> tuple[tuple[str, str], ...]:
    """Select the terms that actually occur in this source segment."""

    if not terms:
        return ()
    segment = segment or ""
    segment_nospace = segment.replace(" ", "")
    hits: list[tuple[str, str]] = []
    for source, target in terms:
        source_nospace = source.replace(" ", "")
        if len(source_nospace) >= _NOSPACE_MIN:
            matched = source_nospace in segment_nospace
        else:
            matched = source in segment
        if matched:
            hits.append((source, target))
    return tuple(hits)


def build_glossary_block(
    terms: tuple[tuple[str, str], ...], direction: Direction
) -> str:
    """Render the selected terms as the prompt's terminology block."""

    if not terms:
        return ""
    head = _HEAD_ZH2EN if direction == "zh2en" else _HEAD_EN2ZH
    lines = "\n".join(f"- {source} -> {target}" for source, target in terms)
    return f"{head}\n{lines}"
