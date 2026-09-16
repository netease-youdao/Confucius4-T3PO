"""Command-line and Python API for text-only streaming translation.

Examples (after installing this project)::

    # In-process Hugging Face weights
    python -m inference.text_inference --model-id ORG/MODEL --direction zh2en

    # vLLM / SGLang OpenAI-compatible endpoint
    python -m inference.text_inference --backend openai \\
        --base-url http://127.0.0.1:8010/v1 --model-id MODEL --direction zh2en

Each input line is treated as the next source increment.  The model may emit
an empty response (WAIT); non-empty responses are printed as incremental
translation events and are never retranslated by this wrapper.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Iterable, Iterator

from .glossary import normalize_terms
from .latency import (
    DEFAULT_LATENCY_MODE,
    LATENCY_MODES,
    build_logit_bias,
    normalize_tau,
    resolve_latency_mode,
)
from .runtime import run_blocking
from .translation import (
    Direction,
    EngineConfig,
    HFTextGenerator,
    OpenAICompatibleGenerator,
    TranslationEngine,
    split_source,
)


class StreamingTextTranslator:
    """Convenient stateful text API for local HF or OpenAI-compatible serving.

    ``backend="hf"`` loads the checkpoint in-process.  ``backend="openai"``
    talks to a vLLM/SGLang OpenAI-compatible endpoint, which is the usual
    choice for the 14B checkpoint because weights stay in the server process.

    ``latency_mode`` selects a quality-latency operating point (``low``,
    ``native``, ``high``); ``wait_margin_threshold`` overrides it with a raw
    tau.  Both require the ``openai`` backend, which is where the WAIT bias is
    applied.
    """

    def __init__(
        self,
        model_id: str = "",
        direction: Direction = "zh2en",
        *,
        backend: str = "hf",
        base_url: str = "",
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        latency_mode: str = DEFAULT_LATENCY_MODE,
        wait_margin_threshold: float | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 128,
        force_max_new_tokens: int | None = None,
        min_new_tokens_on_force: int = 1,
        force_break_threshold: int = 20,
        max_buffer_units: int = 200,
        punct_force: bool = False,
        history_window: int = 30,
        terms: object = None,
        generator: Any | None = None,
    ) -> None:
        backend = str(backend or "hf").strip().lower()
        if backend not in {"hf", "openai"}:
            raise ValueError("backend must be 'hf' or 'openai'")
        mode = resolve_latency_mode(latency_mode)
        tau = mode.tau if wait_margin_threshold is None else normalize_tau(wait_margin_threshold)
        self.latency_mode = mode.name
        self.wait_margin_threshold = tau
        if generator is not None:
            self.generator = generator
        elif backend == "openai":
            self.generator = OpenAICompatibleGenerator(
                base_url,
                model_id,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_new_tokens=max_new_tokens,
                force_max_new_tokens=force_max_new_tokens,
                min_tokens_on_force=min_new_tokens_on_force,
                logit_bias=build_logit_bias(tau, mode.stop_token_ids, mode.bias_scales),
                repetition_penalty=mode.repetition_penalty,
            )
        else:
            if tau != 0.0:
                raise ValueError(
                    "a non-native latency mode requires backend='openai'; the "
                    "local Hugging Face backend does not apply the WAIT bias"
                )
            self.generator = HFTextGenerator.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                device=device,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
                force_max_new_tokens=force_max_new_tokens,
            )

        async def complete(message: str, force: bool) -> str:
            return str(await run_blocking(self.generator, message, force) or "")

        self.engine = TranslationEngine(
            EngineConfig(
                direction=direction,
                force_break_threshold=force_break_threshold,
                max_buffer_units=max_buffer_units,
                punct_force=punct_force,
                history_window=history_window,
                terms=normalize_terms(terms),
            ),
            complete,
        )

    async def feed(self, text: str) -> list[dict]:
        return await self.engine.feed(text)

    async def flush(self) -> list[dict]:
        return await self.engine.flush()

    @property
    def stats(self) -> dict:
        return self.engine.stats


def _load_glossary(path: str | None) -> list[dict[str, str]]:
    """Read ``source=target`` pairs, skipping blanks and ``#`` comments."""

    if not path:
        return []
    pairs: list[dict[str, str]] = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        source, target = line.split("=", 1)
        pairs.append({"src": source.strip(), "trg": target.strip()})
    return pairs


def _chunks_from_text(text: str, direction: Direction, chunk_size: int) -> Iterator[str]:
    if chunk_size <= 0:
        yield text
        return
    units = split_source(text, direction)
    for start in range(0, len(units), chunk_size):
        yield "".join(units[start : start + chunk_size]) if direction == "zh2en" else " ".join(units[start : start + chunk_size])


def _print_events(events: Iterable[dict], *, json_events: bool) -> None:
    for event in events:
        if json_events:
            print(json.dumps(event, ensure_ascii=False), flush=True)
        elif event.get("type") == "translation" and event.get("text"):
            print(str(event["text"]), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EOS-style streaming translation against local Hugging Face weights "
            "or a vLLM/SGLang OpenAI-compatible endpoint."
        )
    )
    parser.add_argument("--model-id", default=os.getenv("TRANSLATION_MODEL_ID", ""))
    parser.add_argument("--direction", choices=("zh2en", "en2zh"), default="zh2en")
    parser.add_argument(
        "--backend",
        choices=("auto", "hf", "openai"),
        default=os.getenv("TRANSLATION_BACKEND", "auto"),
        help="auto selects openai when --base-url/VLLM_BASE_URL is set, otherwise hf",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", ""),
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8010/v1",
    )
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY") or None)
    parser.add_argument(
        "--latency-mode",
        choices=tuple(LATENCY_MODES),
        default=os.getenv("LATENCY_MODE", DEFAULT_LATENCY_MODE),
        help=(
            "quality-latency operating point: "
            + "; ".join(f"{m.name} (tau={m.tau:g})" for m in LATENCY_MODES.values())
        ),
    )
    parser.add_argument(
        "--wait-margin-threshold",
        type=float,
        default=(
            float(os.environ["WAIT_MARGIN_THRESHOLD"])
            if os.getenv("WAIT_MARGIN_THRESHOLD")
            else None
        ),
        help="raw tau override; positive lowers latency, negative raises it",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=float(os.getenv("VLLM_TIMEOUT_SECONDS", "120"))
    )
    parser.add_argument("--min-new-tokens-on-force", type=int, default=1)
    parser.add_argument(
        "--glossary",
        default=None,
        help=(
            "terminology file, one 'source=target' pair per line; only terms "
            "occurring in a segment are added to that segment's request"
        ),
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--force-max-new-tokens", type=int, default=None)
    parser.add_argument("--force-break-threshold", type=int, default=20)
    parser.add_argument("--max-buffer-units", type=int, default=200)
    parser.add_argument("--punct-force", action="store_true")
    parser.add_argument("--history-window", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=0, help="Split --text into this many source units")
    parser.add_argument("--text", default=None, help="Translate one text value instead of reading stdin")
    parser.add_argument("--json-events", action="store_true", help="Print machine-readable WAIT/TRANS events")
    parser.add_argument("--no-final-flush", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> int:
    base_url = str(args.base_url or "").strip()
    backend = str(args.backend or "auto").strip().lower()
    if backend == "auto":
        backend = "openai" if base_url else "hf"
    model_id = args.model_id.strip()
    if backend == "openai":
        # VLLM_MODEL names the served model, which may differ from the HF repo id.
        model_id = model_id or os.getenv("VLLM_MODEL", "").strip()
        if not base_url:
            raise SystemExit("--base-url (or VLLM_BASE_URL) is required for the openai backend")
        if not model_id:
            raise SystemExit("--model-id (or VLLM_MODEL) is required for the openai backend")
    elif not model_id:
        raise SystemExit("--model-id (or TRANSLATION_MODEL_ID) is required")
    translator = StreamingTextTranslator(
        model_id,
        args.direction,
        backend=backend,
        base_url=base_url,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        latency_mode=args.latency_mode,
        wait_margin_threshold=args.wait_margin_threshold,
        min_new_tokens_on_force=args.min_new_tokens_on_force,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=args.token,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        force_max_new_tokens=args.force_max_new_tokens,
        force_break_threshold=args.force_break_threshold,
        max_buffer_units=args.max_buffer_units,
        punct_force=args.punct_force,
        history_window=args.history_window,
        terms=_load_glossary(args.glossary),
    )

    if args.text is not None:
        chunks = _chunks_from_text(args.text, args.direction, args.chunk_size)
    else:
        chunks = (line.rstrip("\n") for line in sys.stdin)
    for chunk in chunks:
        if not chunk.strip():
            continue
        _print_events(await translator.feed(chunk), json_events=args.json_events)
    if not args.no_final_flush:
        _print_events(await translator.flush(), json_events=args.json_events)
    if args.json_events:
        print(json.dumps({"type": "stats", **translator.stats}, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
