"""Local Hugging Face text generation and incremental translation state.

The model is an EOS-style policy: a visible completion is a ``TRANS`` event
and an empty completion is ``WAIT``.  This module contains no ASR or web
specific code, so it can be used from a terminal, a notebook, or the web
server alike.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from .glossary import build_glossary_block, terms_in_segment
from .prompts import build_user_message, parse_model_response, sanitize_history

Direction = Literal["zh2en", "en2zh"]
Completion = Callable[[str, bool], str | Awaitable[str]]

PUNCTUATION_END = frozenset({"。", "！", "？", "!", "?", "；", ";", "…", "～", "~"})
LATIN_TOKEN = re.compile(r"^[A-Za-z0-9]+(?:[._'’-][A-Za-z0-9]+)*$")


def _split_zh2en(text: str) -> list[str]:
    """Split Chinese into character units while keeping ASCII words intact."""

    tokens: list[str] = []
    index = 0
    value = str(text or "")
    while index < len(value):
        char = value[index]
        if char.isspace():
            end = index + 1
            while end < len(value) and value[end].isspace():
                end += 1
            tokens.append(value[index:end])
            index = end
            continue
        if char.isascii() and (char.isalnum() or char in "_'\u2019"):
            end = index + 1
            while end < len(value):
                candidate = value[end]
                if not (candidate.isascii() and (candidate.isalnum() or candidate in "_\u2019'-")):
                    break
                end += 1
            tokens.append(value[index:end])
            index = end
            continue
        tokens.append(char)
        index += 1
    return tokens


def split_source(text: str, direction: Direction) -> list[str]:
    """Split an incoming source increment according to the model's units."""

    if direction == "zh2en":
        return _split_zh2en(text)
    return str(text or "").strip().split()


def join_source(tokens: list[str], direction: Direction) -> str:
    if direction == "zh2en":
        return "".join(tokens)
    return " ".join(tokens)


def source_units(tokens: list[str], direction: Direction) -> int:
    if direction == "zh2en":
        return sum(not token.isspace() for token in tokens)
    return len(tokens)


def join_translation_segments(parts: list[str], direction: Direction) -> str:
    """Join emitted segments without damaging English word boundaries."""

    values = [str(part or "").strip() for part in parts if str(part or "").strip()]
    if direction == "en2zh":
        return "".join(values)
    if not values:
        return ""
    text = " ".join(values)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\b([A-Za-z]+)\s+('(?:s|re|ve|ll|d|m|t)\b)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class EngineConfig:
    direction: Direction
    force_break_threshold: int = 20
    max_buffer_units: int = 200
    punct_force: bool = False
    history_window: int = 30
    # Session-scoped glossary as (source, target) pairs. Empty means the
    # request is built exactly as it would be without this feature.
    terms: tuple[tuple[str, str], ...] = ()


@dataclass
class EngineState:
    buffer: list[str] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    probe_calls: int = 0
    translation_calls: int = 0
    segments: int = 0
    waits: int = 0
    started_at: float = field(default_factory=time.monotonic)


class TranslationEngine:
    """Stateful incremental policy runner.

    ``complete`` receives the user message and a boolean indicating whether
    this segment is forced, which the generator turns into a minimum-token
    requirement so the response cannot come back empty.  It may be synchronous
    or async; the engine normalizes both forms so it remains convenient in
    scripts and FastAPI handlers.
    """

    def __init__(self, config: EngineConfig, complete: Completion) -> None:
        if config.force_break_threshold < 1:
            raise ValueError("force_break_threshold must be positive")
        if config.max_buffer_units < 1:
            raise ValueError("max_buffer_units must be positive")
        self.config = config
        self._complete = complete
        self._state = EngineState()

    @property
    def direction(self) -> Direction:
        return self.config.direction

    def _history_text(self) -> str:
        pairs = self._state.history
        if self.config.history_window >= 0 and len(pairs) > self.config.history_window:
            pairs = pairs[-self.config.history_window :] if self.config.history_window else []
        return "".join(f"{src}¦{target}§" for src, target in pairs)

    def _user_message(self, source: str) -> str:
        # Render only the terms that occur in this segment, so a configured
        # glossary costs nothing on the segments it does not apply to.
        glossary = build_glossary_block(
            terms_in_segment(self.config.terms, source), self.config.direction
        )
        # The message is identical whether or not this segment is forced; the
        # generator enforces a non-empty response via min_tokens instead.
        return build_user_message(
            self.config.direction,
            self._history_text(),
            source,
            glossary=glossary,
        )

    async def _complete_message(self, source: str, force: bool) -> str:
        result = self._complete(self._user_message(source), force)
        if inspect.isawaitable(result):
            result = await result
        return str(result or "")

    def _record_translation(self, source: str, target: str) -> dict[str, Any]:
        clean_source = sanitize_history(source)
        clean_target = sanitize_history(target)
        self._state.history.append((clean_source, clean_target))
        if self.config.history_window >= 0 and len(self._state.history) > self.config.history_window:
            keep = self.config.history_window
            self._state.history = self._state.history[-keep:] if keep else []
        self._state.buffer.clear()
        self._state.segments += 1
        return {
            "type": "translation",
            "source": clean_source,
            "text": clean_target,
            "source_units": source_units(
                split_source(source, self.config.direction), self.config.direction
            ),
        }

    async def _translate_current(self, *, force: bool) -> dict[str, Any] | None:
        source = join_source(self._state.buffer, self.config.direction)
        if not source.strip():
            return None
        if force:
            self._state.translation_calls += 1
        else:
            self._state.probe_calls += 1
        raw = await self._complete_message(source, force)
        _, target = parse_model_response(raw)
        if not target:
            self._state.waits += 1
            return None
        return self._record_translation(source, target)

    async def feed(self, text: str) -> list[dict[str, Any]]:
        """Append a source increment and return zero or one translation event."""

        incoming = split_source(text, self.config.direction)
        incoming = [token for token in incoming if token or token.isspace()]
        if not incoming:
            return []
        if (
            self.config.direction == "zh2en"
            and self._state.buffer
            and LATIN_TOKEN.match(self._state.buffer[-1] or "")
            and LATIN_TOKEN.match(incoming[0] or "")
        ):
            self._state.buffer[-1] += incoming.pop(0)
        self._state.buffer.extend(incoming)
        units = source_units(self._state.buffer, self.config.direction)
        if not units:
            return []

        if units >= self.config.max_buffer_units:
            event = await self._translate_current(force=True)
            return [event] if event else [{"type": "wait", "source_units": units}]
        if self.config.punct_force and any(
            any(mark in token for mark in PUNCTUATION_END) for token in self._state.buffer
        ):
            event = await self._translate_current(force=True)
            return [event] if event else [{"type": "wait", "source_units": units}]
        # A trailing ASCII word is often still being spoken.  Waiting for the
        # next ASR update avoids translating ``prioritiz`` and then repeating it.
        if (
            self.config.direction == "zh2en"
            and self._state.buffer
            and LATIN_TOKEN.match(self._state.buffer[-1] or "")
        ):
            return []
        if units >= self.config.force_break_threshold:
            event = await self._translate_current(force=True)
            return [event] if event else [{"type": "wait", "source_units": units}]

        event = await self._translate_current(force=False)
        return [event] if event else [{"type": "wait", "source_units": units}]

    async def flush(self) -> list[dict[str, Any]]:
        """Force a final translation of any uncommitted source text."""

        if not self._state.buffer:
            return []
        event = await self._translate_current(force=True)
        if event:
            return [event]
        return [
            {
                "type": "wait",
                "source_units": source_units(self._state.buffer, self.config.direction),
            }
        ]

    @property
    def pending_source(self) -> str:
        return join_source(self._state.buffer, self.config.direction)

    @property
    def history(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._state.history)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "segments": self._state.segments,
            "probe_calls": self._state.probe_calls,
            "translation_calls": self._state.translation_calls,
            "waits": self._state.waits,
            "buffer_units": source_units(self._state.buffer, self.config.direction),
        }


class HFTextGenerator:
    """Thread-safe lazy wrapper around ``transformers`` causal generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 128,
        force_max_new_tokens: int | None = None,
        min_new_tokens_on_force: int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.force_max_new_tokens = max(
            1,
            int(force_max_new_tokens or max_new_tokens),
        )
        self.min_new_tokens_on_force = max(0, int(min_new_tokens_on_force))
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
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
    ) -> "HFTextGenerator":
        if not str(model_id or "").strip():
            raise ValueError("a translation model id or local path is required")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "transformers and torch are required for local inference; "
                "install the project dependencies first"
            ) from error

        common: dict[str, Any] = {
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "token": token,
        }
        common = {key: value for key, value in common.items() if value is not None}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **common)

        torch_dtype: Any = "auto"
        if dtype and dtype != "auto":
            aliases = {
                "fp16": torch.float16,
                "float16": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }
            try:
                torch_dtype = aliases[dtype.lower()]
            except KeyError as error:
                raise ValueError(f"unsupported TRANSLATION_DTYPE: {dtype}") from error

        model_kwargs = dict(common)
        model_kwargs["torch_dtype"] = torch_dtype
        use_auto_map = device == "auto" and torch.cuda.is_available()
        if use_auto_map:
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if device not in {"auto", ""} and not use_auto_map:
            model.to(device)
        model.eval()
        return cls(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            force_max_new_tokens=force_max_new_tokens,
            min_new_tokens_on_force=min_new_tokens_on_force,
        )

    def _model_device(self) -> Any:
        try:
            return self.model.device
        except Exception:
            try:
                return next(self.model.parameters()).device
            except Exception:
                return None

    def _format_prompt(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        # Qwen-compatible fallback for tokenizers without a registered chat
        # template.  Users can still provide a custom tokenizer with the
        # normal Hugging Face interface.
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def __call__(self, user_message: str, force: bool = False) -> str:
        prompt = self._format_prompt(user_message)
        with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = self._model_device()
            if device is not None:
                inputs = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            kwargs: dict[str, Any] = {
                "max_new_tokens": self.force_max_new_tokens if force else self.max_new_tokens,
                "do_sample": False,
            }
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                kwargs["eos_token_id"] = eos_token_id
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                kwargs["pad_token_id"] = pad_token_id
            if force and self.min_new_tokens_on_force:
                kwargs["min_new_tokens"] = self.min_new_tokens_on_force
            output = self.model.generate(**inputs, **kwargs)
            sequence = output[0] if hasattr(output, "__getitem__") else output
            input_ids = inputs.get("input_ids")
            if input_ids is not None:
                sequence = sequence[input_ids.shape[-1] :]
            return self.tokenizer.decode(sequence, skip_special_tokens=True).strip()


class OpenAICompatibleGenerator:
    """Synchronous client for a vLLM/SGLang OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_new_tokens: int = 128,
        force_max_new_tokens: int | None = None,
        min_tokens_on_force: int = 1,
        logit_bias: dict[int, float] | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        root = str(base_url or "").strip().rstrip("/")
        if not root:
            raise ValueError("VLLM_BASE_URL is required for the openai backend")
        if not str(model or "").strip():
            raise ValueError("VLLM_MODEL is required for the openai backend")
        if root.endswith("/chat/completions"):
            self.url = root
        elif root.endswith("/v1"):
            self.url = f"{root}/chat/completions"
        else:
            self.url = f"{root}/v1/chat/completions"
        self.model = model.strip()
        self.api_key = str(api_key or "").strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.force_max_new_tokens = max(1, int(force_max_new_tokens or max_new_tokens))
        self.min_tokens_on_force = max(0, int(min_tokens_on_force))
        # WAIT-margin controls. Only sent when a non-native operating point is
        # selected, so the native request stays byte-for-byte unchanged.
        self.logit_bias = dict(logit_bias) if logit_bias else None
        self.repetition_penalty = (
            float(repetition_penalty) if self.logit_bias and repetition_penalty else None
        )
        self._client: Any | None = None
        self._lock = threading.Lock()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as error:  # pragma: no cover - dependency environment
            raise RuntimeError("httpx is required for the openai translation backend") from error
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout_seconds, connect=min(15.0, self.timeout_seconds)),
        )
        return self._client

    def __call__(self, user_message: str, force: bool = False) -> str:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.force_max_new_tokens if force else self.max_new_tokens,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ],
        }
        if force and self.min_tokens_on_force:
            payload["min_tokens"] = self.min_tokens_on_force
        # The WAIT bias must never reach the final force-flush request: that
        # call has to produce the remaining translation and must not be allowed
        # to resolve to WAIT.
        if not force and self.logit_bias:
            payload["logit_bias"] = {str(k): v for k, v in self.logit_bias.items()}
            if self.repetition_penalty is not None:
                payload["repetition_penalty"] = self.repetition_penalty
        with self._lock:
            try:
                response = self._ensure_client().post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                return str(response.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception as error:
                raise RuntimeError(
                    "the OpenAI-compatible translation endpoint did not return a usable completion"
                ) from error

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


async def make_local_engine(
    direction: Direction,
    generator: HFTextGenerator,
    *,
    force_break_threshold: int = 20,
    max_buffer_units: int = 200,
    punct_force: bool = False,
    history_window: int = 30,
) -> TranslationEngine:
    """Small convenience factory used by async applications."""

    # Keeping this as an async factory makes call sites symmetrical with model
    # managers that may preload weights in a worker thread in the future.
    await asyncio.sleep(0)
    return TranslationEngine(
        EngineConfig(
            direction=direction,
            force_break_threshold=force_break_threshold,
            max_buffer_units=max_buffer_units,
            punct_force=punct_force,
            history_window=history_window,
        ),
        generator,
    )
