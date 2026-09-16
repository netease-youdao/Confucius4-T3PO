"""Pre-download the public translation checkpoint.

This is optional because ``transformers``/``vllm`` download models lazily. It
is useful for air-gapped deployments. ASR is not downloaded here: it runs in
the separately released Confucius4-R2T2 WebSocket service, deployed from its
own repository.
"""

from __future__ import annotations

import argparse
import os


def _download(repo_id: str, *, revision: str | None, cache_dir: str | None, token: str | None) -> str:
    if not repo_id.strip():
        raise ValueError("model repository id cannot be empty")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency environment
        raise SystemExit("install huggingface_hub before downloading models") from error
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the released translation checkpoint into the local cache")
    parser.add_argument("--translation-model", default=os.getenv("TRANSLATION_MODEL_ID", ""))
    parser.add_argument("--translation-revision", default=os.getenv("TRANSLATION_REVISION") or None)
    parser.add_argument("--cache-dir", default=os.getenv("HF_CACHE_DIR") or None)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.translation_model:
        raise SystemExit("--translation-model (or TRANSLATION_MODEL_ID) is required")
    path = _download(
        args.translation_model,
        revision=args.translation_revision,
        cache_dir=args.cache_dir,
        token=args.token,
    )
    print(f"translation: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
