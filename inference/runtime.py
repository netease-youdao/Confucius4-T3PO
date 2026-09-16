"""Small async/runtime helpers shared by model-serving modules."""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


async def run_blocking(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one blocking model operation without stalling the event loop.

    A short-lived executor is intentional here.  Some embedded Python 3.13
    builds used by notebook/container runners do not reliably reuse the
    default asyncio executor after a long native-model call.  Model loading
    and ASR/MT calls are relatively coarse operations, so one bounded worker
    per outstanding call is a safer cross-environment compromise than a stuck
    WebSocket session.  The executor is shut down as soon as the call finishes.
    """

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-infer")
    bound = functools.partial(function, *args, **kwargs)
    try:
        return await loop.run_in_executor(executor, bound)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
