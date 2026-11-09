from __future__ import annotations

import asyncio
import ctypes
import gc
import weakref
from collections.abc import Sequence
from typing import Any, TypeVar

_T = TypeVar("_T")


def _load_malloc_trim():
    try:
        function = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return None
    function.argtypes = [ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


_MALLOC_TRIM = _load_malloc_trim()


def release_process_memory() -> None:
    """Release unreachable Python objects and return free arenas to Linux."""
    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


class ProcessRolloutLimiter:
    """Share one asyncio semaphore across agent loops in a worker process."""

    _SEMAPHORES: weakref.WeakKeyDictionary[
        Any, dict[int, asyncio.Semaphore]
    ] = weakref.WeakKeyDictionary()

    @classmethod
    def semaphore(cls, limit: int) -> asyncio.Semaphore:
        if limit < 1:
            raise ValueError("rollout limit must be positive")
        loop = asyncio.get_running_loop()
        per_loop = cls._SEMAPHORES.setdefault(loop, {})
        return per_loop.setdefault(limit, asyncio.Semaphore(limit))


def round_robin_slice(
    items: Sequence[_T],
    count: int,
    cursor: int,
) -> tuple[list[_T], int]:
    """Return consecutive round-robin items and the next cursor."""
    if not items or count <= 0:
        return [], cursor % len(items) if items else 0
    start = cursor % len(items)
    selected = [
        items[(start + offset) % len(items)]
        for offset in range(count)
    ]
    return selected, (start + count) % len(items)
