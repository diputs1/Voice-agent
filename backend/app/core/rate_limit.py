from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable


class InMemoryRateLimiter:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        if limit <= 0 or window_seconds <= 0:
            return True

        now = self._clock()
        oldest_allowed = now - window_seconds
        async with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= oldest_allowed:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True
