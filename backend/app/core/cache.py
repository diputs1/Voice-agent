from __future__ import annotations

import asyncio
import heapq
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QACacheKey:
    query: str
    doc_set_hash: str


@dataclass
class _CacheEntry:
    value: dict[str, Any]
    expires_at: float


class TTLQACache:
    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._items: OrderedDict[QACacheKey, _CacheEntry] = OrderedDict()
        self._expirations: list[tuple[float, int, QACacheKey]] = []
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def get(self, key: QACacheKey) -> dict[str, Any] | None:
        if self.ttl_seconds <= 0:
            return None
        async with self._lock:
            entry = self._items.get(key)
            if not entry:
                return None
            if entry.expires_at <= time.monotonic():
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return dict(entry.value)

    async def set(self, key: QACacheKey, value: dict[str, Any]) -> None:
        if self.ttl_seconds <= 0:
            return
        async with self._lock:
            self._purge_expired(limit=max(1, self.max_entries // 8))
            expires_at = time.monotonic() + self.ttl_seconds
            self._items[key] = _CacheEntry(
                value=dict(value),
                expires_at=expires_at,
            )
            self._sequence += 1
            heapq.heappush(self._expirations, (expires_at, self._sequence, key))
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
            if len(self._expirations) > self.max_entries * 4:
                self._rebuild_expiration_heap()

    def _purge_expired(self, *, limit: int) -> None:
        now = time.monotonic()
        purged = 0
        while self._expirations and purged < limit:
            expires_at, _, key = self._expirations[0]
            if expires_at > now:
                break
            heapq.heappop(self._expirations)
            entry = self._items.get(key)
            if entry and entry.expires_at == expires_at:
                self._items.pop(key, None)
                purged += 1

    def _rebuild_expiration_heap(self) -> None:
        self._expirations = []
        for key, entry in self._items.items():
            self._sequence += 1
            self._expirations.append((entry.expires_at, self._sequence, key))
        heapq.heapify(self._expirations)


def normalize_cache_query(query: str) -> str:
    return " ".join(query.casefold().split())
