from __future__ import annotations

import asyncio
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
        self._lock = asyncio.Lock()

    async def get(self, key: QACacheKey) -> dict[str, Any] | None:
        if self.ttl_seconds <= 0:
            return None
        async with self._lock:
            self._purge_expired()
            entry = self._items.get(key)
            if not entry:
                return None
            self._items.move_to_end(key)
            return dict(entry.value)

    async def set(self, key: QACacheKey, value: dict[str, Any]) -> None:
        if self.ttl_seconds <= 0:
            return
        async with self._lock:
            self._purge_expired()
            self._items[key] = _CacheEntry(
                value=dict(value),
                expires_at=time.monotonic() + self.ttl_seconds,
            )
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, entry in self._items.items() if entry.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)


def normalize_cache_query(query: str) -> str:
    return " ".join(query.casefold().split())
