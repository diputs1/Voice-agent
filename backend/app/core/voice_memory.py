from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VoiceMemoryTurn:
    transcript: str
    search_query: str
    site_id: str | None
    created_at: float


class VoiceConversationMemory:
    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_conversations: int,
        max_turns_per_conversation: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_conversations = max_conversations
        self.max_turns_per_conversation = max_turns_per_conversation
        self._clock = clock
        self._items: OrderedDict[str, deque[VoiceMemoryTurn]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def history(self, conversation_id: str | None) -> list[dict[str, Any]]:
        if not conversation_id or self.ttl_seconds <= 0 or self.max_conversations <= 0:
            return []
        async with self._lock:
            turns = self._items.get(conversation_id)
            if not turns:
                return []
            self._purge_expired_turns(turns)
            if not turns:
                self._items.pop(conversation_id, None)
                return []
            self._items.move_to_end(conversation_id)
            return [
                {
                    "transcript": turn.search_query or turn.transcript,
                    "raw_transcript": turn.transcript,
                    "site_id": turn.site_id,
                    "created_at": turn.created_at,
                }
                for turn in turns
            ]

    async def append(
        self,
        *,
        conversation_id: str | None,
        transcript: str,
        search_query: str,
        site_id: str | None,
    ) -> None:
        if not conversation_id or self.ttl_seconds <= 0 or self.max_conversations <= 0:
            return
        async with self._lock:
            turns = self._items.setdefault(conversation_id, deque())
            self._purge_expired_turns(turns)
            turns.append(
                VoiceMemoryTurn(
                    transcript=transcript,
                    search_query=search_query,
                    site_id=site_id,
                    created_at=self._clock(),
                )
            )
            while len(turns) > self.max_turns_per_conversation:
                turns.popleft()
            self._items.move_to_end(conversation_id)
            while len(self._items) > self.max_conversations:
                self._items.popitem(last=False)

    def _purge_expired_turns(self, turns: deque[VoiceMemoryTurn]) -> None:
        oldest_allowed = self._clock() - self.ttl_seconds
        while turns and turns[0].created_at <= oldest_allowed:
            turns.popleft()
