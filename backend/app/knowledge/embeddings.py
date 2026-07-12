import asyncio
import hashlib
import math
import os
from collections import OrderedDict
from typing import Iterable

from openai import AsyncOpenAI

EMBEDDING_BATCH_SIZE = 128


class EmbeddingProvider:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        dimensions: int = 1536,
        max_query_cache_entries: int = 512,
    ) -> None:
        self.dimensions = dimensions
        self.max_query_cache_entries = max_query_cache_entries
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_lock = asyncio.Lock()
        self._openai = None
        if api_key:
            self._openai = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed_query(self, text: str) -> list[float]:
        cached = await self._get_cached_query_embedding(text)
        if cached is not None:
            return cached
        if self._openai:
            vector = await self._embed_openai([text])
            vector = vector[0]
        else:
            vector = _deterministic_embedding(text, self.dimensions)
        await self._set_cached_query_embedding(text, vector)
        return list(vector)

    async def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        if self._openai:
            return await self._embed_openai(items)
        return [_deterministic_embedding(text, self.dimensions) for text in items]

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            response = await self._openai.embeddings.create(
                model=self.model,
                input=texts[start : start + EMBEDDING_BATCH_SIZE],
                **self._embedding_options(),
            )
            vectors.extend(list(item.embedding) for item in response.data)
        return vectors

    def _embedding_options(self) -> dict[str, int]:
        if self.model.startswith("text-embedding-3"):
            return {"dimensions": self.dimensions}
        return {}

    async def _get_cached_query_embedding(self, text: str) -> list[float] | None:
        if self.max_query_cache_entries <= 0:
            return None
        async with self._query_cache_lock:
            vector = self._query_cache.get(text)
            if vector is None:
                return None
            self._query_cache.move_to_end(text)
            return list(vector)

    async def _set_cached_query_embedding(self, text: str, vector: list[float]) -> None:
        if self.max_query_cache_entries <= 0:
            return
        async with self._query_cache_lock:
            self._query_cache[text] = list(vector)
            self._query_cache.move_to_end(text)
            while len(self._query_cache) > self.max_query_cache_entries:
                self._query_cache.popitem(last=False)


def _deterministic_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    normalized = f" {text.lower()} "
    tokens = normalized.split()
    if not tokens:
        tokens = [normalized]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            idx = int.from_bytes(digest[offset : offset + 2], "big") % dimensions
            sign = 1.0 if digest[offset + 2] % 2 == 0 else -1.0
            vector[idx] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def pgvector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
