import hashlib
import math
import os
from typing import Iterable

from langchain_openai import OpenAIEmbeddings


class EmbeddingProvider:
    def __init__(self, api_key: str | None, model: str, dimensions: int = 1536) -> None:
        self.dimensions = dimensions
        self._openai = None
        if api_key:
            self._openai = OpenAIEmbeddings(model=model, api_key=api_key)

    async def embed_query(self, text: str) -> list[float]:
        if self._openai:
            return await self._openai.aembed_query(text)
        return _deterministic_embedding(text, self.dimensions)

    async def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if self._openai:
            return await self._openai.aembed_documents(items)
        return [_deterministic_embedding(text, self.dimensions) for text in items]


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

