import pytest

from app.knowledge.embeddings import EmbeddingProvider


@pytest.mark.asyncio
async def test_embed_query_uses_lru_cache_and_returns_copy():
    provider = EmbeddingProvider("test-key", "test-model", dimensions=3)
    fake = FakeOpenAIEmbeddings()
    provider._openai = fake

    first = await provider.embed_query("same question")
    first[0] = 99.0
    second = await provider.embed_query("same question")

    assert fake.calls == ["same question"]
    assert second == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_embed_query_cache_evicts_oldest_entry():
    provider = EmbeddingProvider("test-key", "test-model", dimensions=3, max_query_cache_entries=1)
    fake = FakeOpenAIEmbeddings()
    provider._openai = fake

    await provider.embed_query("first")
    await provider.embed_query("second")
    await provider.embed_query("first")

    assert fake.calls == ["first", "second", "first"]


class FakeOpenAIEmbeddings:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.embeddings = self

    async def create(self, *, model: str, input: list[str], **kwargs: object) -> object:
        self.calls.extend(input)
        return FakeEmbeddingResponse([[1.0, 2.0, 3.0] for _ in input])


class FakeEmbeddingResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [FakeEmbeddingItem(embedding) for embedding in embeddings]


class FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding
