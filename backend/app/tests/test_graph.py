import pytest

from app.agents.graph import SafariAgentGraph
from app.config import Settings
from app.embeddings import EmbeddingProvider
from app.kb import InMemoryKnowledgeBase


@pytest.mark.asyncio
async def test_schedule_question_routes_to_answer():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Vinpearl Safari mở cửa mấy giờ?"}, "test-thread")

    assert result["intent"] == "stable"
    assert "09:00" in result["answer"]
    assert result["handoff_required"] is False
    assert result["citations"]


@pytest.mark.asyncio
async def test_ticket_price_question_requires_handoff():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Giá vé hôm nay bao nhiêu?"}, "test-thread-2")

    assert result["intent"] == "time_sensitive"
    assert result["handoff_required"] is True
    assert result["handoff_reason"] == "stale_or_missing_time_sensitive_data"
    assert result["recommended_action"] == "contact_hotline_or_booking"
    assert "kiểm tra" in result["answer"].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Vé vào cổng bao nhiêu tiền?",
        "Hiện còn khuyến mãi không?",
        "Cuối tuần này có áp dụng không?",
    ],
)
async def test_paraphrased_time_sensitive_questions_use_rules_fast_path(question):
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)
    graph.llm = FailIfCalledLLM()

    result = await graph.ainvoke({"transcript": question}, f"thread-{question}")

    assert result["intent"] == "time_sensitive"
    assert result["classifier_source"] == "rules"
    assert result["handoff_required"] is True


@pytest.mark.asyncio
async def test_llm_classifier_fallback_marks_time_sensitive_question():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)
    graph.llm = FakeLLM(['{"intent":"time_sensitive","confidence":0.91}'])

    result = await graph.ainvoke({"transcript": "Đi một người lớn thì cần chuẩn bị ngân sách thế nào?"}, "classifier-thread")

    assert result["intent"] == "time_sensitive"
    assert result["classifier_source"] == "llm"
    assert result["classifier_confidence"] == 0.91
    assert result["handoff_required"] is True


@pytest.mark.asyncio
async def test_ungrounded_quantitative_answer_escalates():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)
    graph.llm = FakeLLM(["Công viên mở cửa từ 08:00 đến 16:00."])

    result = await graph.ainvoke({"transcript": "Vinpearl Safari mở cửa mấy giờ?"}, "grounded-thread")

    assert result["handoff_required"] is True
    assert result["handoff_reason"] == "ungrounded_answer"
    assert "08:00" not in result["answer"]


@pytest.mark.asyncio
async def test_grounded_quantitative_answer_passes():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)
    graph.llm = FakeLLM(["Công viên mở cửa từ 09:00 đến 16:00."])

    result = await graph.ainvoke({"transcript": "Vinpearl Safari mở cửa mấy giờ?"}, "grounded-pass-thread")

    assert result["handoff_required"] is False
    assert result["answer"] == "Công viên mở cửa từ 09:00 đến 16:00."


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLM was called more times than expected")
        return FakeResponse(self.responses.pop(0))


class FailIfCalledLLM:
    async def ainvoke(self, messages):
        raise AssertionError("LLM should not be called for rule fast-path questions")
