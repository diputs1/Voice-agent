import pytest

from app.agents.graph import SafariAgentGraph
from app.config import Settings
from app.embeddings import EmbeddingProvider
from app.kb import InMemoryKnowledgeBase, KnowledgeHit


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


@pytest.mark.asyncio
async def test_follow_up_without_history_uses_original_query():
    settings = Settings(openai_api_key=None)
    kb = RecordingKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Thế còn giá vé trẻ em thì sao?"}, "empty-thread")

    assert result["search_query"] == "Thế còn giá vé trẻ em thì sao?"
    assert result["rewrite_source"] == "original"
    assert kb.last_query == "Thế còn giá vé trẻ em thì sao?"


@pytest.mark.asyncio
async def test_follow_up_uses_heuristic_rewrite_when_llm_is_unavailable():
    settings = Settings(openai_api_key=None)
    kb = RecordingKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    await kb.save_thread_turn(
        "rewrite-thread",
        "Tôi muốn hỏi về giá vé Vinpearl Safari Phú Quốc",
        "Bạn nên kiểm tra website chính thức.",
    )
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "thế còn giá vé trẻ em thì sao?"}, "rewrite-thread")

    assert result["rewrite_source"] == "heuristic"
    assert "Tôi muốn hỏi về giá vé Vinpearl Safari Phú Quốc" in result["search_query"]
    assert "thế còn giá vé trẻ em thì sao?" in result["search_query"]
    assert kb.last_query == result["search_query"]


@pytest.mark.asyncio
async def test_follow_up_uses_llm_rewrite_when_available():
    settings = Settings(openai_api_key=None)
    kb = RecordingKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    await kb.save_thread_turn(
        "llm-rewrite-thread",
        "Tôi muốn hỏi về giá vé Vinpearl Safari Phú Quốc",
        "Bạn nên kiểm tra website chính thức.",
    )
    graph = SafariAgentGraph(kb, settings)
    graph.llm = FakeLLM(
        [
            '{"search_query":"Giá vé trẻ em Vinpearl Safari Phú Quốc"}',
            "Bạn nên kiểm tra giá vé trẻ em trên website chính thức.",
        ]
    )

    result = await graph.ainvoke({"transcript": "thế còn trẻ em thì sao?"}, "llm-rewrite-thread")

    assert result["rewrite_source"] == "llm"
    assert result["search_query"] == "Giá vé trẻ em Vinpearl Safari Phú Quốc"
    assert kb.last_query == "Giá vé trẻ em Vinpearl Safari Phú Quốc"


@pytest.mark.asyncio
async def test_low_confidence_threshold_is_configurable():
    settings = Settings(openai_api_key=None, low_confidence_threshold=0.95)
    kb = LowConfidenceKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Có khu vực gửi hành lý không?"}, "low-confidence-thread")

    assert result["handoff_required"] is True
    assert result["handoff_reason"] == "low_confidence"
    assert result["recommended_action"] == "check_official_site"


@pytest.mark.asyncio
async def test_small_talk_routes_directly_without_kb_search():
    settings = Settings(openai_api_key=None)
    kb = FailIfSearchKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Xin chào"}, "small-talk-thread")

    assert result["intent"] == "small_talk"
    assert result["route"] == "direct_response"
    assert result["handoff_required"] is False
    assert result["citations"] == []
    assert "Vinpearl Safari Phú Quốc" in result["answer"]


@pytest.mark.asyncio
async def test_out_of_scope_routes_directly_without_kb_search():
    settings = Settings(openai_api_key=None)
    kb = FailIfSearchKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Thời tiết Hà Nội hôm nay thế nào?"}, "oos-thread")

    assert result["intent"] == "out_of_scope"
    assert result["route"] == "direct_response"
    assert result["handoff_required"] is False
    assert "chỉ hỗ trợ" in result["answer"].lower()


@pytest.mark.asyncio
async def test_streaming_pre_answer_uses_direct_node_for_small_talk():
    settings = Settings(openai_api_key=None)
    kb = FailIfSearchKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    nodes = []
    async for node_name, _, _ in graph.astream_pre_answer({"transcript": "Xin chào"}, "stream-direct"):
        nodes.append(node_name)

    assert nodes == ["supervisor", "direct_response"]


class RecordingKnowledgeBase(InMemoryKnowledgeBase):
    def __init__(self, embeddings: EmbeddingProvider) -> None:
        super().__init__(embeddings)
        self.last_query = ""

    async def search(self, query: str, limit: int = 5):
        self.last_query = query
        return await super().search(query, limit)


class LowConfidenceKnowledgeBase(InMemoryKnowledgeBase):
    async def search(self, query: str, limit: int = 5):
        return [
            KnowledgeHit(
                id="low-score",
                title="Vinpearl Safari Phú Quốc",
                section="Dịch vụ",
                category="service",
                content="Trong khuôn viên có các dịch vụ hỗ trợ khách tham quan.",
                source_url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
                language="vi",
                crawled_at=None,
                valid_until=None,
                metadata={},
                score=0.1,
            )
        ]


class FailIfSearchKnowledgeBase(InMemoryKnowledgeBase):
    async def search(self, query: str, limit: int = 5):
        raise AssertionError("KB search should not be called for direct routes")


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
