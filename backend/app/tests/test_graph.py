import pytest

from app.agents.graph import SafariAgentGraph
from app.core.config import Settings
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.kb import InMemoryKnowledgeBase, KnowledgeHit


@pytest.mark.asyncio
async def test_schedule_question_routes_to_answer():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Vinpearl Safari mở cửa mấy giờ?"}, "test-thread")

    assert result["intent"] == "stable"
    assert result["primary_intent"] == "opening_hours"
    assert result["suggested_route"] == "safari_knowledge"
    assert "09:00" in result["answer"]
    assert result["handoff_required"] is False
    assert result["citations"]


@pytest.mark.asyncio
async def test_vinsafari_product_question_answers_from_knowledge():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Các sản phẩm của VinSafari Phú Quốc."}, "products-thread")

    assert result["intent"] == "stable"
    assert result["suggested_route"] == "safari_knowledge"
    assert "sản phẩm" in result["intent_entities"]
    assert result["handoff_required"] is False
    assert result["retrieved_context"]
    assert "sản phẩm/dịch vụ" in result["answer"].lower()
    assert "kid zoo" in result["answer"].lower()


@pytest.mark.asyncio
async def test_ticket_price_question_requires_handoff():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Giá vé hôm nay bao nhiêu?"}, "test-thread-2")

    assert result["intent"] == "time_sensitive"
    assert result["primary_intent"] == "ticket_price"
    assert result["suggested_route"] == "safari_knowledge"
    assert result["retrieval_debug"]["candidate_count"] > 0
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
        "Có những voucher nào cho VinWonders?",
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
    assert result["suggested_route"] == "safari_knowledge"
    assert result["retrieval_debug"]["candidate_count"] > 0
    assert result["handoff_required"] is True


@pytest.mark.asyncio
async def test_affiliate_info_question_answers_without_noisy_handoff():
    settings = Settings(openai_api_key=None)
    kb = AffiliateKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Thông tin về affiliate"}, "affiliate-thread")

    assert result["intent"] == "stable"
    assert result["suggested_route"] == "safari_knowledge"
    assert result["handoff_required"] is False
    assert "hoa hồng hấp dẫn" in result["answer"]
    assert "Đăng nhập" not in result["answer"]
    assert "Mục lục" not in result["answer"]
    assert "Trang chủ" not in result["answer"]


@pytest.mark.asyncio
async def test_wonderpedia_info_question_uses_wonderpedia_landing_context():
    settings = Settings(openai_api_key=None)
    kb = WonderpediaKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Thông tin về Wonderpedia"}, "wonderpedia-thread")

    assert result["intent"] == "stable"
    assert result["handoff_required"] is False
    assert "cẩm nang" in result["answer"].lower()
    assert "voucher" not in result["answer"].lower()
    assert result["citations"][0]["category"] == "wonderpedia"


@pytest.mark.asyncio
async def test_llm_classifier_fallback_marks_time_sensitive_question():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)
    graph.llm = FakeLLM(
        [
            json_payload(
                {
                    "primary_intent": "ticket_price",
                    "is_time_sensitive": True,
                    "entities": ["ngân sách", "người lớn"],
                    "confidence": 0.91,
                    "suggested_route": "direct_handoff",
                }
            )
        ]
    )

    result = await graph.ainvoke({"transcript": "Đi một người lớn thì cần chuẩn bị ngân sách thế nào?"}, "classifier-thread")

    assert result["intent"] == "time_sensitive"
    assert result["primary_intent"] == "ticket_price"
    assert result["classifier_source"] == "llm_json"
    assert result["classifier_confidence"] == 0.91
    assert result["suggested_route"] == "safari_knowledge"
    assert result["retrieval_debug"]["candidate_count"] > 0
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

    result = await graph.ainvoke({"transcript": "Thế còn khu bò sát thì sao?"}, "empty-thread")

    assert result["search_query"] == "Thế còn khu bò sát thì sao?"
    assert result["rewrite_source"] == "original"
    assert kb.last_query == "Thế còn khu bò sát thì sao?"


@pytest.mark.asyncio
async def test_follow_up_uses_heuristic_rewrite_when_llm_is_unavailable():
    settings = Settings(openai_api_key=None)
    kb = RecordingKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    await kb.save_thread_turn(
        "rewrite-thread",
        "Tôi muốn hỏi về các khu tham quan ở Vinpearl Safari Phú Quốc",
        "Có nhiều trải nghiệm như Safari bán hoang dã, Kid Zoo và khu bò sát.",
    )
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "thế còn khu bò sát thì sao?"}, "rewrite-thread")

    assert result["rewrite_source"] == "heuristic"
    assert "Tôi muốn hỏi về các khu tham quan ở Vinpearl Safari Phú Quốc" in result["search_query"]
    assert "thế còn khu bò sát thì sao?" in result["search_query"]
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
    assert result["query_retry_count"] == 1
    assert result["rewrite_source"] == "retry_expanded"


@pytest.mark.asyncio
async def test_low_confidence_retries_with_expanded_query_before_answering():
    settings = Settings(openai_api_key=None, low_confidence_threshold=0.5)
    kb = RetryThenAnswerKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Có khu bò sát không?"}, "retry-thread")

    assert kb.queries[0] == "Có khu bò sát không?"
    assert "Vinpearl Safari Phú Quốc" in kb.queries[1]
    assert "động vật loài thú" in kb.queries[1]
    assert result["query_retry_count"] == 1
    assert result["rewrite_source"] == "retry_expanded"
    assert result["handoff_required"] is False
    assert "khu bò sát" in result["answer"].lower()


@pytest.mark.asyncio
async def test_small_talk_routes_directly_without_kb_search():
    settings = Settings(openai_api_key=None)
    kb = FailIfSearchKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    result = await graph.ainvoke({"transcript": "Xin chào"}, "small-talk-thread")

    assert result["intent"] == "small_talk"
    assert result["primary_intent"] == "small_talk"
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
    assert result["primary_intent"] == "out_of_scope"
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


@pytest.mark.asyncio
async def test_streaming_pre_answer_contextualizes_before_kb_route():
    settings = Settings(openai_api_key=None)
    kb = RecordingKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    nodes = []
    async for node_name, _, _ in graph.astream_pre_answer(
        {"transcript": "Vinpearl Safari mở cửa mấy giờ?"},
        "stream-knowledge",
    ):
        nodes.append(node_name)

    assert nodes == ["supervisor", "contextualize_query", "safari_knowledge", "offer_freshness"]
    assert kb.last_query == "Vinpearl Safari mở cửa mấy giờ?"


@pytest.mark.asyncio
async def test_streaming_pre_answer_emits_retry_query_for_low_confidence():
    settings = Settings(openai_api_key=None, low_confidence_threshold=0.5)
    kb = RetryThenAnswerKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    nodes = []
    async for node_name, _, _ in graph.astream_pre_answer(
        {"transcript": "Có khu bò sát không?"},
        "stream-retry",
    ):
        nodes.append(node_name)

    assert nodes == [
        "supervisor",
        "contextualize_query",
        "safari_knowledge",
        "offer_freshness",
        "retry_query",
        "safari_knowledge",
        "offer_freshness",
    ]


@pytest.mark.asyncio
async def test_supervisor_classifies_domain_primary_intents():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    cases = [
        ("Có show động vật lúc nào?", "show_schedule"),
        ("Có hổ Bengal không?", "animal_info"),
        ("Mở cửa mấy giờ?", "opening_hours"),
    ]

    for question, primary_intent in cases:
        result = await graph.ainvoke({"transcript": question}, f"primary-{primary_intent}")
        assert result["primary_intent"] == primary_intent
        assert result["suggested_route"] == "safari_knowledge"


@pytest.mark.asyncio
async def test_streaming_pre_answer_searches_before_handoff_for_ticket_price():
    settings = Settings(openai_api_key=None)
    kb = RecordingKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    nodes = []
    final_state = {}
    async for node_name, _, current_state in graph.astream_pre_answer(
        {"transcript": "Giá vé hôm nay bao nhiêu?"},
        "stream-handoff",
    ):
        nodes.append(node_name)
        final_state = current_state

    assert nodes == [
        "supervisor",
        "contextualize_query",
        "safari_knowledge",
        "offer_freshness",
        "escalation",
    ]
    assert kb.last_query == "Giá vé hôm nay bao nhiêu?"
    assert final_state["handoff_required"] is True


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


class RetryThenAnswerKnowledgeBase(InMemoryKnowledgeBase):
    def __init__(self, embeddings: EmbeddingProvider) -> None:
        super().__init__(embeddings)
        self.queries = []

    async def search(self, query: str, limit: int = 5):
        self.queries.append(query)
        if len(self.queries) == 1:
            return [
                KnowledgeHit(
                    id="weak-score",
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
        return [
            KnowledgeHit(
                id="reptile-area",
                title="Vinpearl Safari Phú Quốc",
                section="Khu bò sát",
                category="animal",
                content="Vinpearl Safari có khu bò sát và nhiều loài động vật cho khách tham quan.",
                source_url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
                language="vi",
                crawled_at=None,
                valid_until=None,
                metadata={},
                score=0.9,
            )
        ]


class AffiliateKnowledgeBase(InMemoryKnowledgeBase):
    async def search(self, query: str, limit: int = 5):
        return [
            KnowledgeHit(
                id="affiliate",
                title="VinWonders chính thức ra mắt VinWonders Affiliate",
                section="VinWonders chính thức ra mắt VinWonders Affiliate",
                category="affiliate",
                content="\n".join(
                    [
                        "vi",
                        "[Đăng nhập](https://booking.vinwonders.com/login?redirectUri=https://vinwonders.com/demo)",
                        "[Đăng ký](https://booking.vinwonders.com/login?tab=register&redirectUri=https://vinwonders.com/demo)",
                        "- [Trang chủ](https://vinwonders.com/)",
                        "- Wonderpedia",
                        "- Bài viết",
                        "# VinWonders chính thức ra mắt VinWonders Affiliate",
                        "VinWonders Affiliate có cơ chế hoa hồng hấp dẫn, chính sách thưởng đa dạng và bộ tài nguyên truyền thông sẵn có.",
                        "Mục lục",
                        "- [1 . Giới thiệu chương trình VinWonders Affiliate](https://vinwonders.com/demo#intro)",
                    ]
                ),
                source_url="https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/",
                language="vi",
                crawled_at=None,
                valid_until=None,
                metadata={},
                score=0.9,
            )
        ]


class WonderpediaKnowledgeBase(InMemoryKnowledgeBase):
    async def search(self, query: str, limit: int = 5):
        return [
            KnowledgeHit(
                id="wonderpedia",
                title="Wonderpedia | Cẩm Nang Du Lịch, Điểm Đến & Lịch Trình",
                section="WONDERPEDIA",
                category="wonderpedia",
                content=(
                    "WONDERPEDIA\n"
                    "Nơi mở ra những vùng đất diệu kỳ, câu chuyện lý thú, khoảnh khắc tuyệt hơn mơ…\n"
                    "Cùng VinWonders khám phá ngay thôi!\n"
                    "WonderCulture WonderLand WonderMoment WonderCreature Tin công ty"
                ),
                source_url="https://vinwonders.com/vi/wonderpedia/",
                language="vi",
                crawled_at=None,
                valid_until=None,
                metadata={},
                score=0.95,
            ),
            KnowledgeHit(
                id="voucher",
                title="Voucher Vinpearl Safari Phú Quốc ưu đãi và bí quyết săn 2026",
                section="Voucher",
                category="booking",
                content="Thông tin về voucher/combo Vinpearl Safari Phú Quốc giá hời.",
                source_url="https://vinwonders.com/vi/wonderpedia/news/voucher-vinpearl-safari-phu-quoc/",
                language="vi",
                crawled_at=None,
                valid_until=None,
                metadata={},
                score=0.8,
            ),
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

    def with_structured_output(self, schema):
        del schema
        raise RuntimeError("structured output unavailable in this fake")


class FailIfCalledLLM:
    async def ainvoke(self, messages):
        raise AssertionError("LLM should not be called for rule fast-path questions")


def json_payload(value: dict) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
