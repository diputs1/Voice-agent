from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agents.checkpointers.memory import build_memory_checkpointer
from app.agents.graphs import WebsiteAgentGraph
from app.agents.logic.answer_grounding import handoff_state, is_grounded_quantitative_answer
from app.agents.logic.intent import parse_supervisor_payload
from app.agents.nodes.offer_freshness import route_after_freshness
from app.agents.schemas import SupervisorDecision
from app.agents.state import AgentState
from app.agents.tools.knowledge_tools import TOOL_WHITELIST, search_kb
from app.api.schemas import ChatRequest, CrawlRequest, VoiceAgentKnowledgeRequest
from app.core.config import Settings
from app.core.observability import configure_langsmith
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.kb import InMemoryKnowledgeBase, KnowledgeHit
from app.prompts import load_prompt
from app.routers.admin import require_admin_api_key


def test_prompt_and_instruction_layer_loads_versioned_voice_prompt():
    prompt = load_prompt("voice_answer")

    assert prompt.name == "voice_answer"
    assert prompt.version.startswith("voice_answer@")
    assert "Chỉ trả lời dựa trên context" in prompt.template


@pytest.mark.asyncio
async def test_context_management_keeps_only_recent_thread_history():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, settings.openai_embedding_model))
    graph = WebsiteAgentGraph(kb, settings)

    for index in range(6):
        await kb.save_thread_turn("history-thread", f"question-{index}", f"answer-{index}")

    state = await graph._with_thread_history({"transcript": "follow up"}, "history-thread")

    assert [turn["transcript"] for turn in state["thread_history"]] == [
        "question-2",
        "question-3",
        "question-4",
        "question-5",
    ]


def test_memory_layer_uses_langgraph_memory_checkpointer():
    checkpointer = build_memory_checkpointer()

    assert checkpointer.__class__.__name__ in {"MemorySaver", "InMemorySaver"}
    assert hasattr(checkpointer, "get_tuple")
    assert hasattr(checkpointer, "put")


def test_tool_layer_exposes_curated_whitelist_without_dynamic_dispatcher():
    assert TOOL_WHITELIST == {
        "rewrite_query",
        "search_kb",
        "verify_evidence",
        "finish_answer",
        "handoff",
    }


@pytest.mark.asyncio
async def test_tool_layer_search_kb_returns_retrieval_payload_shape():
    result = await search_kb(kb=FakeRetriever(), query="opening hours", site_id="vin", limit=3)

    assert result["confidence"] == 0.82
    assert result["retrieval_debug"] == {
        "site_id": "vin",
        "candidate_count": 1,
        "retriever": "hybrid_pgvector_fts_rrf",
    }
    assert result["retrieved_context"][0]["content"] == "Open 09:00-16:00."
    assert result["citations"][0]["source_url"] == "https://example.test/source"


def test_state_management_tracks_retrieval_voice_routing_and_counters():
    expected_keys = {
        "messages",
        "transcript",
        "thread_history",
        "search_query",
        "retrieved_context",
        "citations",
        "confidence",
        "retrieval_debug",
        "answer",
        "grounded",
        "evaluator_score",
        "handoff_required",
        "handoff_reason",
        "recommended_action",
        "intent",
        "primary_intent",
        "suggested_route",
        "route",
        "query_retry_count",
        "retry_reason",
        "agent_iteration_count",
        "tool_call_count",
    }

    assert expected_keys <= set(AgentState.__annotations__)


@pytest.mark.asyncio
async def test_orchestrator_loop_emits_expected_nodes_before_answer():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, settings.openai_embedding_model))
    await kb.ensure_ready()
    graph = WebsiteAgentGraph(kb, settings, legacy_node_names=True)

    nodes = []
    async for node_name, _, _ in graph.astream_pre_answer(
        {"transcript": "Vinpearl Safari mở cửa mấy giờ?"},
        "capability-loop",
    ):
        nodes.append(node_name)

    assert nodes == ["supervisor", "contextualize_query", "safari_knowledge", "offer_freshness"]


@pytest.mark.asyncio
async def test_retrieval_and_knowledge_layer_supports_site_scoped_search_and_hash():
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()

    default_hash = await kb.doc_set_hash()
    site_hash = await kb.doc_set_hash("missing-site")
    hits = await kb.search("Vinpearl Safari mở cửa mấy giờ?", site_id=None)

    assert default_hash
    assert site_hash
    assert default_hash != site_hash
    assert hits
    assert hits[0].source_url


def test_guardrails_safety_routes_low_confidence_to_retry_then_handoff():
    low_confidence_state = {
        "route": "safari_knowledge",
        "intent": "stable",
        "confidence": 0.1,
        "handoff_reason": "low_confidence",
        "handoff_required": True,
        "query_retry_count": 0,
    }
    exhausted_retry_state = {**low_confidence_state, "query_retry_count": 1}

    assert route_after_freshness(low_confidence_state, low_confidence_threshold=0.7) == "retry"
    assert route_after_freshness(exhausted_retry_state, low_confidence_threshold=0.7) == "escalate"
    assert handoff_state({}, "low_confidence")["recommended_action"] == "check_official_site"
    assert not is_grounded_quantitative_answer(
        "Công viên mở cửa từ 08:00.",
        [{"content": "Công viên mở cửa từ 09:00 đến 16:00."}],
    )


def test_validation_and_structured_output_contracts_are_enforced():
    decision = parse_supervisor_payload(
        '{"primary_intent":"opening_hours","is_time_sensitive":false,'
        '"entities":["giờ mở cửa"],"confidence":0.9,"suggested_route":"safari_knowledge"}'
    )

    assert isinstance(decision, SupervisorDecision)
    assert decision.primary_intent == "opening_hours"
    assert ChatRequest(transcript="Xin chào").transcript == "Xin chào"
    with pytest.raises(ValidationError):
        ChatRequest(transcript="")
    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.test/", unknown_field=True)


def test_voice_agent_tool_schema_accepts_elevenlabs_wrapped_parameters():
    payload = VoiceAgentKnowledgeRequest.model_validate(
        {
            "conversation_id": "conversation-1",
            "request_id": "request-1",
            "parameters": {"query": "giờ mở cửa", "site_id": " vin ", "limit": 3},
        }
    )

    assert payload.query == "giờ mở cửa"
    assert payload.site_id == "vin"
    assert payload.limit == 3
    assert payload.conversation_id == "conversation-1"
    assert payload.request_id == "request-1"


def test_evaluation_assets_have_required_case_contract():
    cases_path = Path(__file__).parents[1] / "evals" / "cases.json"
    eval_runner = Path(__file__).parents[1] / "evals" / "run_eval.py"

    assert cases_path.exists()
    assert eval_runner.exists()
    assert '"expected_intent"' in cases_path.read_text(encoding="utf-8")
    assert '"requires_citations"' in cases_path.read_text(encoding="utf-8")


def test_observability_can_enable_langsmith_from_settings(monkeypatch):
    for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    status = configure_langsmith(
        Settings(
            langsmith_tracing=True,
            langsmith_api_key="ls-test",
            langsmith_project="vin-agent-test",
        )
    )

    assert status["enabled"] is True
    assert status["api_key_configured"] is True
    assert status["project"] == "vin-agent-test"


@pytest.mark.asyncio
async def test_security_permission_admin_guard_fails_closed_outside_dev_without_key():
    request = FakeRequest(
        Settings(
            app_env="production",
            admin_api_key=None,
        )
    )

    with pytest.raises(Exception) as exc:
        await require_admin_api_key(request)

    assert getattr(exc.value, "status_code", None) == 500
    assert "ADMIN_API_KEY" in str(getattr(exc.value, "detail", ""))


class FakeRetriever:
    async def search(self, query: str, limit: int = 5, site_id: str | None = None):
        assert query == "opening hours"
        assert limit == 3
        assert site_id == "vin"
        return [
            KnowledgeHit(
                id="hit-1",
                title="Source",
                section="Opening Hours",
                category="schedule",
                content="Open 09:00-16:00.",
                source_url="https://example.test/source",
                language="en",
                crawled_at=None,
                valid_until=None,
                metadata={},
                score=0.82,
                site_id=site_id,
            )
        ]


class FakeRequest:
    def __init__(self, settings: Settings) -> None:
        self.app = FakeApp(settings)


class FakeApp:
    def __init__(self, settings: Settings) -> None:
        self.state = FakeState(settings)


class FakeState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
