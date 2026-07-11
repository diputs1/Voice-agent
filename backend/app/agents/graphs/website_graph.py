from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Callable, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.agents.checkpointers import build_memory_checkpointer
from app.agents.edges.route_rules import route_after_supervisor
from app.agents.nodes.contextualize import contextualize_query_node
from app.agents.nodes.direct_response import direct_handoff_node, direct_response_node
from app.agents.nodes.escalation import escalation_node
from app.agents.nodes.offer_freshness import offer_freshness_node, route_after_freshness
from app.agents.nodes.retry_query import retry_query_node
from app.agents.nodes.safari_knowledge import safari_knowledge_node
from app.agents.nodes.supervisor import supervisor_node
from app.agents.nodes.voice_answer import (
    finalize_streamed_answer,
    stream_voice_answer_tokens,
    voice_answer_node,
)
from app.agents.schemas import SupervisorDecision
from app.agents.state import AgentState, AgentUpdate
from app.core.config import Settings
from app.knowledge.kb import AgentKnowledgeBase


class WebsiteAgentGraph:
    graph_version = "website-agent-v1"

    def __init__(
        self,
        kb: AgentKnowledgeBase,
        settings: Settings,
        *,
        legacy_node_names: bool = False,
        checkpointer_factory: Callable[[], Any] = build_memory_checkpointer,
    ) -> None:
        self.kb = kb
        self.settings = settings
        self.legacy_node_names = legacy_node_names
        self.checkpointer_factory = checkpointer_factory
        self.llm = (
            ChatOpenAI(
                model=settings.openai_chat_model,
                api_key=settings.openai_api_key,
                temperature=0.2,
            )
            if settings.openai_api_key
            else None
        )
        self.graph = self._build_graph()

    async def ainvoke(self, state: AgentState, thread_id: str) -> AgentState:
        state = await self._with_thread_history(state, thread_id)
        return await self.graph.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

    async def astream_updates(self, state: AgentState, thread_id: str):
        state = await self._with_thread_history(state, thread_id)
        async for event in self.graph.astream(
            state,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            yield event

    async def astream_pre_answer(
        self, state: AgentState, thread_id: str
    ) -> AsyncIterator[tuple[str, AgentUpdate, AgentState]]:
        state = await self._with_thread_history(state, thread_id)
        current: AgentState = dict(state)
        update = await self._supervisor_node(current)
        current.update(update)
        yield "supervisor", update, current

        route = self._route_after_supervisor(current)
        if route == "direct":
            update = await self._direct_response_node(current)
            current.update(update)
            yield "direct_response", update, current
            return
        if route == "handoff":
            update = await self._direct_handoff_node(current)
            current.update(update)
            yield "direct_handoff", update, current
            return

        update = await self._contextualize_query_node(current)
        current.update(update)
        yield "contextualize_query", update, current

        knowledge_event = "safari_knowledge" if self.legacy_node_names else "knowledge"
        while True:
            for node_name, node in (
                (knowledge_event, self._knowledge_node),
                ("offer_freshness", self._offer_freshness_node),
            ):
                update = await node(current)
                current.update(update)
                yield node_name, update, current

            if self._route_after_freshness(current) != "retry":
                break

            update = await self._retry_query_node(current)
            current.update(update)
            yield "retry_query", update, current

        if self._route_after_freshness(current) == "escalate":
            update = await self._escalation_node(current)
            current.update(update)
            yield "escalation", update, current

    async def astream_voice_answer_tokens(self, state: AgentState) -> AsyncIterator[str]:
        async for token in stream_voice_answer_tokens(state, self.llm):
            yield token

    def finalize_streamed_answer(self, state: AgentState, answer: str) -> AgentState:
        return finalize_streamed_answer(state, answer)

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("contextualize_query", self._contextualize_query_node)
        graph.add_node("direct_response", self._direct_response_node)
        graph.add_node("direct_handoff", self._direct_handoff_node)
        graph.add_node("knowledge", self._knowledge_node)
        graph.add_node("retry_query", self._retry_query_node)
        graph.add_node("offer_freshness", self._offer_freshness_node)
        graph.add_node("escalation", self._escalation_node)
        graph.add_node("voice_answer", self._voice_answer_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "direct": "direct_response",
                "handoff": "direct_handoff",
                "knowledge": "contextualize_query",
            },
        )
        graph.add_edge("direct_response", END)
        graph.add_edge("direct_handoff", END)
        graph.add_edge("contextualize_query", "knowledge")
        graph.add_edge("knowledge", "offer_freshness")
        graph.add_conditional_edges(
            "offer_freshness",
            self._route_after_freshness,
            {"retry": "retry_query", "escalate": "escalation", "answer": "voice_answer"},
        )
        graph.add_edge("retry_query", "knowledge")
        graph.add_edge("escalation", "voice_answer")
        graph.add_edge("voice_answer", END)

        return graph.compile(checkpointer=self.checkpointer_factory())

    async def _supervisor_node(self, state: AgentState) -> AgentUpdate:
        return await supervisor_node(state, self.llm)

    async def _contextualize_query_node(self, state: AgentState) -> AgentUpdate:
        return await contextualize_query_node(state, self.llm)

    async def _knowledge_node(self, state: AgentState) -> AgentUpdate:
        update = await safari_knowledge_node(state, self.kb)
        update["retrieval_debug"] = {
            "site_id": state.get("site_id"),
            "retriever": "hybrid_vector_lexical_rrf",
            "candidate_count": len(update.get("retrieved_context", [])),
        }
        return update

    async def _offer_freshness_node(self, state: AgentState) -> AgentUpdate:
        return await offer_freshness_node(state, self.settings.low_confidence_threshold)

    async def _retry_query_node(self, state: AgentState) -> AgentUpdate:
        return await retry_query_node(state)

    async def _escalation_node(self, state: AgentState) -> AgentUpdate:
        return await escalation_node(state)

    async def _voice_answer_node(self, state: AgentState) -> AgentUpdate:
        return await voice_answer_node(state, self.llm)

    async def _direct_response_node(self, state: AgentState) -> AgentUpdate:
        return await direct_response_node(state)

    async def _direct_handoff_node(self, state: AgentState) -> AgentUpdate:
        return await direct_handoff_node(state)

    def _route_after_freshness(self, state: AgentState) -> Literal["retry", "escalate", "answer"]:
        return route_after_freshness(state, self.settings.low_confidence_threshold)

    def _route_after_supervisor(self, state: AgentState) -> Literal["direct", "handoff", "knowledge"]:
        return route_after_supervisor(state)

    async def _with_thread_history(self, state: AgentState, thread_id: str) -> AgentState:
        if state.get("thread_history") is not None:
            return state
        history = await self.kb.get_thread(thread_id)
        return {**state, "thread_history": history[-4:]}


__all__ = ["WebsiteAgentGraph", "AgentState", "SupervisorDecision"]
