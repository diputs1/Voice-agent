from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

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
from app.agents.schemas import SafariState, SupervisorDecision
from app.config import Settings
from app.kb import AgentKnowledgeBase


class SafariAgentGraph:
    def __init__(self, kb: AgentKnowledgeBase, settings: Settings) -> None:
        self.kb = kb
        self.settings = settings
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

    async def ainvoke(self, state: SafariState, thread_id: str) -> SafariState:
        state = await self._with_thread_history(state, thread_id)
        return await self.graph.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

    async def astream_updates(self, state: SafariState, thread_id: str):
        state = await self._with_thread_history(state, thread_id)
        async for event in self.graph.astream(
            state,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            yield event

    async def astream_pre_answer(
        self, state: SafariState, thread_id: str
    ) -> AsyncIterator[tuple[str, SafariState, SafariState]]:
        state = await self._with_thread_history(state, thread_id)
        current: SafariState = dict(state)
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

        while True:
            for node_name, node in (
                ("safari_knowledge", self._safari_knowledge_node),
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

    async def astream_voice_answer_tokens(self, state: SafariState) -> AsyncIterator[str]:
        async for token in stream_voice_answer_tokens(state, self.llm):
            yield token

    def finalize_streamed_answer(self, state: SafariState, answer: str) -> SafariState:
        return finalize_streamed_answer(state, answer)

    def _build_graph(self):
        graph = StateGraph(SafariState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("contextualize_query", self._contextualize_query_node)
        graph.add_node("direct_response", self._direct_response_node)
        graph.add_node("direct_handoff", self._direct_handoff_node)
        graph.add_node("safari_knowledge", self._safari_knowledge_node)
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
        graph.add_edge("contextualize_query", "safari_knowledge")
        graph.add_edge("safari_knowledge", "offer_freshness")
        graph.add_conditional_edges(
            "offer_freshness",
            self._route_after_freshness,
            {"retry": "retry_query", "escalate": "escalation", "answer": "voice_answer"},
        )
        graph.add_edge("retry_query", "safari_knowledge")
        graph.add_edge("escalation", "voice_answer")
        graph.add_edge("voice_answer", END)

        return graph.compile(checkpointer=MemorySaver())

    async def _supervisor_node(self, state: SafariState) -> SafariState:
        return await supervisor_node(state, self.llm)

    async def _contextualize_query_node(self, state: SafariState) -> SafariState:
        return await contextualize_query_node(state, self.llm)

    async def _safari_knowledge_node(self, state: SafariState) -> SafariState:
        return await safari_knowledge_node(state, self.kb)

    async def _offer_freshness_node(self, state: SafariState) -> SafariState:
        return await offer_freshness_node(state, self.settings.low_confidence_threshold)

    async def _retry_query_node(self, state: SafariState) -> SafariState:
        return await retry_query_node(state)

    async def _escalation_node(self, state: SafariState) -> SafariState:
        return await escalation_node(state)

    async def _voice_answer_node(self, state: SafariState) -> SafariState:
        return await voice_answer_node(state, self.llm)

    async def _direct_response_node(self, state: SafariState) -> SafariState:
        return await direct_response_node(state)

    async def _direct_handoff_node(self, state: SafariState) -> SafariState:
        return await direct_handoff_node(state)

    def _route_after_freshness(self, state: SafariState) -> Literal["retry", "escalate", "answer"]:
        return route_after_freshness(state, self.settings.low_confidence_threshold)

    def _route_after_supervisor(self, state: SafariState) -> Literal["direct", "handoff", "knowledge"]:
        if state.get("route") == "direct_response":
            return "direct"
        if state.get("route") == "direct_handoff":
            return "handoff"
        return "knowledge"

    async def _with_thread_history(self, state: SafariState, thread_id: str) -> SafariState:
        if state.get("thread_history") is not None:
            return state
        history = await self.kb.get_thread(thread_id)
        return {**state, "thread_history": history[-4:]}


__all__ = ["SafariAgentGraph", "SafariState", "SupervisorDecision"]
