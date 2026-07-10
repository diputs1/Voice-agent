from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from app.agents.logic.answer_grounding import (
    fallback_answer,
    handoff_state,
    is_grounded_quantitative_answer,
    message_content_to_text,
    supporting_citations,
    word_chunks,
)
from app.agents.schemas import SafariState
from app.prompts import load_prompt


@traceable(name="graph.voice_answer")
async def voice_answer_node(state: SafariState, llm) -> SafariState:
    if state.get("answer") and state.get("handoff_required"):
        return state

    if llm:
        response = await llm.ainvoke(voice_answer_messages(state))
        answer = str(response.content)
    else:
        transcript = state.get("transcript", "")
        answer = fallback_answer(transcript, state.get("retrieved_context", []))

    return finalize_streamed_answer(state, answer)


async def stream_voice_answer_tokens(state: SafariState, llm) -> AsyncIterator[str]:
    if state.get("answer") and state.get("route") == "direct_response":
        yield state["answer"]
        return
    if state.get("answer") and state.get("route") == "direct_handoff":
        yield state["answer"]
        return
    if state.get("answer") and state.get("handoff_required"):
        yield state["answer"]
        return

    if llm:
        async for chunk in llm.astream(voice_answer_messages(state)):
            text = message_content_to_text(chunk.content)
            if text:
                yield text
        return

    answer = fallback_answer(state.get("transcript", ""), state.get("retrieved_context", []))
    for token in word_chunks(answer):
        yield token


def finalize_streamed_answer(state: SafariState, answer: str) -> SafariState:
    if state.get("answer") and state.get("handoff_required"):
        return state
    if not is_grounded_quantitative_answer(answer, state.get("retrieved_context", [])):
        return {**handoff_state(state, "ungrounded_answer"), "grounded": False, "evaluator_score": 0.0}
    return {
        "answer": answer,
        "citations": supporting_citations(answer, state.get("retrieved_context", [])),
        "handoff_required": False,
        "handoff_reason": None,
        "recommended_action": None,
        "grounded": True,
        "evaluator_score": 1.0,
    }


def voice_answer_messages(state: SafariState):
    context = "\n\n".join(
        f"[{idx + 1}] {item['section']} - {item['content']}"
        for idx, item in enumerate(state.get("retrieved_context", []))
    )
    transcript = state.get("transcript", "")
    return [
        SystemMessage(content=load_prompt("voice_answer").template),
        HumanMessage(
            content=(
                f"Câu hỏi: {transcript}\n\n"
                f"Context:\n{context or 'Không tìm thấy context phù hợp.'}"
            )
        ),
    ]
