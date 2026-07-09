from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from app.agents.logic.query_rewrite import (
    heuristic_rewrite,
    history_context,
    looks_like_follow_up,
    parse_rewrite_payload,
)
from app.agents.schemas import SafariState


@traceable(name="graph.contextualize_query")
async def contextualize_query_node(state: SafariState, llm) -> SafariState:
    search_query, rewrite_source = await resolve_search_query(
        state.get("transcript", ""),
        state.get("thread_history", []),
        llm,
    )
    return {
        "search_query": search_query,
        "rewrite_source": rewrite_source,
        "query_retry_count": int(state.get("query_retry_count", 0)),
        "retry_reason": None,
    }


@traceable(name="graph.resolve_search_query")
async def resolve_search_query(
    transcript: str,
    history: list[dict[str, Any]],
    llm,
) -> tuple[str, str]:
    if not history:
        return transcript, "original"
    if not looks_like_follow_up(transcript):
        return transcript, "original"

    if llm:
        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Viết lại câu hỏi follow-up của người dùng thành một câu hỏi "
                            "độc lập để truy xuất knowledge base Vinpearl Safari Phú Quốc. "
                            "Giữ nguyên ý định, ngôn ngữ tiếng Việt và các thực thể quan trọng. "
                            "Chỉ trả JSON hợp lệ: "
                            '{"search_query":"câu hỏi độc lập"}'
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Lịch sử gần nhất:\n{history_context(history)}\n\n"
                            f"Câu hỏi mới: {transcript}"
                        )
                    ),
                ]
            )
            rewritten = parse_rewrite_payload(str(response.content))
            if rewritten:
                return rewritten, "llm"
        except Exception:
            pass

    fallback = heuristic_rewrite(transcript, history)
    if fallback != transcript:
        return fallback, "heuristic"
    return transcript, "original"
