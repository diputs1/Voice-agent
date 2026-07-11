from __future__ import annotations

from langsmith import traceable

from app.agents.logic.answer_grounding import handoff_state
from app.agents.schemas import SafariState
from app.agents.state import AgentUpdate


@traceable(name="graph.direct_response")
async def direct_response_node(state: SafariState) -> AgentUpdate:
    intent = state.get("intent")
    if intent == "small_talk":
        answer = (
            "Xin chào! Mình có thể hỗ trợ các câu hỏi về Vinpearl Safari Phú Quốc "
            "như giờ mở cửa, trải nghiệm, show, dịch vụ hoặc thông tin cần xác nhận."
        )
    else:
        answer = (
            "Mình chỉ hỗ trợ thông tin liên quan đến Vinpearl Safari Phú Quốc. "
            "Bạn có thể hỏi về lịch hoạt động, trải nghiệm, dịch vụ trong công viên "
            "hoặc kiểm tra nguồn VinWonders chính thức cho thông tin ngoài phạm vi này."
        )
    return {
        "answer": answer,
        "citations": [],
        "confidence": 1.0,
        "handoff_required": False,
        "handoff_reason": None,
        "recommended_action": None,
    }


@traceable(name="graph.direct_handoff")
async def direct_handoff_node(state: SafariState) -> AgentUpdate:
    return {
        **handoff_state(state, "stale_or_missing_time_sensitive_data"),
        "route": "direct_handoff",
    }
