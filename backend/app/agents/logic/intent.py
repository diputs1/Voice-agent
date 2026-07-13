from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.schemas import SupervisorDecision

TIME_SENSITIVE_KEYWORDS = {
    "giá",
    "vé",
    "bao nhiêu",
    "khuyến mãi",
    "ưu đãi",
    "voucher",
    "combo",
    "vinclub",
    "hôm nay",
    "ngày mai",
    "bao nhiêu tiền",
    "còn khuyến mãi",
    "còn ưu đãi",
    "cuối tuần",
    "dịp lễ",
    "nghỉ lễ",
    "mới nhất",
    "hiện tại",
    "áp dụng",
    "phí",
    "chi phí",
    "vào cổng",
}
TIME_SENSITIVE_CATEGORIES = {"offer", "price", "booking", "vinclub"}
LLM_CLASSIFIER_HINTS = {
    "ngân sách",
    "chi tiêu",
    "tốn",
    "nên mua",
    "nên đặt",
    "loại nào",
    "gói nào",
    "budget",
    "cost",
}
SMALL_TALK_KEYWORDS = {
    "xin chào",
    "chào",
    "hello",
    "hi",
    "cảm ơn",
    "cam on",
    "thanks",
    "thank you",
}
OUT_OF_SCOPE_KEYWORDS = {
    "thời tiết",
    "weather",
    "bóng đá",
    "football",
    "chứng khoán",
    "stock",
    "bitcoin",
    "khách sạn nha trang",
    "vinwonders nha trang",
    "grand world",
    "ocean city",
    "hà nội",
    "ha noi",
    "vũ yên",
    "vu yen",
}
DOMAIN_ANCHORS = {
    "vinpearl",
    "vinwonders",
    "safari",
    "phú quốc",
    "phu quoc",
}


def decision(
    primary_intent: str,
    is_time_sensitive: bool,
    entities: list[str],
    confidence: float,
    suggested_route: str,
) -> SupervisorDecision:
    return SupervisorDecision(
        primary_intent=primary_intent,
        is_time_sensitive=is_time_sensitive,
        entities=entities,
        confidence=max(0.0, min(1.0, confidence)),
        suggested_route=suggested_route,
    )


def supervisor_messages(transcript: str):
    return [
        SystemMessage(
            content=(
                "Phân loại intent cho câu hỏi du khách Vinpearl Safari Phú Quốc. "
                "Trả về đúng schema: primary_intent là một trong ticket_price, opening_hours, "
                "animal_info, show_schedule, time_sensitive, general, small_talk, out_of_scope; "
                "is_time_sensitive là true nếu câu hỏi phụ thuộc dữ liệu có thể đổi như giá vé, "
                "ưu đãi, voucher, lịch theo ngày, hôm nay, ngày mai, cuối tuần, dịp lễ; "
                "entities là các cụm thực thể quan trọng; confidence từ 0 đến 1; "
                "suggested_route là safari_knowledge, direct_response hoặc direct_handoff. "
                "Chọn direct_response cho chào hỏi/ngoài phạm vi. Chọn direct_handoff khi câu hỏi "
                "không thể xử lý bằng knowledge base. Với giá vé/ưu đãi/booking, chọn "
                "safari_knowledge để tìm dữ liệu trước; hệ thống sẽ tự handoff nếu dữ liệu thiếu "
                "hoặc hết hạn."
            )
        ),
        HumanMessage(content=transcript),
    ]


def coerce_supervisor_decision(value: Any) -> SupervisorDecision:
    if isinstance(value, SupervisorDecision):
        return _normalize_supervisor_decision(value)
    if isinstance(value, dict):
        return _normalize_supervisor_decision(SupervisorDecision.model_validate(value))
    return parse_supervisor_payload(str(value))


def parse_supervisor_payload(content: str) -> SupervisorDecision:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    payload = json.loads(stripped)
    if "intent" in payload and "primary_intent" not in payload:
        intent = str(payload["intent"])
        payload = {
            "primary_intent": "time_sensitive" if intent == "time_sensitive" else intent,
            "is_time_sensitive": intent == "time_sensitive",
            "entities": payload.get("entities") or [],
            "confidence": payload.get("confidence", 0.0),
            "suggested_route": "direct_response"
            if intent in {"small_talk", "out_of_scope"}
            else "safari_knowledge",
        }
    return _normalize_supervisor_decision(SupervisorDecision.model_validate(payload))


def _normalize_supervisor_decision(value: SupervisorDecision) -> SupervisorDecision:
    if value.is_time_sensitive and value.suggested_route == "direct_handoff":
        return value.model_copy(update={"suggested_route": "safari_knowledge"})
    return value


def route_from_decision(decision_value: SupervisorDecision) -> str:
    if decision_value.suggested_route == "direct_response":
        return "direct_response"
    if decision_value.suggested_route == "direct_handoff":
        return "direct_handoff"
    return "safari_knowledge"


def primary_intent_from_rules(lowered: str) -> str:
    if any(keyword in lowered for keyword in ("giá", "vé", "bao nhiêu tiền", "vào cổng")):
        return "ticket_price"
    if any(keyword in lowered for keyword in ("show", "biểu diễn", "lịch diễn")):
        return "show_schedule"
    if any(
        keyword in lowered
        for keyword in ("động vật", "con gì", "hổ", "sư tử", "voi", "bò sát", "bo sat", "animal")
    ):
        return "animal_info"
    if any(keyword in lowered for keyword in ("giờ", "mở cửa", "thời gian", "lúc nào")):
        return "opening_hours"
    if any(keyword in lowered for keyword in TIME_SENSITIVE_KEYWORDS):
        return "time_sensitive"
    return "general"


def entities_from_rules(lowered: str) -> list[str]:
    keywords = (
        "giá vé",
        "giờ mở cửa",
        "show",
        "biểu diễn",
        "động vật",
        "hổ",
        "sư tử",
        "night safari",
        "vinclub",
        "voucher",
        "khuyến mãi",
        "affiliate",
        "sản phẩm",
        "san pham",
        "product",
        "products",
        "tour",
        "gói",
        "dịch vụ",
    )
    return matched_keywords(lowered, keywords)


def matched_keywords(lowered: str, keywords) -> list[str]:
    return sorted({keyword for keyword in keywords if keyword in lowered})


def is_small_talk(lowered: str) -> bool:
    stripped = lowered.strip(" !?.")
    return any(stripped == keyword or stripped.startswith(f"{keyword} ") for keyword in SMALL_TALK_KEYWORDS)


def is_out_of_scope(lowered: str) -> bool:
    if any(anchor in lowered for anchor in DOMAIN_ANCHORS):
        return False
    return any(keyword in lowered for keyword in OUT_OF_SCOPE_KEYWORDS)
