from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.kb import KnowledgeBase, KnowledgeHit
from app.prompts import load_prompt


class SafariState(TypedDict, total=False):
    messages: list[dict[str, str]]
    transcript: str
    intent: str
    retrieved_context: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    confidence: float
    classifier_confidence: float
    classifier_source: str
    handoff_required: bool
    handoff_reason: str | None
    recommended_action: str | None
    route: str


TIME_SENSITIVE_KEYWORDS = {
    "giá",
    "vé",
    "bao nhiêu",
    "khuyến mãi",
    "ưu đãi",
    "voucher",
    "combo",
    "vinclub",
    "affiliate",
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
HANDOFF_ACTIONS = {
    "stale_or_missing_time_sensitive_data": "contact_hotline_or_booking",
    "low_confidence": "check_official_site",
    "ungrounded_answer": "contact_hotline_or_booking",
}


class SafariAgentGraph:
    def __init__(self, kb: KnowledgeBase, settings: Settings) -> None:
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
        return await self.graph.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

    async def astream_updates(self, state: SafariState, thread_id: str):
        async for event in self.graph.astream(
            state,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            yield event

    async def astream_pre_answer(
        self, state: SafariState, thread_id: str
    ) -> AsyncIterator[tuple[str, SafariState, SafariState]]:
        del thread_id
        current: SafariState = dict(state)
        for node_name, node in (
            ("supervisor", self._supervisor_node),
            ("safari_knowledge", self._safari_knowledge_node),
            ("offer_freshness", self._offer_freshness_node),
        ):
            update = await node(current)
            current.update(update)
            yield node_name, update, current

        if self._route_after_freshness(current) == "escalate":
            update = await self._escalation_node(current)
            current.update(update)
            yield "escalation", update, current

    async def astream_voice_answer_tokens(self, state: SafariState) -> AsyncIterator[str]:
        if state.get("answer") and state.get("handoff_required"):
            yield state["answer"]
            return

        if self.llm:
            async for chunk in self.llm.astream(self._voice_answer_messages(state)):
                text = _message_content_to_text(chunk.content)
                if text:
                    yield text
            return

        answer = _fallback_answer(state.get("transcript", ""), state.get("retrieved_context", []))
        for token in _word_chunks(answer):
            yield token

    def finalize_streamed_answer(self, state: SafariState, answer: str) -> SafariState:
        if state.get("answer") and state.get("handoff_required"):
            return state
        if not _is_grounded_quantitative_answer(answer, state.get("retrieved_context", [])):
            return _handoff_state(state, "ungrounded_answer")
        return {
            "answer": answer,
            "citations": _supporting_citations(answer, state.get("retrieved_context", [])),
            "handoff_required": False,
            "handoff_reason": None,
            "recommended_action": None,
        }

    def _build_graph(self):
        graph = StateGraph(SafariState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("safari_knowledge", self._safari_knowledge_node)
        graph.add_node("offer_freshness", self._offer_freshness_node)
        graph.add_node("escalation", self._escalation_node)
        graph.add_node("voice_answer", self._voice_answer_node)

        graph.add_edge(START, "supervisor")
        graph.add_edge("supervisor", "safari_knowledge")
        graph.add_edge("safari_knowledge", "offer_freshness")
        graph.add_conditional_edges(
            "offer_freshness",
            self._route_after_freshness,
            {"escalate": "escalation", "answer": "voice_answer"},
        )
        graph.add_edge("escalation", "voice_answer")
        graph.add_edge("voice_answer", END)

        return graph.compile(checkpointer=MemorySaver())

    async def _supervisor_node(self, state: SafariState) -> SafariState:
        transcript = state.get("transcript", "")
        intent, confidence, source = await self._classify_intent(transcript)
        return {
            "intent": intent,
            "classifier_confidence": confidence,
            "classifier_source": source,
            "messages": [{"role": "user", "content": transcript}],
            "route": "safari_knowledge",
        }

    async def _classify_intent(self, transcript: str) -> tuple[str, float, str]:
        lowered = transcript.lower()
        if any(k in lowered for k in TIME_SENSITIVE_KEYWORDS):
            return "time_sensitive", 1.0, "rules"
        if not any(k in lowered for k in LLM_CLASSIFIER_HINTS):
            return "stable", 1.0, "rules"
        if not self.llm:
            return "stable", 0.0, "rules_fallback"

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Phân loại câu hỏi cho Vinpearl Safari Phú Quốc. "
                            "Trả về JSON hợp lệ duy nhất: "
                            '{"intent":"time_sensitive|stable","confidence":0.0}. '
                            "time_sensitive nếu câu hỏi phụ thuộc dữ liệu có thể đổi như giá, vé, "
                            "ưu đãi, voucher, combo, booking, lịch theo ngày, hôm nay, ngày mai, "
                            "cuối tuần, dịp lễ, còn áp dụng hay mới nhất."
                        )
                    ),
                    HumanMessage(content=transcript),
                ]
            )
            payload = _parse_classifier_payload(str(response.content))
            return payload["intent"], payload["confidence"], "llm"
        except Exception:
            return "stable", 0.0, "llm_failed"

    async def _safari_knowledge_node(self, state: SafariState) -> SafariState:
        hits = await self.kb.search(state.get("transcript", ""), limit=5)
        citations = [_citation(hit) for hit in hits]
        confidence = max((hit.score for hit in hits), default=0.0)
        return {
            "retrieved_context": [_hit_payload(hit) for hit in hits],
            "citations": citations,
            "confidence": float(confidence),
        }

    async def _offer_freshness_node(self, state: SafariState) -> SafariState:
        now = datetime.now(UTC)
        handoff_required = False
        handoff_reason: str | None = None
        if state.get("intent") == "time_sensitive":
            handoff_required = True
            handoff_reason = "stale_or_missing_time_sensitive_data"
            for item in state.get("retrieved_context", []):
                if item.get("category") not in TIME_SENSITIVE_CATEGORIES:
                    continue
                valid_until = _parse_datetime(item.get("valid_until"))
                if valid_until and valid_until > now:
                    handoff_required = False
                    handoff_reason = None
                    break
        if state.get("confidence", 0.0) < 0.05:
            handoff_required = True
            handoff_reason = "low_confidence"
        return {
            "handoff_required": handoff_required,
            "handoff_reason": handoff_reason,
            "recommended_action": HANDOFF_ACTIONS.get(handoff_reason),
        }

    async def _escalation_node(self, state: SafariState) -> SafariState:
        return _handoff_state(state, state.get("handoff_reason") or "stale_or_missing_time_sensitive_data")

    async def _voice_answer_node(self, state: SafariState) -> SafariState:
        if state.get("answer") and state.get("handoff_required"):
            return state

        if self.llm:
            response = await self.llm.ainvoke(self._voice_answer_messages(state))
            answer = str(response.content)
        else:
            transcript = state.get("transcript", "")
            answer = _fallback_answer(transcript, state.get("retrieved_context", []))

        return self.finalize_streamed_answer(state, answer)

    def _voice_answer_messages(self, state: SafariState):
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

    def _route_after_freshness(self, state: SafariState) -> Literal["escalate", "answer"]:
        return "escalate" if state.get("handoff_required") else "answer"


def _parse_classifier_payload(content: str) -> dict[str, float | str]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    payload = json.loads(stripped)
    intent = payload.get("intent")
    if intent not in {"time_sensitive", "stable"}:
        raise ValueError(f"Unsupported classifier intent: {intent}")
    confidence = float(payload.get("confidence", 0.0))
    return {"intent": intent, "confidence": max(0.0, min(1.0, confidence))}


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content or "")


def _word_chunks(text: str) -> list[str]:
    words = text.split(" ")
    return [word + (" " if idx < len(words) - 1 else "") for idx, word in enumerate(words)]


def _handoff_state(state: SafariState, reason: str) -> SafariState:
    partial_answer = _partial_answer_from_context(state.get("retrieved_context", []))
    answer = load_prompt("escalation").format(partial_answer=partial_answer)
    return {
        "answer": answer,
        "handoff_required": True,
        "handoff_reason": reason,
        "recommended_action": HANDOFF_ACTIONS.get(reason, "check_official_site"),
    }


def _partial_answer_from_context(context: list[dict[str, Any]]) -> str:
    stable_items = [
        item
        for item in context
        if item.get("category") not in TIME_SENSITIVE_CATEGORIES
        and item.get("category") != "contact"
        and item.get("content")
    ]
    if stable_items:
        top = stable_items[0]
        return str(top["content"]).strip()
    if context:
        return (
            "Mình có tìm thấy thông tin liên quan, nhưng phần này có thể phụ thuộc dữ liệu "
            "mới như giá vé, ưu đãi, lịch áp dụng hoặc booking."
        )
    return "Mình chưa tìm thấy thông tin đủ phù hợp trong knowledge base."


def _supporting_citations(answer: str, context: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    claims = {_normalize_claim(claim) for claim in _quantitative_claims(answer)}
    answer_tokens = _meaningful_tokens(answer)
    price_or_offer_answer = any(
        keyword in answer.lower()
        for keyword in ("giá", "vé", "vnđ", "vnd", "ưu đãi", "khuyến mãi", "booking", "đặt")
    )

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, item in enumerate(context):
        item_claims = {_normalize_claim(claim) for claim in _quantitative_claims(str(item.get("content", "")))}
        claim_overlap = len(claims & item_claims)
        item_tokens = _meaningful_tokens(
            " ".join(str(item.get(key, "")) for key in ("title", "section", "content", "source_url"))
        )
        token_overlap = len(answer_tokens & item_tokens)

        if claims and claim_overlap == 0:
            continue
        if (
            item.get("category") in TIME_SENSITIVE_CATEGORIES
            and not price_or_offer_answer
            and claim_overlap == 0
        ):
            continue
        if not claims and token_overlap < 2:
            continue

        score = claim_overlap * 2.0 + token_overlap * 0.1 + max(0.0, float(item.get("score") or 0.0)) * 0.05
        scored.append((score, idx, item))

    if not scored:
        scored = [(float(item.get("score") or 0.0), idx, item) for idx, item in enumerate(context[:limit])]

    scored.sort(key=lambda value: (-value[0], value[1]))
    citations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for _, _, item in scored:
        source_url = str(item.get("source_url") or "")
        if source_url in seen_urls:
            continue
        seen_urls.add(source_url)
        citations.append(
            {
                "source_url": item.get("source_url"),
                "title": item.get("title"),
                "section": item.get("section"),
                "category": item.get("category"),
                "language": item.get("language"),
                "crawled_at": item.get("crawled_at"),
                "valid_until": item.get("valid_until"),
                "metadata": item.get("metadata") or {},
            }
        )
        if len(citations) >= limit:
            break
    return citations


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "vinpearl",
        "safari",
        "phú",
        "quốc",
        "phu",
        "quoc",
        "https",
        "www",
        "com",
        "cho",
        "các",
        "vào",
        "của",
        "được",
        "hàng",
        "ngày",
        "where",
        "with",
        "from",
        "and",
        "the",
    }
    return {
        token
        for token in re.findall(r"[\wÀ-ỹ]+", text.lower())
        if len(token) >= 3 and token not in stopwords
    }


def _fallback_answer(transcript: str, context: list[dict[str, Any]]) -> str:
    if not context:
        return (
            "Mình chưa tìm thấy thông tin phù hợp trong knowledge base. "
            "Bạn có thể hỏi lại cụ thể hơn hoặc kiểm tra website VinWonders chính thức."
        )
    lowered = transcript.lower()
    if any(keyword in lowered for keyword in ["vinclub", "affiliate", "ưu đãi", "voucher", "giá vé"]):
        selected = []
        seen_sections = set()
        for item in context:
            key = (item["category"], item["section"])
            if item["category"] in {"vinclub", "affiliate", "offer", "price", "booking"} and key not in seen_sections:
                selected.append(item)
                seen_sections.add(key)
        if selected:
            sentences = []
            for item in selected[:3]:
                prefix = {
                    "vinclub": "VinClub",
                    "affiliate": "Affiliate",
                    "offer": "Ưu đãi",
                    "price": "Giá vé",
                    "booking": "Đặt vé",
                }.get(item["category"], item["section"])
                sentences.append(f"{prefix}: {item['content'][:360].strip()}")
            return (
                "Mình tìm thấy các thông tin liên quan trong KB mới crawl. "
                + " ".join(sentences)
                + " Với ưu đãi, giá vé hoặc hạn áp dụng, bạn nên kiểm tra lại nguồn chính thức trước khi mua."
            )
    top = context[0]
    if top["category"] == "schedule":
        return "Vinpearl Safari Phú Quốc thường mở cửa hằng ngày từ 09:00 đến 16:00. Bạn nên kiểm tra lại lịch vận hành trước ngày đi."
    if top["category"] == "contact":
        return "Với các thông tin cần xác nhận như đặt vé, ưu đãi hoặc lịch show, bạn nên liên hệ hotline/booking chính thức trên website VinWonders."
    return top["content"]


def _is_grounded_quantitative_answer(answer: str, context: list[dict[str, Any]]) -> bool:
    claims = _quantitative_claims(answer)
    if not claims:
        return True
    context_text = "\n".join(str(item.get("content", "")) for item in context)
    context_claims = _quantitative_claims(context_text)
    normalized_context = {_normalize_claim(claim) for claim in context_claims}
    return all(_normalize_claim(claim) in normalized_context for claim in claims)


def _quantitative_claims(text: str) -> set[str]:
    patterns = [
        r"\b\d{1,2}:\d{2}\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        r"\b\d{1,3}(?:[.,]\d{3})+(?:\s?(?:VNĐ|VND|đ|đồng))?\b",
        r"\b\d{1,3}\s?%\b",
    ]
    claims: set[str] = set()
    for pattern in patterns:
        claims.update(re.findall(pattern, text, flags=re.IGNORECASE))
    return claims


def _normalize_claim(value: str) -> str:
    lowered = value.lower().replace("vnđ", "vnd").replace("đồng", "vnd")
    return re.sub(r"\s+", "", lowered)


def _citation(hit: KnowledgeHit) -> dict[str, Any]:
    return {
        "source_url": hit.source_url,
        "title": hit.title,
        "section": _citation_label(hit),
        "category": hit.category,
        "language": hit.language,
        "crawled_at": hit.crawled_at,
        "valid_until": hit.valid_until,
        "metadata": hit.metadata or {},
    }


def _hit_payload(hit: KnowledgeHit) -> dict[str, Any]:
    return {
        **_citation(hit),
        "id": hit.id,
        "content": _context_content(hit),
        "score": hit.score,
    }


def _context_content(hit: KnowledgeHit) -> str:
    content = hit.content
    metadata = hit.metadata or {}
    firecrawl_metadata = metadata.get("firecrawl_metadata") or {}
    descriptions = [
        str(firecrawl_metadata.get("description") or "").strip(),
        str(firecrawl_metadata.get("ogDescription") or "").strip(),
    ]
    for description in descriptions:
        if description and description not in content:
            return f"Mô tả trang: {description}\n\n{content}"
    return content


def _citation_label(hit: KnowledgeHit) -> str:
    section = (hit.section or "").strip()
    if _is_clean_label(section):
        return _clean_label_text(section)
    title = (hit.title or "").strip()
    if title:
        return _clean_label_text(title)
    return hit.source_url


def _clean_label_text(value: str) -> str:
    return " - ".join(part.strip() for part in value.split("|") if part.strip())


def _is_clean_label(value: str) -> bool:
    if len(value) < 12:
        return False
    if value[0].islower():
        return False
    if "=" in value or "&" in value:
        return False
    lowered = value.lower()
    bad_fragments = (
        "http://",
        "https://",
        "utm_",
        "redirecturi",
        "đăng nhập",
        "đăng ký",
        "log in",
        "register",
        "copy to clipboard",
    )
    return not any(fragment in lowered for fragment in bad_fragments)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
