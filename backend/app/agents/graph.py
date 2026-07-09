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
from langsmith import traceable

from app.config import Settings
from app.kb import KnowledgeBase, KnowledgeHit
from app.prompts import load_prompt


class SafariState(TypedDict, total=False):
    messages: list[dict[str, str]]
    transcript: str
    thread_history: list[dict[str, Any]]
    search_query: str
    rewrite_source: str
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
HANDOFF_ACTIONS = {
    "stale_or_missing_time_sensitive_data": "contact_hotline_or_booking",
    "low_confidence": "check_official_site",
    "ungrounded_answer": "contact_hotline_or_booking",
}
FOLLOW_UP_MARKERS = {
    "thế còn",
    "vậy còn",
    "còn ",
    "thì sao",
    "nó",
    "đó",
    "cái này",
    "cái đó",
    "ở đó",
    "chỗ đó",
    "như vậy",
    "trẻ em",
    "người lớn",
    "người già",
    "vé đó",
    "show đó",
    "dịch vụ đó",
}
DOMAIN_ANCHORS = {
    "vinpearl",
    "vinwonders",
    "safari",
    "phú quốc",
    "phu quoc",
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

        if self._route_after_supervisor(current) == "direct":
            update = await self._direct_response_node(current)
            current.update(update)
            yield "direct_response", update, current
            return

        for node_name, node in (
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
        if state.get("answer") and state.get("route") == "direct_response":
            yield state["answer"]
            return
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
        graph.add_node("direct_response", self._direct_response_node)
        graph.add_node("safari_knowledge", self._safari_knowledge_node)
        graph.add_node("offer_freshness", self._offer_freshness_node)
        graph.add_node("escalation", self._escalation_node)
        graph.add_node("voice_answer", self._voice_answer_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {"direct": "direct_response", "knowledge": "safari_knowledge"},
        )
        graph.add_edge("direct_response", END)
        graph.add_edge("safari_knowledge", "offer_freshness")
        graph.add_conditional_edges(
            "offer_freshness",
            self._route_after_freshness,
            {"escalate": "escalation", "answer": "voice_answer"},
        )
        graph.add_edge("escalation", "voice_answer")
        graph.add_edge("voice_answer", END)

        return graph.compile(checkpointer=MemorySaver())

    @traceable(name="graph.supervisor")
    async def _supervisor_node(self, state: SafariState) -> SafariState:
        transcript = state.get("transcript", "")
        intent, confidence, source = await self._classify_intent(transcript)
        search_query, rewrite_source = await self._resolve_search_query(
            transcript,
            state.get("thread_history", []),
        )
        return {
            "intent": intent,
            "classifier_confidence": confidence,
            "classifier_source": source,
            "search_query": search_query,
            "rewrite_source": rewrite_source,
            "messages": [{"role": "user", "content": transcript}],
            "route": "direct_response" if intent in {"small_talk", "out_of_scope"} else "safari_knowledge",
        }

    @traceable(name="graph.classify_intent")
    async def _classify_intent(self, transcript: str) -> tuple[str, float, str]:
        lowered = transcript.lower()
        if _is_small_talk(lowered):
            return "small_talk", 1.0, "rules"
        if _is_out_of_scope(lowered):
            return "out_of_scope", 1.0, "rules"
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
                            '{"intent":"time_sensitive|stable|small_talk|out_of_scope","confidence":0.0}. '
                            "time_sensitive nếu câu hỏi phụ thuộc dữ liệu có thể đổi như giá, vé, "
                            "ưu đãi, voucher, combo, booking, lịch theo ngày, hôm nay, ngày mai, "
                            "cuối tuần, dịp lễ, còn áp dụng hay mới nhất. "
                            "small_talk nếu chỉ chào hỏi/cảm ơn. out_of_scope nếu không liên quan "
                            "Vinpearl Safari Phú Quốc."
                        )
                    ),
                    HumanMessage(content=transcript),
                ]
            )
            payload = _parse_classifier_payload(str(response.content))
            return payload["intent"], payload["confidence"], "llm"
        except Exception:
            return "stable", 0.0, "llm_failed"

    @traceable(name="graph.safari_knowledge")
    async def _safari_knowledge_node(self, state: SafariState) -> SafariState:
        hits = await self.kb.search(state.get("search_query") or state.get("transcript", ""), limit=5)
        citations = [_citation(hit) for hit in hits]
        confidence = max((hit.score for hit in hits), default=0.0)
        return {
            "retrieved_context": [_hit_payload(hit) for hit in hits],
            "citations": citations,
            "confidence": float(confidence),
        }

    @traceable(name="graph.offer_freshness")
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
        if state.get("confidence", 0.0) < self.settings.low_confidence_threshold:
            handoff_required = True
            handoff_reason = "low_confidence"
        return {
            "handoff_required": handoff_required,
            "handoff_reason": handoff_reason,
            "recommended_action": HANDOFF_ACTIONS.get(handoff_reason),
        }

    @traceable(name="graph.escalation")
    async def _escalation_node(self, state: SafariState) -> SafariState:
        return _handoff_state(state, state.get("handoff_reason") or "stale_or_missing_time_sensitive_data")

    @traceable(name="graph.voice_answer")
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

    @traceable(name="graph.direct_response")
    async def _direct_response_node(self, state: SafariState) -> SafariState:
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

    def _route_after_supervisor(self, state: SafariState) -> Literal["direct", "knowledge"]:
        return "direct" if state.get("intent") in {"small_talk", "out_of_scope"} else "knowledge"

    async def _with_thread_history(self, state: SafariState, thread_id: str) -> SafariState:
        if state.get("thread_history") is not None:
            return state
        history = await self.kb.get_thread(thread_id)
        return {**state, "thread_history": history[-4:]}

    @traceable(name="graph.resolve_search_query")
    async def _resolve_search_query(
        self,
        transcript: str,
        history: list[dict[str, Any]],
    ) -> tuple[str, str]:
        if not history:
            return transcript, "original"
        if not _looks_like_follow_up(transcript):
            return transcript, "original"

        if self.llm:
            try:
                response = await self.llm.ainvoke(
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
                                f"Lịch sử gần nhất:\n{_history_context(history)}\n\n"
                                f"Câu hỏi mới: {transcript}"
                            )
                        ),
                    ]
                )
                rewritten = _parse_rewrite_payload(str(response.content))
                if rewritten:
                    return rewritten, "llm"
            except Exception:
                pass

        fallback = _heuristic_rewrite(transcript, history)
        if fallback != transcript:
            return fallback, "heuristic"
        return transcript, "original"


def _parse_classifier_payload(content: str) -> dict[str, float | str]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    payload = json.loads(stripped)
    intent = payload.get("intent")
    if intent not in {"time_sensitive", "stable", "small_talk", "out_of_scope"}:
        raise ValueError(f"Unsupported classifier intent: {intent}")
    confidence = float(payload.get("confidence", 0.0))
    return {"intent": intent, "confidence": max(0.0, min(1.0, confidence))}


def _is_small_talk(lowered: str) -> bool:
    stripped = lowered.strip(" !?.")
    return any(stripped == keyword or stripped.startswith(f"{keyword} ") for keyword in SMALL_TALK_KEYWORDS)


def _is_out_of_scope(lowered: str) -> bool:
    if any(anchor in lowered for anchor in DOMAIN_ANCHORS):
        return False
    return any(keyword in lowered for keyword in OUT_OF_SCOPE_KEYWORDS)


def _parse_rewrite_payload(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        payload = json.loads(stripped)
        return str(payload.get("search_query") or "").strip()
    except json.JSONDecodeError:
        return stripped


def _looks_like_follow_up(transcript: str) -> bool:
    lowered = transcript.lower().strip()
    if any(marker in lowered for marker in FOLLOW_UP_MARKERS):
        return True
    if "?" in lowered and len(lowered.split()) <= 8 and not any(
        anchor in lowered for anchor in DOMAIN_ANCHORS
    ):
        return True
    return False


def _history_context(history: list[dict[str, Any]]) -> str:
    turns = history[-3:]
    lines = []
    for idx, turn in enumerate(turns, start=1):
        transcript = str(turn.get("transcript") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if transcript:
            lines.append(f"Lượt {idx} - người dùng: {transcript}")
        if answer:
            lines.append(f"Lượt {idx} - trợ lý: {answer[:500]}")
    return "\n".join(lines) or "Không có lịch sử."


def _heuristic_rewrite(transcript: str, history: list[dict[str, Any]]) -> str:
    previous = _latest_transcript(history)
    if not previous:
        return transcript
    if any(anchor in transcript.lower() for anchor in DOMAIN_ANCHORS):
        return transcript
    return f"{previous}. Câu hỏi tiếp theo: {transcript}"


def _latest_transcript(history: list[dict[str, Any]]) -> str:
    for turn in reversed(history):
        transcript = str(turn.get("transcript") or "").strip()
        if transcript:
            return transcript
    return ""


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
