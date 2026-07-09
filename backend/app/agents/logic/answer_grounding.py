from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.agents.logic.intent import TIME_SENSITIVE_CATEGORIES
from app.agents.schemas import SafariState
from app.kb import KnowledgeHit
from app.prompts import load_prompt

HANDOFF_ACTIONS = {
    "stale_or_missing_time_sensitive_data": "contact_hotline_or_booking",
    "low_confidence": "check_official_site",
    "ungrounded_answer": "contact_hotline_or_booking",
}


def message_content_to_text(content: Any) -> str:
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


def word_chunks(text: str) -> list[str]:
    words = text.split(" ")
    return [word + (" " if idx < len(words) - 1 else "") for idx, word in enumerate(words)]


def handoff_state(state: SafariState, reason: str) -> SafariState:
    partial_answer = partial_answer_from_context(state.get("retrieved_context", []))
    answer = load_prompt("escalation").format(partial_answer=partial_answer)
    return {
        "answer": answer,
        "handoff_required": True,
        "handoff_reason": reason,
        "recommended_action": HANDOFF_ACTIONS.get(reason, "check_official_site"),
    }


def partial_answer_from_context(context: list[dict[str, Any]]) -> str:
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


def supporting_citations(
    answer: str,
    context: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    claims = {normalize_claim(claim) for claim in quantitative_claims(answer)}
    answer_tokens = meaningful_tokens(answer)
    price_or_offer_answer = any(
        keyword in answer.lower()
        for keyword in ("giá", "vé", "vnđ", "vnd", "ưu đãi", "khuyến mãi", "booking", "đặt")
    )

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, item in enumerate(context):
        item_claims = {
            normalize_claim(claim) for claim in quantitative_claims(str(item.get("content", "")))
        }
        claim_overlap = len(claims & item_claims)
        item_tokens = meaningful_tokens(
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


def meaningful_tokens(text: str) -> set[str]:
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


def fallback_answer(transcript: str, context: list[dict[str, Any]]) -> str:
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


def is_grounded_quantitative_answer(answer: str, context: list[dict[str, Any]]) -> bool:
    claims = quantitative_claims(answer)
    if not claims:
        return True
    context_text = "\n".join(str(item.get("content", "")) for item in context)
    context_claims = quantitative_claims(context_text)
    normalized_context = {normalize_claim(claim) for claim in context_claims}
    return all(normalize_claim(claim) in normalized_context for claim in claims)


def quantitative_claims(text: str) -> set[str]:
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


def normalize_claim(value: str) -> str:
    lowered = value.lower().replace("vnđ", "vnd").replace("đồng", "vnd")
    return re.sub(r"\s+", "", lowered)


def citation(hit: KnowledgeHit) -> dict[str, Any]:
    return {
        "source_url": hit.source_url,
        "title": hit.title,
        "section": citation_label(hit),
        "category": hit.category,
        "language": hit.language,
        "crawled_at": hit.crawled_at,
        "valid_until": hit.valid_until,
        "metadata": hit.metadata or {},
    }


def hit_payload(hit: KnowledgeHit) -> dict[str, Any]:
    return {
        **citation(hit),
        "id": hit.id,
        "content": context_content(hit),
        "score": hit.score,
    }


def context_content(hit: KnowledgeHit) -> str:
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


def citation_label(hit: KnowledgeHit) -> str:
    section = (hit.section or "").strip()
    if is_clean_label(section):
        return clean_label_text(section)
    title = (hit.title or "").strip()
    if title:
        return clean_label_text(title)
    return hit.source_url


def clean_label_text(value: str) -> str:
    return " - ".join(part.strip() for part in value.split("|") if part.strip())


def is_clean_label(value: str) -> bool:
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


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
