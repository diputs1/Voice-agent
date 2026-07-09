from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from bs4 import BeautifulSoup

SOURCE_URL = "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/"


@dataclass
class IngestedChunk:
    title: str
    section: str
    category: str
    content: str
    source_url: str
    content_hash: str
    crawled_at: datetime
    valid_until: datetime | None
    language: str = "vi"
    metadata: dict[str, Any] | None = None


STABLE_SEED_FACTS = [
    (
        "Tổng quan",
        "Giới thiệu",
        "overview",
        "Vinpearl Safari Phú Quốc là công viên chăm sóc và bảo tồn động vật bán hoang dã tại Phú Quốc, phù hợp cho gia đình, trẻ em và khách muốn tham quan thế giới động vật.",
    ),
    (
        "Giờ hoạt động",
        "Giờ mở cửa",
        "schedule",
        "Vinpearl Safari Phú Quốc mở cửa hằng ngày từ 09:00 đến 16:00. Khách nên kiểm tra lại lịch vận hành trước ngày đi nếu đi vào dịp lễ hoặc giai đoạn bảo trì.",
    ),
    (
        "Trải nghiệm",
        "Hoạt động nổi bật",
        "experience",
        "Các trải nghiệm nổi bật gồm tham quan Safari bán hoang dã, xem biểu diễn động vật, Kid Zoo, khu bò sát, vườn chim và các hoạt động tương tác phù hợp cho trẻ em.",
    ),
    (
        "Night Safari",
        "Hoạt động buổi tối",
        "experience",
        "Night Safari là trải nghiệm tham quan vào buổi tối tại Vinpearl Safari. Lịch và tình trạng mở bán có thể thay đổi, nên cần xác nhận lại khi đặt vé.",
    ),
    (
        "Ẩm thực và mua sắm",
        "Dịch vụ trong công viên",
        "service",
        "Trong khuôn viên có các điểm ăn uống, mua sắm quà lưu niệm và dịch vụ hỗ trợ khách tham quan.",
    ),
    (
        "Liên hệ",
        "Hotline",
        "contact",
        "Khi cần xác nhận giá vé, ưu đãi, lịch show hoặc hỗ trợ đặt dịch vụ, khách nên liên hệ hotline/booking chính thức được hiển thị trên website VinWonders.",
    ),
]


async def fetch_vinwonders_chunks(url: str = SOURCE_URL) -> list[IngestedChunk]:
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = _clean(soup.title.get_text(" ")) if soup.title else "Vinpearl Safari Phú Quốc"
    blocks: list[tuple[str, str]] = []
    current_section = "Trang VinWonders"
    for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        text = _clean(element.get_text(" "))
        if len(text) < 20:
            continue
        if element.name in {"h1", "h2", "h3"}:
            current_section = text[:140]
            continue
        blocks.append((current_section, text))

    if not blocks:
        return seed_chunks(url)

    grouped: dict[str, list[str]] = {}
    for section, text in blocks:
        grouped.setdefault(section, []).append(text)

    chunks: list[IngestedChunk] = []
    crawled_at = datetime.now(UTC)
    for section, texts in grouped.items():
        for content in _chunk_text(" ".join(texts)):
            category = classify_category(content)
            chunks.append(
                IngestedChunk(
                    title=title[:200],
                    section=section[:200],
                    category=category,
                    content=content,
                    source_url=url,
                    content_hash=_hash(content),
                    crawled_at=crawled_at,
                    valid_until=_valid_until(category, crawled_at),
                    metadata={
                        "seed_url": url,
                        "canonical_url": url,
                        "source_type": "vinwonders_page",
                        "render_mode": "http",
                        "crawl_depth": 0,
                        "confidence": 0.7,
                    },
                )
            )
    return chunks or seed_chunks(url)


def seed_chunks(url: str = SOURCE_URL) -> list[IngestedChunk]:
    crawled_at = datetime.now(UTC)
    return [
        IngestedChunk(
            title="Vinpearl Safari Phú Quốc",
            section=section,
            category=category,
            content=content,
            source_url=url,
            content_hash=_hash(content),
            crawled_at=crawled_at,
            valid_until=_valid_until(category, crawled_at),
            metadata={
                "seed_url": url,
                "canonical_url": url,
                "source_type": "seed_fact",
                "render_mode": "seed",
                "crawl_depth": 0,
                "confidence": 0.55,
            },
        )
        for title, section, category, content in STABLE_SEED_FACTS
    ]


def classify_category(text: str) -> str:
    lowered = text.lower()
    if any(
        keyword in lowered
        for keyword in [
            "vinwonders affiliate",
            "chương trình affiliate",
            "hoa hồng",
            "đối tác tiếp thị",
            "nhà sáng tạo nội dung",
        ]
    ):
        return "affiliate"
    if any(keyword in lowered for keyword in ["vinclub", "hội viên", "hạng gold", "platinum", "diamond"]):
        return "vinclub"
    if any(keyword in lowered for keyword in ["booking.vinwonders", "đặt vé online", "qr", "thanh toán"]):
        return "booking"
    if any(keyword in lowered for keyword in ["nội quy", "quy định", "điều kiện", "điều khoản"]):
        return "policy"
    if any(keyword in lowered for keyword in ["giá vé", "bảng giá", "vnđ", "vnd"]):
        return "price"
    if any(keyword in lowered for keyword in ["giờ", "mở cửa", "lịch", "09:00", "16:00"]):
        return "schedule"
    if any(keyword in lowered for keyword in ["giá", "vé", "ưu đãi", "voucher", "khuyến mãi"]):
        return "offer"
    if any(keyword in lowered for keyword in ["nhà hàng", "ẩm thực", "mua sắm", "quà"]):
        return "service"
    if any(keyword in lowered for keyword in ["hotline", "liên hệ", "booking", "đặt"]):
        return "contact"
    if any(keyword in lowered for keyword in ["show", "safari", "kid zoo", "night", "biểu diễn"]):
        return "experience"
    return "overview"


def _valid_until(category: str, crawled_at: datetime) -> datetime | None:
    if category in {"offer", "price", "booking", "vinclub"}:
        return crawled_at + timedelta(days=7)
    if category == "schedule":
        return crawled_at + timedelta(days=14)
    return None


def content_hash_for(canonical_url: str, content: str) -> str:
    return _hash(f"{canonical_url}\n{content}")


def extract_structured_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    prices = sorted(set(re.findall(r"\b\d{1,3}(?:[.,]\d{3})+(?:\s?VNĐ|\s?vnđ|\s?VND)?", text)))
    discounts = sorted(set(re.findall(r"(?:giảm|ưu đãi|tới|lên tới)\s*\d{1,3}%|x\d+\s*ưu đãi", text, re.I)))
    date_windows = sorted(
        set(
            re.findall(
                r"(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?\s*[-–]\s*\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d{1,2}/\d{1,2}/\d{4})",
                text,
            )
        )
    )
    if prices:
        metadata["price"] = prices
    if discounts:
        metadata["discount"] = discounts
    if date_windows:
        metadata["applicability"] = date_windows
    return metadata


def soup_to_chunks(
    soup: BeautifulSoup,
    *,
    source_url: str,
    canonical_url: str,
    seed_url: str,
    discovered_from: str | None,
    crawl_depth: int,
    render_mode: str,
    status_code: int,
    crawl_job_id: str | None,
) -> list[IngestedChunk]:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = _clean(soup.title.get_text(" ")) if soup.title else "VinWonders"
    meta_description = ""
    description_tag = soup.find("meta", attrs={"name": "description"})
    if description_tag and description_tag.get("content"):
        meta_description = _clean(str(description_tag["content"]))

    blocks: list[tuple[str, str]] = []
    current_section = title[:140]
    if meta_description:
        blocks.append(("Meta description", meta_description))

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        text = _clean(script.get_text(" "))
        if text:
            try:
                payload = json.loads(text)
                blocks.append(("Structured data", _clean(json.dumps(payload, ensure_ascii=False))))
            except json.JSONDecodeError:
                blocks.append(("Structured data", text))

    for element in soup.find_all(["h1", "h2", "h3", "p", "li", "td", "th", "figcaption"]):
        text = _clean(element.get_text(" "))
        if len(text) < 12:
            continue
        if element.name in {"h1", "h2", "h3"}:
            current_section = text[:180]
            if _is_related_or_navigation_section(current_section):
                current_section = "__skip_related__"
                continue
            blocks.append((current_section, text))
            continue
        if current_section == "__skip_related__":
            continue
        blocks.append((current_section, text))

    grouped: dict[str, list[str]] = {}
    for section, text in blocks:
        grouped.setdefault(section, []).append(text)

    chunks: list[IngestedChunk] = []
    crawled_at = datetime.now(UTC)
    for section, texts in grouped.items():
        for content in _chunk_text(" ".join(texts), size=1100, overlap=160):
            category = classify_category(f"{title} {section} {content}")
            extracted = extract_structured_metadata(content)
            metadata = {
                "crawl_job_id": crawl_job_id,
                "seed_url": seed_url,
                "canonical_url": canonical_url,
                "discovered_from": discovered_from,
                "crawl_depth": crawl_depth,
                "source_type": "web_page",
                "render_mode": render_mode,
                "status_code": status_code,
                "valid_from": crawled_at.isoformat(),
                "location": _infer_location(f"{title} {content}"),
                "applicability": extracted.get("applicability"),
                "price": extracted.get("price"),
                "discount": extracted.get("discount"),
                "confidence": 0.82 if status_code == 200 else 0.45,
            }
            chunks.append(
                IngestedChunk(
                    title=title[:200],
                    section=section[:200],
                    category=category,
                    content=content,
                    source_url=source_url,
                    content_hash=content_hash_for(canonical_url, content),
                    crawled_at=crawled_at,
                    valid_until=_valid_until(category, crawled_at),
                    language=_infer_language(canonical_url),
                    metadata={key: value for key, value in metadata.items() if value is not None},
                )
            )
    return chunks


def _chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    text = _clean(text)
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        split_at = text.rfind(". ", start, end)
        if split_at > start + size // 2:
            end = split_at + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if len(chunk) > 80]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_related_or_navigation_section(section: str) -> bool:
    lowered = section.lower()
    return any(
        keyword in lowered
        for keyword in [
            "ưu đãi nổi bật",
            "bài viết liên quan",
            "có thể bạn quan tâm",
            "tin liên quan",
            "địa điểm khác",
            "khám phá thêm",
        ]
    )


def _infer_location(text: str) -> str | None:
    lowered = text.lower()
    if "phú quốc" in lowered or "phu quoc" in lowered:
        return "Phú Quốc"
    if "nha trang" in lowered:
        return "Nha Trang"
    if "vũ yên" in lowered or "vu yen" in lowered:
        return "Vũ Yên"
    return None


def _infer_language(url: str) -> str:
    lowered = url.lower()
    if "/en/" in lowered:
        return "en"
    return "vi"
