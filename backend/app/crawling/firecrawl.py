from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from app.crawling.ingestion import (
    IngestedChunk,
    classify_category,
    content_hash_for,
    extract_structured_metadata,
    _infer_language,
    _valid_until,
)
from app.crawling.link_discovery import normalize_url
from app.core.sites import site_id_for_url


@dataclass
class FirecrawlPage:
    markdown: str
    metadata: dict[str, Any]
    links: list[str]


class FirecrawlClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.firecrawl.dev",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    async def start_crawl(
        self,
        *,
        url: str,
        max_depth: int,
        limit: int,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        allow_subdomains: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": url,
            "maxDiscoveryDepth": max_depth,
            "limit": limit,
            "crawlEntireDomain": True,
            "sitemap": "include",
            "ignoreQueryParameters": True,
            "allowExternalLinks": False,
            "allowSubdomains": allow_subdomains,
            "ignoreRobotsTxt": False,
            "scrapeOptions": {
                "formats": ["markdown", "links"],
                "onlyMainContent": True,
                "removeBase64Images": True,
                "blockAds": True,
                "proxy": "auto",
                "timeout": 60000,
            },
        }
        if include_paths:
            payload["includePaths"] = include_paths
        if exclude_paths:
            payload["excludePaths"] = exclude_paths
        return await self._request("POST", "/v2/crawl", json=payload)

    async def scrape(self, url: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v2/scrape",
            json={
                "url": url,
                "formats": ["markdown", "links"],
                "onlyMainContent": True,
                "removeBase64Images": True,
                "blockAds": True,
                "proxy": "auto",
                "timeout": 60000,
            },
        )

    async def map(
        self,
        *,
        url: str,
        search: str | None = None,
        limit: int = 5000,
        include_subdomains: bool = False,
        ignore_cache: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": url,
            "sitemap": "include",
            "includeSubdomains": include_subdomains,
            "ignoreQueryParameters": True,
            "ignoreCache": ignore_cache,
            "limit": limit,
            "timeout": 60000,
        }
        if search:
            payload["search"] = search
        return await self._request("POST", "/v2/map", json=payload)

    async def get_crawl_status(self, crawl_id_or_url: str) -> dict[str, Any]:
        if crawl_id_or_url.startswith("http"):
            return await self._request_url("GET", crawl_id_or_url)
        return await self._request("GET", f"/v2/crawl/{crawl_id_or_url}")

    async def wait_for_crawl(
        self,
        crawl_id: str,
        *,
        poll_interval: float = 2.0,
        timeout_seconds: float = 180.0,
    ) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        latest: dict[str, Any] = {}
        while asyncio.get_running_loop().time() < deadline:
            latest = await self.get_crawl_status(crawl_id)
            status = latest.get("status")
            if status in {"completed", "failed", "cancelled"}:
                if status != "completed":
                    raise RuntimeError(f"Firecrawl crawl ended with status={status}")
                return await self._collect_paginated_result(latest)
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Firecrawl crawl timed out after {timeout_seconds}s")

    async def _collect_paginated_result(self, first_page: dict[str, Any]) -> dict[str, Any]:
        merged = dict(first_page)
        data = list(first_page.get("data") or [])
        next_url = first_page.get("next")
        while next_url:
            next_page = await self.get_crawl_status(str(next_url))
            data.extend(next_page.get("data") or [])
            next_url = next_page.get("next")
        merged["data"] = data
        merged["next"] = None
        return merged

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        return await self._request_url(method, f"{self.base_url}{path}", **kwargs)

    async def _request_url(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        headers = kwargs.pop("headers", {})
        headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        async with httpx.AsyncClient(timeout=90.0, transport=self.transport) as client:
            response = await client.request(method, url, headers=headers, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"Firecrawl {method} {url} failed: {response.status_code} {response.text}")
        return response.json()


def firecrawl_result_to_chunks(
    *,
    result: dict[str, Any],
    seed_url: str,
    crawl_job_id: str | None,
    site_id: str | None = None,
    url_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[IngestedChunk]:
    pages = [_page_from_item(item) for item in result.get("data") or []]
    chunks: list[IngestedChunk] = []
    crawled_at = datetime.now(UTC)
    resolved_site_id = site_id or site_id_for_url(seed_url)
    for page in pages:
        source_url = str(page.metadata.get("sourceURL") or page.metadata.get("url") or seed_url)
        canonical_url = str(page.metadata.get("url") or source_url)
        extra_metadata = _url_metadata(url_metadata or {}, source_url, canonical_url)
        title = str(page.metadata.get("title") or "Firecrawl document")
        for content in _chunk_markdown(page.markdown):
            category = classify_category(f"{title}\n{content}\n{source_url}")
            extracted = extract_structured_metadata(content)
            metadata = {
                "crawl_job_id": crawl_job_id,
                "site_id": resolved_site_id,
                "seed_url": seed_url,
                "canonical_url": canonical_url,
                "source_type": "firecrawl",
                "render_mode": "firecrawl",
                "status_code": page.metadata.get("statusCode"),
                "firecrawl_metadata": page.metadata,
                "price": extracted.get("price"),
                "discount": extracted.get("discount"),
                "applicability": extracted.get("applicability"),
                "valid_from": crawled_at.isoformat(),
                "confidence": 0.9,
            }
            metadata.update(extra_metadata)
            chunks.append(
                IngestedChunk(
                    title=title[:200],
                    section=_section_from_markdown(content, title)[:200],
                    category=category,
                    content=content,
                    source_url=source_url,
                    content_hash=content_hash_for(canonical_url, content),
                    crawled_at=crawled_at,
                    valid_until=_valid_until(category, crawled_at),
                    language=_infer_language(canonical_url),
                    metadata={key: value for key, value in metadata.items() if value is not None},
                    site_id=resolved_site_id,
                )
            )
    return chunks


def scrape_result_to_chunks(
    *,
    result: dict[str, Any],
    seed_url: str,
    crawl_job_id: str | None,
    site_id: str | None = None,
    url_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[IngestedChunk]:
    data = result.get("data") or result
    return firecrawl_result_to_chunks(
        result={"data": [data]},
        seed_url=seed_url,
        crawl_job_id=crawl_job_id,
        site_id=site_id,
        url_metadata=url_metadata,
    )


def _page_from_item(item: dict[str, Any]) -> FirecrawlPage:
    return FirecrawlPage(
        markdown=str(item.get("markdown") or ""),
        metadata=dict(item.get("metadata") or {}),
        links=list(item.get("links") or []),
    )


def _url_metadata(metadata_by_url: dict[str, dict[str, Any]], *urls: str) -> dict[str, Any]:
    for url in urls:
        if url in metadata_by_url:
            return metadata_by_url[url]
        normalized = normalize_url(url)
        if normalized and normalized in metadata_by_url:
            return metadata_by_url[normalized]
    return {}


def _chunk_markdown(markdown: str, size: int = 1200, overlap: int = 160) -> list[str]:
    text = "\n".join(_trim_to_primary_content(_clean_markdown_lines(markdown)))
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        split_at = text.rfind("\n#", start, end)
        if split_at <= start + size // 3:
            split_at = text.rfind("\n", start, end)
        if split_at > start + size // 3:
            end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = _next_chunk_start(text, max(0, end - overlap))
    return chunks


def _next_chunk_start(text: str, preferred_start: int) -> int:
    if preferred_start <= 0:
        return 0
    boundary = text.find("\n", preferred_start)
    if boundary != -1 and boundary <= preferred_start + 120:
        return boundary + 1
    boundary = text.find(". ", preferred_start)
    if boundary != -1 and boundary <= preferred_start + 160:
        return boundary + 2
    boundary = text.find(" ", preferred_start)
    if boundary != -1 and boundary <= preferred_start + 60:
        return boundary + 1
    return preferred_start


def _section_from_markdown(markdown: str, fallback_title: str = "Firecrawl content") -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.strip("# ").strip()
            if _is_good_section_label(heading):
                return heading
    return fallback_title or "Firecrawl content"


def _trim_to_primary_content(lines: list[str]) -> list[str]:
    for preferred in ("# WONDERPEDIA", "# Wonderpedia"):
        try:
            index = lines.index(preferred)
        except ValueError:
            continue
        return lines[index:]

    for index, line in enumerate(lines):
        if not re.match(r"^#\s+.+", line):
            continue
        heading = line.strip("# ").strip()
        if _is_good_section_label(heading) and not _is_booking_widget_heading(heading):
            return lines[index:]
    return lines


def _is_booking_widget_heading(value: str) -> bool:
    lowered = value.lower().strip()
    return lowered in {"đặt vé", "đặt vé vinwonders", "booking", "book tickets"}


def _is_good_section_label(value: str) -> bool:
    if not value or len(value) < 8:
        return False
    lowered = value.lower()
    bad_fragments = (
        "đăng nhập",
        "đăng ký",
        "log in",
        "register",
        "utm_",
        "http://",
        "https://",
        "copy to clipboard",
    )
    return not any(fragment in lowered for fragment in bad_fragments)


def _clean_markdown_lines(markdown: str) -> list[str]:
    lines: list[str] = []
    skip_rest = False
    for raw_line in markdown.splitlines():
        if skip_rest:
            break
        line = raw_line.strip()
        if not line:
            continue
        line = _strip_markdown_images(line).strip()
        if not line:
            continue
        lowered = line.lower()
        if _is_footer_or_related_heading(lowered):
            skip_rest = True
            continue
        if line.startswith("![") or lowered.startswith("![]("):
            continue
        if "data:image/svg+xml" in lowered:
            line = re.sub(r"\(?data:image/svg\+xml[^)\]\s]*\)?", "", line, flags=re.IGNORECASE).strip()
            if not line:
                continue
        if "static.vinwonders.com" in lowered and any(
            ext in lowered for ext in [".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif"]
        ):
            continue
        if "production_style/style/images" in lowered:
            continue
        if _is_language_switcher_line(line):
            continue
        if _is_navigation_or_toc_line(line):
            continue
        if lowered in {
            "turn your device in landscape mode.",
            "menu",
            "close",
            "copy to clipboard",
            "previous",
            "next",
        }:
            continue
        lines.append(line)
    return lines


def _strip_markdown_images(line: str) -> str:
    line = line.replace("\\", " ")

    def replace_linked_image(match: re.Match[str]) -> str:
        alt_text = match.group(1) or ""
        suffix_text = match.group(2) or ""
        return (alt_text + suffix_text).strip()

    line = re.sub(
        r"\[!\[([^\n]*?)\]\([^)]+\)([^\]]*)\]\([^)]+\)",
        replace_linked_image,
        line,
    )
    line = re.sub(r"!\[[^\n]*?\]\([^)]+\)", "", line)
    line = re.sub(r"\s*!\[\s*", " ", line)
    line = re.sub(r"\[\s*\]\([^)]+\)", "", line)
    line = re.sub(r"\[_?[^]\s]*\.(?:jpg|jpeg|png|webp|svg)\)\s*", "[", line, flags=re.IGNORECASE)
    line = re.sub(r"\s*Copy to clipboard\s+\S+", "", line, flags=re.IGNORECASE)
    return line


def _is_footer_or_related_heading(lowered_line: str) -> bool:
    cleaned = lowered_line.lstrip("#- ").strip()
    stop_prefixes = (
        "bài viết liên quan",
        "bài viết dành cho bạn",
        "tin tức & trải nghiệm",
        "ưu đãi nổi bật",
        "sản phẩm được săn đón",
        "lịch sử tìm kiếm",
        "phổ biến nhất",
        "tìm kiếm",
        "related posts",
        "recommended articles",
        "news & experiences",
        "featured offers",
        "search history",
        "most popular",
        "special offers",
    )
    return cleaned.startswith(stop_prefixes)


def _is_language_switcher_line(line: str) -> bool:
    return bool(
        re.fullmatch(
            r"-?\s*\[(Tiếng Việt|English)\]\(https://vinwonders\.com/[^)]+\)",
            line,
        )
    )


def _is_navigation_or_toc_line(line: str) -> bool:
    lowered = line.lower().strip()
    if lowered in {"vi", "en", "mục lục", "table of contents"}:
        return True
    noise_fragments = (
        "booking.vinwonders.com/login",
        "redirecturi=",
        "[đăng nhập]",
        "[đăng ký]",
        "[trang chủ]",
        "- wonderpedia",
        "- bài viết",
    )
    if any(fragment in lowered for fragment in noise_fragments):
        return True
    return bool(re.match(r"^\s*-?\s*\[\s*\d+\s*[.)]?\s*.+#[-\wÀ-ỹ%]+", line, flags=re.IGNORECASE))
