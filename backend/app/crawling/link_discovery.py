from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import urlparse, urlunparse


MIN_SELECTED_URLS = 4


@dataclass(frozen=True)
class DiscoveredUrl:
    url: str
    category: str
    discovery_source: str
    discovered_from: str | None = None


@dataclass
class LinkDiscoveryResult:
    candidate_urls: list[str] = field(default_factory=list)
    selected_urls: list[DiscoveredUrl] = field(default_factory=list)
    skipped_urls: list[str] = field(default_factory=list)
    skip_reasons: dict[str, str] = field(default_factory=dict)
    scraped_urls: list[str] = field(default_factory=list)
    scrape_errors: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ScrapedDiscoveryPage:
    discovered_url: DiscoveredUrl
    result: dict[str, Any]


class FirecrawlScraper(Protocol):
    async def scrape(self, url: str) -> dict[str, Any]: ...


DiscoveryProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


class LinkDiscoveryAgent:
    def __init__(
        self,
        *,
        seed_url: str,
        allowed_domains: list[str] | None = None,
        include_subdomains: bool = False,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        self.seed_url = normalize_url(seed_url) or seed_url
        parsed = urlparse(self.seed_url)
        self._legacy_default_filter = allowed_domains is None and parsed.netloc.lower().endswith("vinwonders.com")
        self.allowed_domains = [domain.lower() for domain in (allowed_domains or [parsed.netloc])]
        self.include_subdomains = include_subdomains
        self.exclude_patterns = exclude_patterns or []

    def discover_from_firecrawl_result(
        self,
        result: dict[str, Any],
        *,
        extra_urls: list[str] | None = None,
    ) -> LinkDiscoveryResult:
        discovery = LinkDiscoveryResult()
        selected: dict[str, DiscoveredUrl] = {}
        candidates = self._candidates_from_result(result)
        candidates.update(self._seed_entrypoint_candidates())
        candidates.update(self._candidates_from_extra_urls(extra_urls or []))
        discovery.candidate_urls = list(candidates)

        for url, discovered_from in candidates.items():
            self._consider_url(
                discovery=discovery,
                selected=selected,
                url=url,
                discovered_from=discovered_from,
                discovery_source="firecrawl_links",
            )

        discovery.selected_urls = sorted_selected_urls(selected)
        return discovery

    async def discover_and_scrape(
        self,
        initial_result: dict[str, Any],
        scraper: FirecrawlScraper,
        *,
        max_scrapes: int,
        extra_urls: list[str] | None = None,
        on_progress: DiscoveryProgressCallback | None = None,
    ) -> tuple[LinkDiscoveryResult, list[ScrapedDiscoveryPage]]:
        discovery = self.discover_from_firecrawl_result(initial_result, extra_urls=extra_urls)
        selected = {item.url: item for item in discovery.selected_urls}
        already_crawled = self._source_urls_from_result(initial_result)
        visited = set(already_crawled)
        scraped_pages: list[ScrapedDiscoveryPage] = []

        while len(scraped_pages) < max_scrapes:
            next_url = self._next_url_to_scrape(selected, visited)
            if not next_url:
                break
            visited.add(next_url.url)

            if on_progress:
                await on_progress(
                    {
                        "current_scrape_url": next_url.url,
                        "candidate_urls": discovery.candidate_urls,
                        "selected_urls": [item.url for item in sorted_selected_urls(selected)],
                        "skipped_urls": discovery.skipped_urls,
                        "skip_reasons": discovery.skip_reasons,
                        "scraped_count": len(scraped_pages),
                    }
                )

            try:
                scrape_result = await scraper.scrape(next_url.url)
            except Exception as exc:
                discovery.scrape_errors.append({"url": next_url.url, "error": str(exc)})
                continue

            discovery.scraped_urls.append(next_url.url)
            scraped_pages.append(ScrapedDiscoveryPage(discovered_url=next_url, result=scrape_result))
            for url, discovered_from in self._candidates_from_scrape_result(scrape_result).items():
                self._consider_url(
                    discovery=discovery,
                    selected=selected,
                    url=url,
                    discovered_from=discovered_from or next_url.url,
                    discovery_source="agent_scrape_links",
                )
            discovery.selected_urls = sorted_selected_urls(selected)

        return discovery, scraped_pages

    def _candidates_from_result(self, result: dict[str, Any]) -> dict[str, str | None]:
        candidates: dict[str, str | None] = {}
        for item in result.get("data") or []:
            metadata = dict(item.get("metadata") or {})
            source_url = normalize_url(str(metadata.get("sourceURL") or metadata.get("url") or ""))
            if source_url:
                candidates.setdefault(source_url, None)
            for link in item.get("links") or []:
                normalized = normalize_url(str(link))
                if normalized:
                    candidates.setdefault(normalized, source_url or self.seed_url)
        return candidates

    def _candidates_from_scrape_result(self, result: dict[str, Any]) -> dict[str, str | None]:
        data = result.get("data") or result
        return self._candidates_from_result({"data": [data]})

    def _candidates_from_extra_urls(self, urls: list[str]) -> dict[str, str | None]:
        candidates: dict[str, str | None] = {}
        for url in urls:
            normalized = normalize_url(str(url))
            if normalized:
                candidates.setdefault(normalized, self.seed_url)
        return candidates

    def _source_urls_from_result(self, result: dict[str, Any]) -> set[str]:
        urls = set()
        for item in result.get("data") or []:
            metadata = dict(item.get("metadata") or {})
            for key in ("sourceURL", "url"):
                normalized = normalize_url(str(metadata.get(key) or ""))
                if normalized:
                    urls.add(normalized)
        return urls

    def _seed_entrypoint_candidates(self) -> dict[str, str | None]:
        parsed = urlparse(self.seed_url)
        if not parsed.netloc.endswith("vinwonders.com"):
            return {}
        language = _language_prefix(parsed.path) or "/vi"
        base = f"{parsed.scheme}://{parsed.netloc}{language}"
        candidates = [
            normalize_url(f"{base}/wonderpedia/"),
            normalize_url(f"{base}/promotions/"),
            normalize_url(f"{base}/uu-dai/"),
        ]
        return {url: self.seed_url for url in candidates if url}

    def _next_url_to_scrape(
        self,
        selected: dict[str, DiscoveredUrl],
        visited: set[str],
    ) -> DiscoveredUrl | None:
        for item in sorted_selected_urls(selected):
            if item.url not in visited:
                return item
        return None

    def _consider_url(
        self,
        *,
        discovery: LinkDiscoveryResult,
        selected: dict[str, DiscoveredUrl],
        url: str,
        discovered_from: str | None,
        discovery_source: str,
    ) -> None:
        normalized = normalize_url(url)
        if not normalized:
            return
        reason = skip_reason(
            normalized,
            allowed_domains=None if self._legacy_default_filter else self.allowed_domains,
            include_subdomains=self.include_subdomains,
            exclude_patterns=self.exclude_patterns,
        )
        if reason:
            discovery.skipped_urls.append(normalized)
            discovery.skip_reasons[normalized] = reason
            return
        category = classify_url_category(normalized) or "page"
        if normalized not in selected:
            selected[normalized] = DiscoveredUrl(
                url=normalized,
                category=category,
                discovery_source=discovery_source,
                discovered_from=discovered_from,
            )


def normalize_url(url: str) -> str | None:
    if not url:
        return None
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", "", ""))


def skip_reason(
    url: str,
    *,
    allowed_domains: list[str] | None = None,
    include_subdomains: bool = True,
    exclude_patterns: list[str] | None = None,
) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    domains = [domain.lower() for domain in (allowed_domains or ["vinwonders.com"])]
    if not _host_allowed(host, domains, include_subdomains):
        return "external_domain"
    if re.search(r"\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|mp4|mov|zip|pdf|woff2?|ttf|eot)$", path):
        return "static_asset"
    if "login" in path or "dang-nhap" in path or "register" in path or "dang-ky" in path:
        return "auth_page"
    if any(segment in path for segment in ("/cart", "/checkout", "/account", "/wp-admin")):
        return "auth_or_transaction_page"
    for pattern in exclude_patterns or []:
        if re.search(pattern, url, flags=re.IGNORECASE):
            return "excluded_by_policy"
    if allowed_domains is None:
        if re.search(r"/(ko|zh|ru)(/|$)", path):
            return "unsupported_language"
        if not re.search(r"/(vi|en)(/|$)", path):
            return "unsupported_language"
    return None


def classify_url_category(url: str) -> str | None:
    lowered = url.lower()
    path = urlparse(lowered).path.rstrip("/")
    if path.endswith("/wonderpedia"):
        return "wonderpedia"
    if "affiliate" in lowered:
        return "affiliate"
    if "vinclub" in lowered or "hoi-vien" in lowered:
        return "vinclub"
    if any(keyword in lowered for keyword in ("gia-ve", "price", "ticket", "bang-gia")):
        return "price"
    if "booking" in lowered or "dat-ve" in lowered:
        return "booking"
    if any(keyword in lowered for keyword in ("promotions", "uu-dai", "voucher", "combo", "khuyen-mai")):
        return "offer"
    if "vinpearl-safari-phu-quoc" in lowered:
        return "experience"
    if any(keyword in lowered for keyword in ("service", "dich-vu", "terms-and-conditions", "quy-dinh")):
        return "service"
    return None


def discovery_metadata_for(discovered: DiscoveredUrl) -> dict[str, str]:
    metadata = {
        "discovery_source": discovered.discovery_source,
        "url_category": discovered.category,
    }
    if discovered.discovered_from:
        metadata["discovered_from"] = discovered.discovered_from
    return metadata


def sorted_selected_urls(selected: dict[str, DiscoveredUrl]) -> list[DiscoveredUrl]:
    return sorted(
        selected.values(),
        key=lambda item: (_category_rank(item.category), item.url),
    )


def _category_rank(category: str) -> int:
    ranks = {
        "experience": 0,
        "wonderpedia": 1,
        "price": 2,
        "offer": 3,
        "vinclub": 4,
        "affiliate": 5,
        "booking": 6,
        "service": 7,
    }
    return ranks.get(category, 99)


def _language_prefix(path: str) -> str | None:
    match = re.match(r"^/(vi|en)(?:/|$)", path.lower())
    return f"/{match.group(1)}" if match else None


def _host_allowed(host: str, domains: list[str], include_subdomains: bool) -> bool:
    for domain in domains:
        if host == domain:
            return True
        if include_subdomains and host.endswith(f".{domain}"):
            return True
    return False
