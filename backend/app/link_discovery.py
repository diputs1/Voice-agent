from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
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


class LinkDiscoveryAgent:
    def __init__(self, *, seed_url: str, fallback_urls: list[str] | None = None) -> None:
        self.seed_url = normalize_url(seed_url) or seed_url
        self.fallback_urls = fallback_urls or []

    def discover_from_firecrawl_result(self, result: dict[str, Any]) -> LinkDiscoveryResult:
        discovery = LinkDiscoveryResult()
        selected: dict[str, DiscoveredUrl] = {}
        candidates = self._candidates_from_result(result)
        discovery.candidate_urls = list(candidates)

        for url, discovered_from in candidates.items():
            self._consider_url(
                discovery=discovery,
                selected=selected,
                url=url,
                discovered_from=discovered_from,
                discovery_source="firecrawl_links",
            )

        if len(selected) < MIN_SELECTED_URLS:
            for url in self.fallback_urls:
                self._consider_url(
                    discovery=discovery,
                    selected=selected,
                    url=url,
                    discovered_from=self.seed_url,
                    discovery_source="fallback",
                )

        discovery.selected_urls = sorted(
            selected.values(),
            key=lambda item: (_category_rank(item.category), item.url),
        )
        return discovery

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
        reason = skip_reason(normalized)
        if reason:
            discovery.skipped_urls.append(normalized)
            discovery.skip_reasons[normalized] = reason
            return
        category = classify_url_category(normalized)
        if not category:
            discovery.skipped_urls.append(normalized)
            discovery.skip_reasons[normalized] = "not_relevant_to_vin_safari"
            return
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


def skip_reason(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if not host.endswith("vinwonders.com"):
        return "external_domain"
    if re.search(r"\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|mp4|mov|zip|pdf)$", path):
        return "static_asset"
    if re.search(r"/(ko|zh|ru)(/|$)", path):
        return "unsupported_language"
    if "login" in path or "dang-nhap" in path or "register" in path or "dang-ky" in path:
        return "auth_page"
    if not re.search(r"/(vi|en)(/|$)", path):
        return "unsupported_language"
    return None


def classify_url_category(url: str) -> str | None:
    lowered = url.lower()
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


def _category_rank(category: str) -> int:
    ranks = {
        "experience": 0,
        "price": 1,
        "offer": 2,
        "vinclub": 3,
        "affiliate": 4,
        "booking": 5,
        "service": 6,
    }
    return ranks.get(category, 99)
