from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse, urlunparse


@dataclass(frozen=True)
class SiteRelevanceProfile:
    site_id: str
    aliases: tuple[str, ...]
    url_markers: tuple[str, ...] = ()
    off_topic_aliases: tuple[str, ...] = ()
    offer_path_penalty: str | None = None


SITE_RELEVANCE_PROFILES: tuple[SiteRelevanceProfile, ...] = (
    SiteRelevanceProfile(
        site_id="vinpearl-safari-phu-quoc",
        aliases=(
            "vinpearl safari phú quốc",
            "vinpearl safari phu quoc",
            "vinpearl-safari-phu-quoc",
        ),
        url_markers=("/vinpearl-safari-phu-quoc/",),
        offer_path_penalty="/vi/uu-dai/",
        off_topic_aliases=(
            "nha trang",
            "vũ yên",
            "vu yen",
            "hà nội",
            "ha noi",
            "ocean city",
            "horse academy",
            "đất nước thiên hùng ca",
            "grand world",
            "vinwonders phú quốc",
            "vinwonders phu quoc",
        ),
    ),
)


def canonical_root_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip().rstrip("/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), "/", "", "", "")).rstrip("/")


def site_id_for_url(url: str) -> str:
    root = canonical_root_url(url)
    parsed = urlparse(root)
    host = parsed.netloc or root
    slug = re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-")
    return slug or "default"


def relevance_profile_for_text(site_id: str | None, text: str) -> SiteRelevanceProfile | None:
    normalized_site_id = (site_id or "").lower()
    normalized_text = text.lower()
    for profile in SITE_RELEVANCE_PROFILES:
        if normalized_site_id == profile.site_id:
            return profile
        if any(alias in normalized_text for alias in profile.aliases + profile.url_markers):
            return profile
    return None
