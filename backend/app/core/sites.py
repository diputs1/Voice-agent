from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse


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
