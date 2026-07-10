from __future__ import annotations

import os
from typing import Any

from app.core.config import Settings


def configure_langsmith(settings: Settings) -> dict[str, Any]:
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        if settings.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    api_key = os.getenv("LANGSMITH_API_KEY")
    return {
        "enabled": os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(api_key),
        "project": os.getenv("LANGSMITH_PROJECT") or settings.langsmith_project,
        "endpoint": os.getenv("LANGSMITH_ENDPOINT") or settings.langsmith_endpoint,
        "api_key_configured": bool(api_key),
    }
