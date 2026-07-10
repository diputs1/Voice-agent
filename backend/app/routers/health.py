from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    kb_status = getattr(
        request.app.state,
        "kb_status",
        {"provider": "unknown", "fallback": False, "fallback_reason": None},
    )
    langsmith_status = getattr(
        request.app.state,
        "langsmith_status",
        {"enabled": False, "api_key_configured": False, "project": None},
    )
    return {
        "status": "degraded" if kb_status.get("fallback") else "ok",
        "kb": kb_status,
        "langsmith": langsmith_status,
    }
