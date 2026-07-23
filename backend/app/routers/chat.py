from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    TTSRequest,
    VoiceAgentKnowledgeRequest,
    VoiceAgentKnowledgeResponse,
)
from app.services.chat_service import ChatService
from app.services.voice_agent_service import search_voice_agent_knowledge as search_voice_agent_knowledge_service

router = APIRouter()


@router.post("/voice/stt-token")
async def create_stt_token(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    if not settings.elevenlabs_api_key:
        raise HTTPException(status_code=501, detail="ELEVENLABS_API_KEY is not configured")

    await _enforce_voice_rate_limit(
        request,
        action="stt-token",
        limit=_setting_int(settings, "voice_agent_token_rate_limit_per_minute", 0),
    )

    async with _http_client(request, timeout=20.0) as client:
        response = await client.post(
            "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe",
            headers={"xi-api-key": settings.elevenlabs_api_key},
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@router.post("/voice/agent-token")
async def create_voice_agent_token(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    if not settings.elevenlabs_api_key:
        raise HTTPException(status_code=501, detail="ELEVENLABS_API_KEY is not configured")
    if not settings.elevenlabs_agent_id:
        raise HTTPException(status_code=501, detail="ELEVENLABS_AGENT_ID is not configured")

    await _enforce_voice_rate_limit(
        request,
        action="agent-token",
        limit=_setting_int(settings, "voice_agent_token_rate_limit_per_minute", 0),
    )

    params = {"agent_id": settings.elevenlabs_agent_id}
    if settings.elevenlabs_agent_environment:
        params["environment"] = settings.elevenlabs_agent_environment

    async with _http_client(request, timeout=20.0) as client:
        response = await client.get(
            "https://api.elevenlabs.io/v1/convai/conversation/token",
            headers={"xi-api-key": settings.elevenlabs_api_key},
            params=params,
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@router.post("/voice/tts")
async def text_to_speech(request: Request, payload: TTSRequest) -> StreamingResponse:
    settings = request.app.state.settings
    if not settings.elevenlabs_api_key:
        raise HTTPException(status_code=501, detail="ELEVENLABS_API_KEY is not configured")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}/stream"
        "?output_format=mp3_44100_128"
    )

    async def audio_stream() -> AsyncIterator[bytes]:
        async with _http_client(request, timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                headers={
                    "xi-api-key": settings.elevenlabs_api_key or "",
                    "Content-Type": "application/json",
                },
                json={
                    "text": payload.text,
                    "model_id": settings.elevenlabs_tts_model,
                    "language_code": settings.elevenlabs_tts_language_code,
                },
            ) as response:
                if response.status_code >= 400:
                    detail = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=detail.decode())
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")


@router.post("/voice-agent/tools/search-knowledge", response_model_exclude_none=True)
async def search_voice_agent_knowledge(
    request: Request,
    payload: VoiceAgentKnowledgeRequest,
    authorization: str | None = Header(default=None),
    x_elevenlabs_webhook_secret: str | None = Header(default=None),
) -> VoiceAgentKnowledgeResponse:
    settings = request.app.state.settings
    _require_voice_webhook_secret(settings, authorization, x_elevenlabs_webhook_secret)
    await _enforce_voice_rate_limit(
        request,
        action="voice-tool",
        limit=_setting_int(settings, "voice_tool_rate_limit_per_minute", 0),
        identity=payload.conversation_id or payload.correlation_id,
    )

    return await search_voice_agent_knowledge_service(
        kb=request.app.state.kb,
        payload=payload,
        raw_payload=await _safe_request_json(request),
        log_raw_payload=bool(getattr(settings, "elevenlabs_log_raw_tool_payload", False)),
        retrieval_cache=getattr(request.app.state, "voice_tool_cache", None),
        conversation_memory=getattr(request.app.state, "voice_conversation_memory", None),
    )


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    service: ChatService = request.app.state.chat_service
    return StreamingResponse(service.stream_chat(payload), media_type="text/event-stream")


@router.get("/threads/{thread_id}")
async def get_thread(request: Request, thread_id: str) -> dict[str, object]:
    return {"thread_id": thread_id, "turns": await request.app.state.kb.get_thread(thread_id)}


@asynccontextmanager
async def _http_client(request: Request, *, timeout: float):
    client = getattr(request.app.state, "http_client", None)
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(timeout=timeout) as fallback_client:
        yield fallback_client


async def _enforce_voice_rate_limit(
    request: Request,
    *,
    action: str,
    limit: int,
    identity: str | None = None,
) -> None:
    limiter = getattr(request.app.state, "voice_rate_limiter", None)
    if limiter is None or limit <= 0:
        return
    settings = request.app.state.settings
    window_seconds = _setting_int(settings, "voice_rate_limit_window_seconds", 60)
    key = f"{action}:{identity or _request_identity(request)}"
    if not await limiter.allow(key, limit=limit, window_seconds=window_seconds):
        raise HTTPException(status_code=429, detail="Voice endpoint rate limit exceeded")


def _require_voice_webhook_secret(
    settings: Any,
    authorization: str | None,
    x_elevenlabs_webhook_secret: str | None,
) -> None:
    expected_secret = getattr(settings, "elevenlabs_webhook_secret", None)
    if not expected_secret:
        if _is_local_env(settings):
            return
        raise HTTPException(status_code=500, detail="ELEVENLABS_WEBHOOK_SECRET is not configured")

    expected_bearer = f"Bearer {expected_secret}"
    valid_authorization = isinstance(authorization, str) and compare_digest(authorization, expected_bearer)
    valid_header = isinstance(x_elevenlabs_webhook_secret, str) and compare_digest(
        x_elevenlabs_webhook_secret,
        expected_secret,
    )
    if not valid_authorization and not valid_header:
        raise HTTPException(status_code=401, detail="Invalid voice agent webhook secret")


def _is_local_env(settings: Any) -> bool:
    app_env = str(getattr(settings, "app_env", "development")).lower()
    return app_env in {"development", "dev", "test", "testing", "local"}


def _setting_int(settings: Any, name: str, default: int) -> int:
    value = getattr(settings, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _request_identity(request: Request) -> str:
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if host:
        return str(host)
    headers = getattr(request, "headers", None)
    forwarded_for = _header_value(headers, "x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return "unknown"


def _header_value(headers: Any, name: str) -> str | None:
    getter = getattr(headers, "get", None)
    if not getter:
        return None
    value = getter(name)
    return str(value) if value else None


async def _safe_request_json(request: Request) -> dict | None:
    json_func = getattr(request, "json", None)
    if not json_func:
        return None
    try:
        data = await json_func()
    except Exception:
        return None
    return data if isinstance(data, dict) else None
