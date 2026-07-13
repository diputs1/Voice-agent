from __future__ import annotations

from collections.abc import AsyncIterator

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

    async with httpx.AsyncClient(timeout=20.0) as client:
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

    params = {"agent_id": settings.elevenlabs_agent_id}
    if settings.elevenlabs_agent_environment:
        params["environment"] = settings.elevenlabs_agent_environment

    async with httpx.AsyncClient(timeout=20.0) as client:
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
        async with httpx.AsyncClient(timeout=60.0) as client:
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


@router.post("/voice-agent/tools/search-knowledge")
async def search_voice_agent_knowledge(
    request: Request,
    payload: VoiceAgentKnowledgeRequest,
    authorization: str | None = Header(default=None),
    x_elevenlabs_webhook_secret: str | None = Header(default=None),
) -> VoiceAgentKnowledgeResponse:
    settings = request.app.state.settings
    expected_secret = settings.elevenlabs_webhook_secret
    if expected_secret:
        bearer = f"Bearer {expected_secret}"
        if authorization != bearer and x_elevenlabs_webhook_secret != expected_secret:
            raise HTTPException(status_code=401, detail="Invalid voice agent webhook secret")

    return await search_voice_agent_knowledge_service(
        kb=request.app.state.kb,
        payload=payload,
        raw_payload=await _safe_request_json(request),
        log_raw_payload=bool(getattr(settings, "elevenlabs_log_raw_tool_payload", False)),
    )


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    service: ChatService = request.app.state.chat_service
    return StreamingResponse(service.stream_chat(payload), media_type="text/event-stream")


@router.get("/threads/{thread_id}")
async def get_thread(request: Request, thread_id: str) -> dict[str, object]:
    return {"thread_id": thread_id, "turns": await request.app.state.kb.get_thread(thread_id)}


async def _safe_request_json(request: Request) -> dict | None:
    json_func = getattr(request, "json", None)
    if not json_func:
        return None
    try:
        data = await json_func()
    except Exception:
        return None
    return data if isinstance(data, dict) else None
