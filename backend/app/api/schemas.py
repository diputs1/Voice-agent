from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class Citation(BaseModel):
    source_url: str
    title: str
    section: str
    category: str
    language: str | None = None
    crawled_at: str | None = None
    valid_until: str | None = None
    metadata: dict | None = None


class ChatRequest(BaseModel):
    transcript: str = Field(min_length=1)
    thread_id: str | None = None
    site_id: str | None = None


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class VoiceAgentKnowledgeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    site_id: str | None = None
    limit: int = Field(default=3, ge=1, le=10)
    conversation_id: str | None = None
    turn_id: str | None = None
    request_id: str | None = None
    tool_call_id: str | None = None
    correlation_id: str | None = None

    @field_validator("site_id", mode="before")
    @classmethod
    def normalize_site_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if normalized.lower() == "default":
            return None
        return normalized or None

    @model_validator(mode="before")
    @classmethod
    def unwrap_elevenlabs_parameters(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        parameters = data.get("parameters")
        if not isinstance(parameters, dict):
            return data
        payload = dict(parameters)
        for key in ("conversation_id", "turn_id", "request_id", "tool_call_id", "correlation_id"):
            if data.get(key) and key not in payload:
                payload[key] = data[key]
        return payload


class VoiceAgentKnowledgeHit(BaseModel):
    content: str
    source_url: str
    title: str
    section: str
    category: str
    score: float
    language: str | None = None
    crawled_at: str | None = None
    valid_until: str | None = None
    metadata: dict | None = None


class VoiceAgentTraceMetadata(BaseModel):
    correlation_id: str
    turn_id: str
    received_at: str
    completed_at: str
    latency_ms: float
    hit_count: int
    top_score: float | None = None
    context_byte_count: int = 0
    context_char_count: int = 0
    cache_hit: bool = False
    rewrite_source: str | None = None
    resolved_query: str | None = None


class VoiceAgentKnowledgeResponse(BaseModel):
    query: str
    site_id: str | None = None
    conversation_id: str | None = None
    correlation_id: str | None = None
    turn_id: str | None = None
    context: list[VoiceAgentKnowledgeHit]
    trace: VoiceAgentTraceMetadata | None = None


class IngestRequest(BaseModel):
    url: str | None = None


class CrawlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    max_depth: int = Field(default=3, ge=0, le=8)
    max_pages: int = Field(default=100, ge=1, le=1000)
    include_subdomains: bool = False
    exclude_patterns: list[str] = Field(default_factory=list)


class CrawlJobResponse(BaseModel):
    job_id: str
    status: str
    url: str
    site_id: str | None = None
