from pydantic import BaseModel, ConfigDict, Field, HttpUrl


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
