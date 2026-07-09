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


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class IngestRequest(BaseModel):
    url: str | None = None


class CrawlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    max_depth: int = Field(default=2, ge=0, le=4)
    max_pages: int = Field(default=40, ge=1, le=100)


class CrawlJobResponse(BaseModel):
    job_id: str
    status: str
    url: str
