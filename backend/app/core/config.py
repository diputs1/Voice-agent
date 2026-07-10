from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    database_url: str = "postgresql://vin_agent:vin_agent@localhost:5432/vin_agent"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    firecrawl_api_key: str | None = None
    firecrawl_base_url: str = "https://api.firecrawl.dev"
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    elevenlabs_tts_model: str = "eleven_multilingual_v2"
    vinwonders_source_url: str = "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/"
    cors_origins: str = Field(default="http://localhost:3000")
    low_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    allow_kb_fallback: bool = True
    admin_api_key: str | None = None
    agent_max_concurrency: int = Field(default=8, ge=1)
    qa_cache_ttl_seconds: int = Field(default=300, ge=0)
    qa_cache_max_entries: int = Field(default=256, ge=1)
    agent_runtime_mode: str = Field(default="website")
    full_agent_max_iterations: int = Field(default=4, ge=1)
    full_agent_timeout_seconds: float = Field(default=6.0, gt=0)
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "vin-agent"
    langsmith_endpoint: str | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
