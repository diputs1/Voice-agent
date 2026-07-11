import asyncio
import json
from collections.abc import Callable

import pytest

from app.agents.config import AgentConfig
from app.agents.graphs import WebsiteAgentGraph
from app.core.cache import TTLQACache
from app.core.config import Settings
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.kb import InMemoryKnowledgeBase
from app.services.chat_service import ChatService


@pytest.fixture
def integration_settings() -> Settings:
    return Settings(openai_api_key=None)


@pytest.fixture
def integration_agent_config(integration_settings: Settings) -> AgentConfig:
    return AgentConfig.from_settings(integration_settings)


@pytest.fixture
async def integration_kb(integration_settings: Settings) -> InMemoryKnowledgeBase:
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, integration_settings.openai_embedding_model))
    await kb.ensure_ready()
    return kb


@pytest.fixture
def integration_chat_service(
    integration_settings: Settings,
    integration_agent_config: AgentConfig,
    integration_kb: InMemoryKnowledgeBase,
) -> ChatService:
    return ChatService(
        agent_graph=WebsiteAgentGraph(
            integration_kb,
            integration_settings,
            agent_config=integration_agent_config,
        ),
        kb=integration_kb,
        agent_semaphore=asyncio.Semaphore(1),
        qa_cache=TTLQACache(ttl_seconds=60, max_entries=8),
    )


@pytest.fixture
def parse_sse_event() -> Callable[[str], dict]:
    def parse(raw: str) -> dict:
        lines = raw.strip().splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        return {"event": event, "data": data}

    return parse
