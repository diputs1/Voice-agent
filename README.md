# Vin Agent

LangGraph-first voice Q&A backend for crawling a website into a knowledge base and answering by text/voice.

## Architecture

- `backend/`: FastAPI + LangGraph orchestration, Firecrawl-powered website ingestion, PostgreSQL/pgvector knowledge base, ElevenLabs voice-agent token and webhook-tool endpoints.
- `frontend/`: Next.js voice chat UI using ElevenLabs Voice Agent sessions.
- `docker-compose.yml`: local PostgreSQL with pgvector.

The main voice UI uses ElevenLabs Voice Agent for speech, reasoning, and spoken responses. Backend retrieval stays in this app: ElevenLabs calls the webhook tool, and the backend searches the PostgreSQL/pgvector KB. OpenAI may still be used for embeddings and the legacy LangGraph `/chat/stream` path.

## Project Structure

```text
Vin_agent/
|
├── 📂 backend/                          # 🚀 PRODUCTION BACKEND
│   │
│   ├── 📂 app/                          # FastAPI application package
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI app entrypoint
│   │   │
│   │   ├── 📂 agents/                   # 🤖 LANGGRAPH AI AGENTS
│   │   │   ├── __init__.py
│   │   │   ├── graph.py                 # Graph construction helpers
│   │   │   ├── schemas.py               # Agent-facing structured models
│   │   │   ├── state.py                 # Shared LangGraph state
│   │   │   ├── 📂 graphs/
│   │   │   │   └── website_graph.py     # Main website Q&A graph
│   │   │   ├── 📂 nodes/                # Agent execution nodes
│   │   │   │   ├── contextualize.py     # Rewrite/contextualize user input
│   │   │   │   ├── direct_response.py   # Non-RAG/direct answer path
│   │   │   │   ├── escalation.py        # Escalation behavior
│   │   │   │   ├── offer_freshness.py   # Freshness checks for offers/prices
│   │   │   │   ├── retry_query.py       # Query retry path
│   │   │   │   ├── safari_knowledge.py  # Knowledge retrieval node
│   │   │   │   ├── supervisor.py        # Routing/supervision node
│   │   │   │   └── voice_answer.py      # Voice-friendly answer generation
│   │   │   ├── 📂 logic/                # Retrieval and answer helpers
│   │   │   │   ├── answer_grounding.py  # Source grounding and citations
│   │   │   │   ├── intent.py            # Intent detection
│   │   │   │   └── query_rewrite.py     # Query normalization/rewrite logic
│   │   │   ├── 📂 tools/
│   │   │   │   └── knowledge_tools.py   # Agent tool wrapper for KB search
│   │   │   ├── 📂 edges/
│   │   │   │   └── route_rules.py       # Conditional graph routing
│   │   │   ├── 📂 evaluators/
│   │   │   │   └── grounding.py         # Grounding quality checks
│   │   │   ├── 📂 config/
│   │   │   │   └── agent_config.py      # Agent behavior/config controls
│   │   │   ├── 📂 checkpointers/
│   │   │   │   └── memory.py            # In-memory graph checkpointing
│   │   │   └── 📂 prompts/
│   │   │       └── loader.py            # Prompt loading utilities
│   │   │
│   │   ├── 📂 crawling/                 # 🕸️ WEBSITE CRAWLING & CHUNKING
│   │   │   ├── __init__.py
│   │   │   ├── firecrawl.py             # Firecrawl client + Markdown chunking
│   │   │   ├── ingestion.py             # HTML fallback ingestion + chunk models
│   │   │   ├── jobs.py                  # Crawl job persistence
│   │   │   └── link_discovery.py        # URL discovery/filtering rules
│   │   │
│   │   ├── 📂 knowledge/                # 📚 RAG KNOWLEDGE BASE
│   │   │   ├── __init__.py
│   │   │   ├── embeddings.py            # OpenAI/fallback embedding provider
│   │   │   └── kb.py                    # PostgreSQL pgvector + in-memory KB
│   │   │
│   │   ├── 📂 services/                 # 📦 CORE BUSINESS SERVICES
│   │   │   ├── __init__.py
│   │   │   ├── app_factory.py           # App state/bootstrap wiring
│   │   │   ├── chat_service.py          # Chat streaming orchestration
│   │   │   └── ingestion_service.py     # Crawl → chunks → embeddings → DB
│   │   │
│   │   ├── 📂 routers/                  # 🌐 HTTP ROUTES
│   │   │   ├── __init__.py
│   │   │   ├── admin.py                 # Admin crawl/ingestion/status APIs
│   │   │   ├── chat.py                  # Chat + voice token endpoints
│   │   │   └── health.py                # Health check endpoint
│   │   │
│   │   ├── 📂 api/                      # 📋 API SCHEMAS
│   │   │   ├── __init__.py
│   │   │   └── schemas.py               # Request/response validation models
│   │   │
│   │   ├── 📂 core/                     # ⚙️ CONFIGURATION & INFRA HELPERS
│   │   │   ├── __init__.py
│   │   │   ├── cache.py                 # QA cache implementation
│   │   │   ├── config.py                # Environment settings
│   │   │   ├── observability.py         # LangSmith/tracing setup
│   │   │   └── sites.py                 # Site IDs, roots, relevance helpers
│   │   │
│   │   ├── 📂 prompts/                  # 📝 LLM PROMPT TEMPLATES
│   │   │   ├── __init__.py
│   │   │   ├── escalation.md            # Escalation response prompt
│   │   │   └── voice_answer.md          # Voice answer style prompt
│   │   │
│   │   ├── 📂 evals/                    # 🧪 LIGHTWEIGHT EVALUATION
│   │   │   ├── __init__.py
│   │   │   ├── cases.json               # Eval dataset
│   │   │   └── run_eval.py              # Local eval runner
│   │   │
│   │   └── 📂 tests/                    # ✅ BACKEND TEST SUITE
│   │       ├── integration/             # Integration graph-flow tests
│   │       └── test_*.py                # Unit tests for agents, KB, crawl, API
│   │
│   ├── SKILL.md                         # Codex/backend working notes
│   └── pyproject.toml                   # Python package, deps, test config
│
├── 📂 frontend/                         # 🎙️ NEXT.JS VOICE CHAT UI
│   ├── 📂 app/
│   │   ├── globals.css                  # Global styles
│   │   ├── layout.tsx                   # App shell
│   │   └── page.tsx                     # Main voice/chat experience
│   ├── 📂 lib/
│   │   └── api.ts                       # Backend API client helpers
│   ├── package.json                     # Frontend dependencies/scripts
│   ├── next.config.ts                   # Next.js config
│   └── tsconfig.json                    # TypeScript config
│
├── docker-compose.yml                   # 🗄️ Local PostgreSQL + pgvector
├── .gitignore                           # Ignored local/env/build artifacts
└── README.md                            # Project guide
```

## Quick Start

1. Copy environment files:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

2. Start Postgres:

```bash
docker compose up -d postgres
```

3. Install and run backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

4. Crawl a website:

```bash
curl -X POST http://localhost:8000/admin/crawl \
  -H 'Content-Type: application/json' \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  -d '{"url":"https://example.com/","max_depth":3,"max_pages":100,"include_subdomains":false}'
```

5. Install and run frontend:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

If `OPENAI_API_KEY` or Postgres is missing, the backend falls back to deterministic local embeddings and an in-memory seed KB so tests and UI development still work.

For full web ingestion, set `FIRECRAWL_API_KEY` in `backend/.env`. Without it, `/admin/ingest-vinwonders` uses a one-page fallback and `/admin/crawl` returns `501`.

For production-like environments, set `ADMIN_API_KEY`; admin endpoints fail closed outside local/dev if it is missing.
Optional LangSmith tracing uses the standard LangChain env vars:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=vin-agent
```

The ElevenLabs tool webhook is traced as `elevenlabs.search_knowledge_tool`; the nested
retrieval spans include query embedding, hybrid SQL, and reranking when Postgres is used.

### Embedding latency

Knowledge retrieval embeds the user query with the official async OpenAI client directly,
instead of routing query embeddings through the LangChain embedding wrapper. The
`EmbeddingProvider` keeps one async client per backend process, batches document embeddings
for ingestion, and keeps a small in-memory LRU cache for repeated query embeddings. In
LangSmith, check `kb.embed_query` separately from `kb.hybrid_sql`; warm embedding calls should
be much lower than cold process startup calls, while cache hits should be near-zero latency.

Runtime controls for production traffic:

```bash
AGENT_MAX_CONCURRENCY=8
QA_CACHE_TTL_SECONDS=300
QA_CACHE_MAX_ENTRIES=256
ELEVENLABS_API_KEY=...
ELEVENLABS_AGENT_ID=agent_...
ELEVENLABS_WEBHOOK_SECRET=choose-a-shared-secret
ELEVENLABS_LOG_RAW_TOOL_PAYLOAD=false
```

`AGENT_MAX_CONCURRENCY` limits concurrent `/chat/stream` agent runs per backend process. QA cache keys use the contextualized query plus the current KB document-set hash, so cached answers expire by TTL and naturally miss after ingestion changes the KB.

## ElevenLabs Voice Agent

The frontend starts a private WebRTC conversation by calling:

```text
POST /voice/agent-token
```

The backend uses `ELEVENLABS_API_KEY` and `ELEVENLABS_AGENT_ID` to request a temporary conversation token from ElevenLabs. The API key is never exposed to the browser.

Configure a webhook tool in the ElevenLabs Agent dashboard:

```text
Name: search_vinpearl_safari_knowledge
Method: POST
URL: https://two-doodles-turn.loca.lt/voice-agent/tools/search-knowledge
Headers:
  Authorization: Bearer <ELEVENLABS_WEBHOOK_SECRET>
Body parameters:
  query   string, required, the user's factual Vinpearl Safari question
  site_id string, optional, use "default" unless you route multiple sites
  limit   integer, optional, default 5
```

The endpoint accepts both direct JSON bodies such as `{"query":"...","site_id":"default","limit":5}` and ElevenLabs-style wrapped tool payloads such as `{"conversation_id":"...","parameters":{"query":"..."}}`.
It also accepts optional `correlation_id`, `turn_id`, `request_id`, or `tool_call_id` fields.
If none is provided, the backend generates a `correlation_id` and returns it with the tool
response so LangSmith runs can be matched with ElevenLabs conversation logs.

Tool description:

```text
Use this tool whenever the user asks for factual, operational, pricing, schedule, booking, policy, service, attraction, or location information about Vinpearl Safari Phu Quoc. Do not answer factual Vinpearl Safari questions from memory. Always call this tool first. If the returned context is not enough, say the information needs to be checked on the official website or hotline.
```

Agent prompt guidance:

```text
Bạn là trợ lý voice cho Vinpearl Safari Phú Quốc.
Với mọi câu hỏi về thông tin thực tế của Safari, luôn gọi tool search_vinpearl_safari_knowledge trước khi trả lời.
Chỉ trả lời dựa trên context tool trả về.
Không tự bịa giá vé, giờ mở cửa, lịch show, ưu đãi, chính sách hoặc dịch vụ.
Với chào hỏi, cảm ơn, hoặc hội thoại xã giao, có thể trả lời trực tiếp không cần gọi tool.
Trả lời ngắn gọn, tự nhiên, phù hợp hội thoại bằng giọng nói.
```

The webhook response contains `context` hits with `content`, `source_url`, `title`, `section`, `category`, `score`, and metadata. It also includes `correlation_id`, `turn_id`, and a `trace` object with receive/complete timestamps, backend latency, hit count, and top score. The backend does not generate the final voice answer on this path; ElevenLabs Agent does.

`ELEVENLABS_LOG_RAW_TOOL_PAYLOAD=true` logs the raw incoming tool payload at the webhook
boundary for short debugging windows. Keep it off in normal production traffic unless your
log sink is approved for conversation content.

## Firecrawl ingestion

Start a clean web crawl and ingest the returned Markdown into the KB:

```bash
curl -X POST http://localhost:8000/admin/crawl \
  -H 'Content-Type: application/json' \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  -d '{"url":"https://vinwonders.com/vi/vinpearl-safari-phu-quoc/","max_depth":3,"max_pages":100,"include_subdomains":false,"exclude_patterns":[".*login.*"]}'
```

Check job status:

```bash
curl http://localhost:8000/admin/crawl-jobs/<job_id> \
  -H "X-Admin-API-Key: $ADMIN_API_KEY"
```

Check KB status for a site:

```bash
curl http://localhost:8000/admin/sites/<site_id>/status \
  -H "X-Admin-API-Key: $ADMIN_API_KEY"
```

## Lightweight eval

Run the local seed-KB eval suite without external services:

```bash
cd backend
python -m app.evals.run_eval --graph compare --runs 3
```
