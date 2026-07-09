# Vin Agent

LangGraph-first voice Q&A MVP for Vinpearl Safari Phu Quoc.

## Architecture

- `backend/`: FastAPI + LangGraph orchestration, Firecrawl-powered VinWonders ingestion, pgvector knowledge base, ElevenLabs TTS/STT token endpoints.
- `frontend/`: Next.js voice chat UI using ElevenLabs Realtime STT.
- `docker-compose.yml`: local PostgreSQL with pgvector.

OpenAI is used only as the LLM and embedding provider. ElevenLabs handles speech-to-text and text-to-speech.

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

4. Ingest VinWonders content:

```bash
curl -X POST http://localhost:8000/admin/ingest-vinwonders \
  -H "X-Admin-API-Key: $ADMIN_API_KEY"
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

Runtime controls for production traffic:

```bash
AGENT_MAX_CONCURRENCY=8
QA_CACHE_TTL_SECONDS=300
QA_CACHE_MAX_ENTRIES=256
```

`AGENT_MAX_CONCURRENCY` limits concurrent `/chat/stream` agent runs per backend process. QA cache keys use the contextualized query plus the current KB document-set hash, so cached answers expire by TTL and naturally miss after ingestion changes the KB.

## Firecrawl ingestion

Start a clean web crawl and ingest the returned Markdown into the KB:

```bash
curl -X POST http://localhost:8000/admin/crawl \
  -H 'Content-Type: application/json' \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  -d '{"url":"https://vinwonders.com/vi/vinpearl-safari-phu-quoc/","max_depth":2,"max_pages":40}'
```

Check job status:

```bash
curl http://localhost:8000/admin/crawl-jobs/<job_id> \
  -H "X-Admin-API-Key: $ADMIN_API_KEY"
```

## Lightweight eval

Run the local seed-KB eval suite without external services:

```bash
cd backend
python -m app.evals.run_eval
```
