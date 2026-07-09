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
curl -X POST http://localhost:8000/admin/ingest-vinwonders
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

## Firecrawl ingestion

Start a clean web crawl and ingest the returned Markdown into the KB:

```bash
curl -X POST http://localhost:8000/admin/crawl \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://vinwonders.com/vi/vinpearl-safari-phu-quoc/","max_depth":2,"max_pages":40}'
```

Check job status:

```bash
curl http://localhost:8000/admin/crawl-jobs/<job_id>
```
