# Project: Vin Agent

Voice/text RAG assistant for Vinpearl Safari Phu Quoc. The production voice path
uses ElevenLabs Conversational AI for realtime conversation, ASR/TTS, and spoken
responses. Backend retrieval remains in this repo through a secured webhook tool.

## Current Stack
- `backend/`: FastAPI, LangGraph text path, PostgreSQL/pgvector or in-memory KB fallback, Firecrawl ingestion, ElevenLabs token/tool endpoints.
- `frontend/`: Next.js UI using `@elevenlabs/react` WebRTC sessions.
- `backend/app/evals/`: local graph evals, voice transcript evals, and voice-agent harnesses.
- `files/`: disk-backed engineering loop prompts, verifier, and run script.

## Voice Architecture
- Browser calls `POST /voice/agent-token` to get a private ElevenLabs conversation token.
- ElevenLabs calls `POST /voice-agent/tools/search-knowledge` for factual Safari questions.
- `backend/app/services/voice_agent_service.py` performs retrieval, cache, follow-up rewrite, trace metadata, and compact context formatting.
- The backend does not generate the final spoken answer on this path; ElevenLabs does.

## Harness Commands
- `cd backend && python -m app.evals.run_eval --graph compare --runs 3` - legacy/local text graph gate.
- `cd backend && python -m app.evals.run_voice_loop --mode local` - deterministic backend voice-tool gate.
- `cd backend && python -m app.evals.run_voice_loop --mode manifest` - print intended ElevenLabs tests to create or review.
- `cd backend && ELEVENLABS_API_KEY=... ELEVENLABS_AGENT_ID=... python -m app.evals.run_voice_loop --mode elevenlabs --test-id test_x --repeat-count 5` - native ElevenLabs Agent Testing gate.

## Conventions
- Factual Vinpearl Safari voice questions must use the backend search tool before final answer.
- Time-sensitive price, promotion, voucher, today/weekend, or booking claims must not be invented.
- `site_id=default` means unscoped/default Vin Agent retrieval unless a real multi-site ID is configured.
- Tool responses must preserve `correlation_id`, `turn_id`, latency, hit count, cache status, and rewrite source for debugging.
- ElevenLabs built-in RAG can be evaluated as an experiment, but backend KB remains the primary source by default.

## Never
- Never weaken `ELEVENLABS_WEBHOOK_SECRET` checks or production fail-closed behavior.
- Never expose `ELEVENLABS_API_KEY` to the frontend.
- Never move ElevenLabs calls into `backend/app/agents/` orchestration logic without a separate design decision.
- Never merge voice prompt/tool changes without local harness output and, when test IDs exist, an ElevenLabs run-tests report.
