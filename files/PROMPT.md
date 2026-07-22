# Goal
Build a voice-agent harness and engineering loop for the ElevenLabs-backed Vin Agent.

The system uses ElevenLabs Conversational AI as the realtime voice shell and this
FastAPI backend as the retrieval/tool backend. The loop must catch regressions in
backend tool behavior, ElevenLabs tool usage, answer grounding, latency, and
multi-turn follow-up handling.

# Done when
- `backend/app/evals/voice_cases.json` covers greeting, factual stable, time-sensitive, out-of-scope, product/service, and multi-turn follow-up cases.
- `python -m app.evals.run_voice_loop --mode local` runs without external API keys and writes a voice report.
- `python -m app.evals.run_voice_loop --mode manifest` prints the intended ElevenLabs Tool Call, Next Reply, and Simulation test manifest.
- `python -m app.evals.run_voice_loop --mode elevenlabs --test-id <id>` can run existing ElevenLabs Agent Testing cases when `ELEVENLABS_API_KEY` and `ELEVENLABS_AGENT_ID` are configured.
- Transcript/run-test evaluation reports tool calls, tool results, tool latency, and answer substring failures.
- `files/run.sh`, `files/verifier.md`, and `files/CLAUDE.md` reference the current repo layout.
- Relevant pytest coverage is green.

# Never touch
- Do not move realtime voice orchestration from ElevenLabs into backend agent code during this task.
- Do not enable ElevenLabs built-in RAG as the primary source without an explicit experiment.
- Do not weaken webhook auth, rate limits, cache TTLs, or trace metadata.
- Do not remove the existing text `/chat/stream` eval path.

# Stop if
- ElevenLabs API behavior differs from the documented `/v1/convai/agents/:agent_id/run-tests` contract.
- A local backend harness failure points to missing or stale seed data rather than harness logic.
- More than 8 non-eval files need edits to satisfy the loop.
