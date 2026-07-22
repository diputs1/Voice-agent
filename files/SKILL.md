---
name: voice-agent-eval-review
description: Runs and interprets local voice-tool harness and ElevenLabs Agent Testing reports for Vin Agent.
when_to_use: voice agent prompt changed, webhook tool changed, ElevenLabs tool config changed, user asks whether voice behavior regressed
---

# Steps
1. Run `cd backend && python -m app.evals.run_voice_loop --mode local`.
2. Confirm `pass_rate` is 1.0 and inspect failures for retrieval, rewrite, cache, or latency issues.
3. Run `cd backend && python -m app.evals.run_voice_loop --mode manifest` and confirm the manifest still covers Tool Call, Next Reply, and Simulation tests.
4. If `ELEVENLABS_API_KEY`, `ELEVENLABS_AGENT_ID`, and test IDs are available, run `python -m app.evals.run_voice_loop --mode elevenlabs --repeat-count 5 --test-id <id>`.
5. Compare report deltas: pass rate, p95 tool latency, missing tool calls, tool result errors, hallucination/forbidden substring flags.
6. Treat any factual voice answer without a backend tool call as a blocking regression.
7. Treat invented current prices, promotions, vouchers, or today/weekend availability as a blocking regression.
