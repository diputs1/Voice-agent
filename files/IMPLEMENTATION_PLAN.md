# Implementation Plan - Voice Agent Harness And Engineering Loop

STATUS: done

## Steps
1. [x] Add `backend/app/evals/voice_cases.json` as the voice golden set.
2. [x] Add local backend voice-tool harness that runs without ElevenLabs API keys.
3. [x] Add ElevenLabs Agent Testing API runner and local test manifest generation.
4. [x] Extend transcript evaluation to report tool calls, tool results, and tool latency.
5. [x] Normalize `site_id=default` to unscoped backend retrieval for the current Vin Agent setup.
6. [x] Add/refresh pytest coverage for local harness, ElevenLabs response normalization, and transcript tool checks.
7. [x] Update README commands for local, manifest, and ElevenLabs voice-loop runs.
8. [x] Run focused tests and local voice harness.
9. [x] Mark `STATUS: done` only after verification is green.

## Log
- 2026-07-22 - Created the voice golden set, local harness, ElevenLabs run-tests client, and unified `run_voice_loop` CLI.
- 2026-07-22 - Extended voice conversation evaluation to parse tool calls/results and latency from ElevenLabs-style transcripts/results.
- 2026-07-22 - Replaced stale loop instructions with the current `backend/app/...` repo layout and voice-agent goal.

<!--
The loop reads this file after each iteration. Keep one actionable unchecked step
at the top of the remaining work so a fresh-context agent can continue without
re-deriving the project state.
-->

- 2026-07-22 - Verified with ruff, full backend pytest, existing graph eval, manifest generation, and local voice harness.
