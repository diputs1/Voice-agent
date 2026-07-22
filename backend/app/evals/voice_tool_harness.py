from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from app.api.schemas import VoiceAgentKnowledgeRequest
from app.core.cache import TTLQACache
from app.core.config import Settings
from app.core.voice_memory import VoiceConversationMemory
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.kb import InMemoryKnowledgeBase
from app.services.voice_agent_service import search_voice_agent_knowledge


VOICE_CASES_PATH = Path(__file__).with_name("voice_cases.json")
DEFAULT_REPORTS_DIR = Path(__file__).with_name("reports") / "voice"


def load_voice_cases(path: Path | None = None) -> list[dict[str, Any]]:
    cases_path = path or VOICE_CASES_PATH
    return json.loads(cases_path.read_text(encoding="utf-8"))


async def run_local_voice_harness(
    cases: list[dict[str, Any]] | None = None,
    *,
    site_id: str | None = None,
) -> dict[str, Any]:
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, settings.openai_embedding_model))
    await kb.ensure_ready()
    retrieval_cache = TTLQACache(ttl_seconds=300, max_entries=128)
    memory = VoiceConversationMemory(ttl_seconds=1800, max_conversations=256)

    selected_cases = cases or load_voice_cases()
    results = []
    latencies = []
    for case in selected_cases:
        result = await _run_local_case(
            case,
            kb=kb,
            retrieval_cache=retrieval_cache,
            memory=memory,
            site_id=site_id,
        )
        if result.get("latency_ms") is not None:
            latencies.append(float(result["latency_ms"]))
        results.append(result)

    pass_count = sum(1 for item in results if item["passed"])
    executable_count = sum(1 for item in results if item["status"] != "skipped")
    return {
        "mode": "local_voice_tool",
        "passed": pass_count == len(results),
        "pass_rate": pass_count / len(results) if results else 0.0,
        "executable_case_count": executable_count,
        "case_count": len(results),
        "p95_tool_latency_ms": _p95(latencies),
        "avg_tool_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "results": results,
    }


async def _run_local_case(
    case: dict[str, Any],
    *,
    kb: InMemoryKnowledgeBase,
    retrieval_cache: TTLQACache,
    memory: VoiceConversationMemory,
    site_id: str | None,
) -> dict[str, Any]:
    if not case.get("requires_tool_call", False):
        return {
            "id": case["id"],
            "passed": True,
            "status": "skipped",
            "reason": "case is owned by ElevenLabs no-tool behavior tests",
            "errors": [],
        }

    errors: list[str] = []
    conversation_id = f"local-{case['id']}"
    calls = []
    for turn_index, transcript in enumerate(case.get("turns") or []):
        payload_site_id = site_id if site_id is not None else case.get("site_id")
        raw_payload = {
            "conversation_id": conversation_id,
            "turn_id": f"{conversation_id}-{turn_index}",
            "parameters": {
                "query": transcript,
                "site_id": payload_site_id,
                "limit": int(case.get("limit") or 3),
            },
        }
        payload = VoiceAgentKnowledgeRequest.model_validate(raw_payload)
        response = await search_voice_agent_knowledge(
            kb=kb,
            payload=payload,
            raw_payload=raw_payload,
            retrieval_cache=retrieval_cache,
            conversation_memory=memory,
        )
        calls.append(response.model_dump(mode="json"))

    if not calls:
        errors.append("missing local tool call")
        return {
            "id": case["id"],
            "passed": False,
            "status": "failed",
            "errors": errors,
            "calls": [],
        }

    final_call = calls[-1]
    trace = final_call.get("trace") or {}
    context_text = "\n".join(
        str(hit.get("content") or "") for hit in final_call.get("context") or []
    )

    for substring in case.get("expected_context_substrings") or []:
        if substring.lower() not in context_text.lower():
            errors.append(f"missing context substring={substring!r}")

    expected_rewrite_source = case.get("expected_rewrite_source")
    if expected_rewrite_source and trace.get("rewrite_source") != expected_rewrite_source:
        errors.append(f"rewrite_source={trace.get('rewrite_source')!r}")

    resolved_query = str(trace.get("resolved_query") or "")
    for substring in case.get("expected_resolved_query_substrings") or []:
        if substring.lower() not in resolved_query.lower():
            errors.append(f"missing resolved query substring={substring!r}")

    latency_ms = float(trace.get("latency_ms") or 0.0)
    max_latency_ms = case.get("max_tool_latency_ms")
    if max_latency_ms is not None and latency_ms > float(max_latency_ms):
        errors.append(f"latency_ms={latency_ms} > {max_latency_ms}")

    return {
        "id": case["id"],
        "passed": not errors,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "latency_ms": latency_ms,
        "hit_count": trace.get("hit_count"),
        "cache_hit": trace.get("cache_hit"),
        "rewrite_source": trace.get("rewrite_source"),
        "resolved_query": trace.get("resolved_query"),
        "calls": calls,
    }


def write_voice_report(report: dict[str, Any], *, reports_dir: Path | None = None) -> Path:
    from datetime import UTC, datetime

    target_dir = reports_dir or DEFAULT_REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target_dir / f"{timestamp}-{report['mode']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]
