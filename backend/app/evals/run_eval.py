from __future__ import annotations

import asyncio
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from app.agents.graph import SafariAgentGraph
from app.agents.graphs import WebsiteAgentGraph
from app.core.config import Settings
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.kb import InMemoryKnowledgeBase


CASES_PATH = Path(__file__).with_name("cases.json")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=["safari", "website", "full_agent", "compare"], default="safari")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--site-id", default=None)
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, settings.openai_embedding_model))
    await kb.ensure_ready()

    modes = ["safari", "full_agent"] if args.graph == "compare" else [args.graph]
    reports = {}
    for mode in modes:
        graph = _build_graph(mode, kb, settings)
        reports[mode] = await _run_cases(graph, cases, runs=max(1, args.runs), site_id=args.site_id)

    passed = all(report["passed"] for report in reports.values())
    print(json.dumps({"passed": passed, "graph": args.graph, "reports": reports}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def _build_graph(mode: str, kb: InMemoryKnowledgeBase, settings: Settings):
    if mode == "safari":
        return SafariAgentGraph(kb, settings)
    return WebsiteAgentGraph(kb, settings)


async def _run_cases(graph, cases: list[dict[str, Any]], *, runs: int, site_id: str | None) -> dict[str, Any]:
    results = []
    latencies = []
    for run_idx in range(runs):
        for case in cases:
            start = time.perf_counter()
            result = await graph.ainvoke(
                {"transcript": case["transcript"], "site_id": site_id},
                f"eval-{getattr(graph, 'graph_version', 'safari')}-{case['id']}-{run_idx}",
            )
            latencies.append(time.perf_counter() - start)
            evaluated = _evaluate_case(case, result)
            evaluated["latency_seconds"] = latencies[-1]
            evaluated["tool_call_count"] = int(result.get("tool_call_count") or 0)
            evaluated["agent_iteration_count"] = int(result.get("agent_iteration_count") or 0)
            results.append(evaluated)
    pass_count = sum(1 for item in results if item["passed"])
    return {
        "passed": pass_count == len(results),
        "pass_rate": pass_count / len(results) if results else 0.0,
        "p95_latency_seconds": _p95(latencies),
        "avg_tool_calls": statistics.fmean(item["tool_call_count"] for item in results) if results else 0.0,
        "avg_agent_iterations": statistics.fmean(item["agent_iteration_count"] for item in results) if results else 0.0,
        "results": results,
    }


def _evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    errors = []
    answer = str(result.get("answer") or "")
    if result.get("intent") != case["expected_intent"]:
        errors.append(f"intent={result.get('intent')!r}")
    if bool(result.get("handoff_required")) != bool(case["expected_handoff"]):
        errors.append(f"handoff_required={result.get('handoff_required')!r}")
    for substring in case.get("required_substrings", []):
        if substring.lower() not in answer.lower():
            errors.append(f"missing substring={substring!r}")
    if case.get("requires_citations") and not result.get("citations"):
        errors.append("missing citations")
    return {
        "id": case["id"],
        "passed": not errors,
        "errors": errors,
        "intent": result.get("intent"),
        "handoff_required": result.get("handoff_required"),
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
