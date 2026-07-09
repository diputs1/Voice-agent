from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.agents.graph import SafariAgentGraph
from app.config import Settings
from app.embeddings import EmbeddingProvider
from app.kb import InMemoryKnowledgeBase


CASES_PATH = Path(__file__).with_name("cases.json")


async def main() -> int:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, settings.openai_embedding_model))
    await kb.ensure_ready()
    graph = SafariAgentGraph(kb, settings)

    results = []
    for case in cases:
        result = await graph.ainvoke({"transcript": case["transcript"]}, f"eval-{case['id']}")
        results.append(_evaluate_case(case, result))

    passed = all(item["passed"] for item in results)
    print(json.dumps({"passed": passed, "results": results}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
