from __future__ import annotations

from typing import Any

import httpx


ELEVENLABS_API_BASE_URL = "https://api.elevenlabs.io"
ELEVENLABS_TEST_PREFIX = "vin-agent:"


def test_ids_from_cases(cases: list[dict[str, Any]]) -> list[str]:
    test_ids: list[str] = []
    for case in cases:
        for test_id in case.get("elevenlabs_test_ids") or []:
            if isinstance(test_id, str) and test_id:
                test_ids.append(test_id)
    return sorted(set(test_ids))


def build_test_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a local manifest for creating ElevenLabs tests in the dashboard/API.

    The run-tests API requires existing test IDs. Keeping this manifest in the
    repo makes the intended Tool Call, Next Reply, and Simulation coverage
    auditable even before workspace-specific test IDs are attached.
    """
    tests = []
    for case in cases:
        turns = [str(turn) for turn in case.get("turns") or []]
        name = f"{ELEVENLABS_TEST_PREFIX}{case['id']}"
        if case.get("requires_tool_call", False):
            tests.append(
                {
                    "name": f"{name}:tool",
                    "type": "tool",
                    "expected_tool": case.get("expected_tool"),
                    "user_message": turns[-1] if turns else "",
                    "expected_params": {
                        "site_id": case.get("site_id", "default"),
                        "limit_max": case.get("limit", 3),
                    },
                }
            )
        tests.append(
            {
                "name": f"{name}:reply",
                "type": "llm",
                "chat_history": [{"role": "user", "message": turn} for turn in turns],
                "success_condition": _success_condition(case),
            }
        )
        if len(turns) > 1:
            tests.append(
                {
                    "name": f"{name}:simulation",
                    "type": "simulation",
                    "turns": turns,
                    "success_condition": _success_condition(case),
                }
            )
    return {"tests": tests}


async def run_elevenlabs_agent_tests(
    *,
    api_key: str,
    agent_id: str,
    test_ids: list[str],
    repeat_count: int = 1,
    branch_id: str | None = None,
    api_base_url: str = ELEVENLABS_API_BASE_URL,
) -> dict[str, Any]:
    if not test_ids:
        raise ValueError("At least one ElevenLabs test ID is required")
    if repeat_count < 1 or repeat_count > 50:
        raise ValueError("repeat_count must be between 1 and 50")

    payload: dict[str, Any] = {
        "tests": [{"test_id": test_id} for test_id in sorted(set(test_ids))],
        "repeat_count": repeat_count,
    }
    if branch_id:
        payload["branch_id"] = branch_id

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{api_base_url}/v1/convai/agents/{agent_id}/run-tests",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
        )
    response.raise_for_status()
    raw = response.json()
    return normalize_run_tests_response(raw)


def normalize_run_tests_response(raw: dict[str, Any]) -> dict[str, Any]:
    runs = raw.get("test_runs") or []
    normalized_runs = [_normalize_test_run(run) for run in runs if isinstance(run, dict)]
    pass_count = sum(1 for run in normalized_runs if run["passed"])
    tool_latencies = [
        latency
        for run in normalized_runs
        for latency in run.get("tool_latency_secs", [])
        if isinstance(latency, int | float)
    ]
    return {
        "mode": "elevenlabs_agent_testing",
        "passed": pass_count == len(normalized_runs) if normalized_runs else False,
        "pass_rate": pass_count / len(normalized_runs) if normalized_runs else 0.0,
        "invocation_id": raw.get("id"),
        "agent_id": raw.get("agent_id"),
        "branch_id": raw.get("branch_id"),
        "repeat_count": raw.get("repeat_count"),
        "bucketing_status": raw.get("bucketing_status"),
        "test_run_count": len(normalized_runs),
        "tool_latency_p95_secs": _p95(tool_latencies),
        "results": normalized_runs,
        "result_groups": raw.get("result_groups") or [],
        "raw": raw,
    }


def _normalize_test_run(run: dict[str, Any]) -> dict[str, Any]:
    condition = run.get("condition_result") if isinstance(run.get("condition_result"), dict) else {}
    result = str(condition.get("result") or run.get("status") or "").lower()
    responses = run.get("agent_responses") or []
    tool_calls = _extract_nested_items(responses, "tool_calls")
    tool_results = _extract_nested_items(responses, "tool_results")
    return {
        "test_id": run.get("test_id"),
        "test_name": run.get("test_name"),
        "test_run_id": run.get("test_run_id"),
        "status": run.get("status"),
        "passed": result in {"success", "passed"},
        "condition_result": condition,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "tool_latency_secs": [
            item.get("tool_latency_secs")
            for item in tool_results
            if isinstance(item.get("tool_latency_secs"), int | float)
        ],
    }


def _extract_nested_items(items: Any, key: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return found
    for item in items:
        if not isinstance(item, dict):
            continue
        nested = item.get(key)
        if isinstance(nested, list):
            found.extend(value for value in nested if isinstance(value, dict))
    return found


def _success_condition(case: dict[str, Any]) -> str:
    required = ", ".join(case.get("expected_answer_substrings") or [])
    forbidden = ", ".join(case.get("forbidden_answer_substrings") or [])
    tool_rule = (
        f"Must call {case.get('expected_tool')} before answering factual Vinpearl Safari questions. "
        if case.get("requires_tool_call")
        else "Should not call the factual search tool for this turn. "
    )
    freshness_rule = (
        "For time-sensitive pricing, promotions, vouchers, or today/weekend availability, "
        "do not invent exact current values; direct the user to the official VinWonders site or hotline. "
        if case.get("time_sensitive")
        else ""
    )
    return (
        f"{tool_rule}{freshness_rule}"
        f"The final Vietnamese answer should include: {required or 'a relevant concise answer'}. "
        f"It must not include: {forbidden or 'unsupported facts'}."
    )


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]
