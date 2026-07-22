from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import httpx


ELEVENLABS_CONVERSATION_URL = "https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}"


async def fetch_elevenlabs_conversation(
    *,
    conversation_id: str,
    api_key: str,
    response_format: str = "json",
) -> dict[str, Any]:
    params = {"format": response_format} if response_format else None
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            ELEVENLABS_CONVERSATION_URL.format(conversation_id=conversation_id),
            headers={"xi-api-key": api_key},
            params=params,
        )
    response.raise_for_status()
    return response.json()


def evaluate_voice_conversation(
    conversation: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    min_similarity: float = 0.72,
) -> dict[str, Any]:
    turns = extract_turns(conversation)
    tool_calls = extract_tool_calls(conversation)
    tool_results = extract_tool_results(conversation)
    results = []
    for case in cases:
        result = _evaluate_case_against_turns(
            case,
            turns,
            tool_calls=tool_calls,
            tool_results=tool_results,
            min_similarity=min_similarity,
        )
        results.append(result)

    pass_count = sum(1 for item in results if item["passed"])
    return {
        "passed": pass_count == len(results),
        "pass_rate": pass_count / len(results) if results else 0.0,
        "conversation_id": conversation.get("conversation_id"),
        "agent_id": conversation.get("agent_id"),
        "status": conversation.get("status"),
        "turn_count": len(turns),
        "tool_call_count": len(tool_calls),
        "tool_result_count": len(tool_results),
        "tool_latency_p95_secs": _p95(
            [
                float(item["tool_latency_secs"])
                for item in tool_results
                if isinstance(item.get("tool_latency_secs"), int | float)
            ]
        ),
        "matched_case_count": sum(1 for item in results if item["matched_transcript"]),
        "results": results,
    }


def extract_turns(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    transcript = conversation.get("transcript") or []
    if not isinstance(transcript, list):
        return []

    turns: list[dict[str, Any]] = []
    for entry in transcript:
        if not isinstance(entry, dict):
            continue
        role = _normalize_role(str(entry.get("role") or ""))
        message = _entry_message(entry)
        if not role or not message:
            continue
        turns.append(
            {
                "role": role,
                "message": message,
                "time_in_call_secs": entry.get("time_in_call_secs"),
            }
        )
    return turns


def extract_tool_calls(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_nested_tool_items(conversation, "tool_calls")


def extract_tool_results(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_nested_tool_items(conversation, "tool_results")


def _evaluate_case_against_turns(
    case: dict[str, Any],
    turns: list[dict[str, Any]],
    *,
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    min_similarity: float,
) -> dict[str, Any]:
    errors = []
    expected_transcript = _case_transcript(case)
    user_turn = _best_user_turn(expected_transcript, turns)
    matched_transcript = (
        user_turn["message"] if user_turn and user_turn["similarity"] >= min_similarity else None
    )
    similarity = round(float(user_turn["similarity"]), 3) if user_turn else 0.0
    answer = (
        _next_agent_answer(turns, user_turn["index"]) if matched_transcript and user_turn else ""
    )

    if not matched_transcript:
        errors.append("missing user turn")
    if matched_transcript and not answer:
        errors.append("missing agent answer")
    for substring in _required_answer_substrings(case):
        if substring.lower() not in answer.lower():
            errors.append(f"missing substring={substring!r}")
    for substring in case.get("forbidden_answer_substrings", []):
        if substring.lower() in answer.lower():
            errors.append(f"forbidden substring={substring!r}")

    matched_tool_calls = _matching_tool_calls(case, tool_calls)
    matched_tool_results = _matching_tool_results(case, tool_results)
    if case.get("requires_tool_call") and not matched_tool_calls:
        errors.append(f"missing tool call={case.get('expected_tool')!r}")
    if matched_tool_results and any(
        _is_tool_result_error(result) for result in matched_tool_results
    ):
        errors.append("tool result error")

    skipped_checks = ["intent", "handoff"]
    if case.get("requires_citations"):
        skipped_checks.append("citations")

    return {
        "id": case["id"],
        "passed": not errors,
        "errors": errors,
        "matched_transcript": matched_transcript,
        "similarity": similarity,
        "answer": answer,
        "tool_call_count": len(matched_tool_calls),
        "tool_result_count": len(matched_tool_results),
        "tool_latency_secs": [
            result.get("tool_latency_secs")
            for result in matched_tool_results
            if isinstance(result.get("tool_latency_secs"), int | float)
        ],
        "skipped_checks": skipped_checks,
    }


def _best_user_turn(expected: str, turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    expected_normalized = _normalize_text(expected)
    best: dict[str, Any] | None = None
    for index, turn in enumerate(turns):
        if turn["role"] != "user":
            continue
        actual_normalized = _normalize_text(turn["message"])
        if expected_normalized in actual_normalized or actual_normalized in expected_normalized:
            similarity = 1.0
        else:
            similarity = SequenceMatcher(None, expected_normalized, actual_normalized).ratio()
        candidate = {**turn, "index": index, "similarity": similarity}
        if best is None or similarity > best["similarity"]:
            best = candidate
    return best


def _next_agent_answer(turns: list[dict[str, Any]], user_index: int) -> str:
    responses = []
    for turn in turns[user_index + 1 :]:
        if turn["role"] == "user":
            break
        if turn["role"] == "agent":
            responses.append(turn["message"])
    return "\n".join(responses).strip()


def _entry_message(entry: dict[str, Any]) -> str:
    value = entry.get("message") or entry.get("text") or entry.get("content")
    return str(value or "").strip()


def _normalize_role(role: str) -> str | None:
    normalized = role.strip().lower()
    if normalized == "user":
        return "user"
    if normalized in {"agent", "assistant", "ai"}:
        return "agent"
    return None


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _case_transcript(case: dict[str, Any]) -> str:
    if case.get("transcript"):
        return str(case["transcript"])
    turns = case.get("turns") or []
    return str(turns[-1] if turns else "")


def _required_answer_substrings(case: dict[str, Any]) -> list[str]:
    values = case.get("required_substrings") or case.get("expected_answer_substrings") or []
    return [str(value) for value in values]


def _extract_nested_tool_items(conversation: dict[str, Any], key: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            nested = value.get(key)
            if isinstance(nested, list):
                found.extend(item for item in nested if isinstance(item, dict))
            for child in value.values():
                if isinstance(child, dict | list):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, dict | list):
                    visit(child)

    for root_key in ("transcript", "agent_responses", "test_runs", "test_info"):
        value = conversation.get(root_key)
        if isinstance(value, dict | list):
            visit(value)
    return found


def _matching_tool_calls(
    case: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected_tool = case.get("expected_tool")
    if not expected_tool:
        return tool_calls
    return [item for item in tool_calls if item.get("tool_name") == expected_tool]


def _matching_tool_results(
    case: dict[str, Any], tool_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected_tool = case.get("expected_tool")
    if not expected_tool:
        return tool_results
    return [item for item in tool_results if item.get("tool_name") == expected_tool]


def _is_tool_result_error(result: dict[str, Any]) -> bool:
    return bool(
        result.get("is_error") or result.get("error_type") or result.get("raw_error_message")
    )


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]
