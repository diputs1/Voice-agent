from __future__ import annotations

import json
import re
from typing import Any

from app.agents.logic.intent import DOMAIN_ANCHORS

FOLLOW_UP_MARKERS = {
    "thế còn",
    "vậy còn",
    "còn ",
    "thì sao",
    "nó",
    "đó",
    "cái này",
    "cái đó",
    "ở đó",
    "chỗ đó",
    "như vậy",
    "trẻ em",
    "người lớn",
    "người già",
    "vé đó",
    "show đó",
    "dịch vụ đó",
}


def parse_rewrite_payload(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        payload = json.loads(stripped)
        return str(payload.get("search_query") or "").strip()
    except json.JSONDecodeError:
        return stripped


def looks_like_follow_up(transcript: str) -> bool:
    lowered = transcript.lower().strip()
    if any(marker in lowered for marker in FOLLOW_UP_MARKERS):
        return True
    if "?" in lowered and len(lowered.split()) <= 8 and not any(
        anchor in lowered for anchor in DOMAIN_ANCHORS
    ):
        return True
    return False


def history_context(history: list[dict[str, Any]]) -> str:
    turns = history[-3:]
    lines = []
    for idx, turn in enumerate(turns, start=1):
        transcript = str(turn.get("transcript") or "").strip()
        answer = str(turn.get("answer") or "").strip()
        if transcript:
            lines.append(f"Lượt {idx} - người dùng: {transcript}")
        if answer:
            lines.append(f"Lượt {idx} - trợ lý: {answer[:500]}")
    return "\n".join(lines) or "Không có lịch sử."


def heuristic_rewrite(transcript: str, history: list[dict[str, Any]]) -> str:
    previous = latest_transcript(history)
    if not previous:
        return transcript
    if any(anchor in transcript.lower() for anchor in DOMAIN_ANCHORS):
        return transcript
    return f"{previous}. Câu hỏi tiếp theo: {transcript}"


def latest_transcript(history: list[dict[str, Any]]) -> str:
    for turn in reversed(history):
        transcript = str(turn.get("transcript") or "").strip()
        if transcript:
            return transcript
    return ""
