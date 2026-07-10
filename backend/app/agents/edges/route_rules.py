from __future__ import annotations

from typing import Literal

from app.agents.state import AgentState


def route_after_supervisor(state: AgentState) -> Literal["direct", "handoff", "knowledge"]:
    if state.get("route") == "direct_response":
        return "direct"
    if state.get("route") == "direct_handoff":
        return "handoff"
    return "knowledge"
