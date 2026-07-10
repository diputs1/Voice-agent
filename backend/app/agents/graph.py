from __future__ import annotations

from app.agents.graphs.website_graph import WebsiteAgentGraph
from app.agents.schemas import SafariState, SupervisorDecision


class SafariAgentGraph(WebsiteAgentGraph):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("legacy_node_names", True)
        super().__init__(*args, **kwargs)


__all__ = ["SafariAgentGraph", "WebsiteAgentGraph", "SafariState", "SupervisorDecision"]
