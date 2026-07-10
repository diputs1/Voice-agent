from __future__ import annotations

from app.agents.logic.answer_grounding import is_grounded_quantitative_answer
from app.agents.state import AgentState


def evaluate_grounding(state: AgentState) -> dict:
    answer = state.get("answer", "")
    context = state.get("retrieved_context", [])
    grounded = is_grounded_quantitative_answer(answer, context)
    return {
        "grounded": grounded,
        "evaluator_score": 1.0 if grounded else 0.0,
    }
