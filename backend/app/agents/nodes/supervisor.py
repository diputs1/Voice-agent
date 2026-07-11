from __future__ import annotations

from langsmith import traceable

from app.agents.logic.intent import (
    LLM_CLASSIFIER_HINTS,
    OUT_OF_SCOPE_KEYWORDS,
    TIME_SENSITIVE_KEYWORDS,
    coerce_supervisor_decision,
    decision,
    entities_from_rules,
    is_out_of_scope,
    is_small_talk,
    matched_keywords,
    parse_supervisor_payload,
    primary_intent_from_rules,
    route_from_decision,
    supervisor_messages,
)
from app.agents.schemas import SafariState, SupervisorDecision
from app.agents.state import AgentUpdate


@traceable(name="graph.supervisor")
async def supervisor_node(state: SafariState, llm) -> AgentUpdate:
    transcript = state.get("transcript", "")
    classifier_decision, source = await classify_intent(transcript, llm)
    route = route_from_decision(classifier_decision)
    return {
        "intent": classifier_decision.legacy_intent,
        "primary_intent": classifier_decision.primary_intent,
        "intent_entities": classifier_decision.entities,
        "supervisor_confidence": classifier_decision.confidence,
        "suggested_route": classifier_decision.suggested_route,
        "classifier_confidence": classifier_decision.confidence,
        "classifier_source": source,
        "messages": [{"role": "user", "content": transcript}],
        "route": route,
    }


@traceable(name="graph.classify_intent")
async def classify_intent(transcript: str, llm) -> tuple[SupervisorDecision, str]:
    lowered = transcript.lower()
    if is_small_talk(lowered):
        return decision("small_talk", False, ["greeting"], 1.0, "direct_response"), "rules"
    if is_out_of_scope(lowered):
        return (
            decision(
                "out_of_scope",
                False,
                matched_keywords(lowered, OUT_OF_SCOPE_KEYWORDS),
                1.0,
                "direct_response",
            ),
            "rules",
        )
    if any(k in lowered for k in TIME_SENSITIVE_KEYWORDS):
        primary_intent = primary_intent_from_rules(lowered)
        return (
            decision(
                primary_intent,
                True,
                matched_keywords(lowered, TIME_SENSITIVE_KEYWORDS),
                1.0,
                "safari_knowledge",
            ),
            "rules",
        )
    if not any(k in lowered for k in LLM_CLASSIFIER_HINTS):
        primary_intent = primary_intent_from_rules(lowered)
        return (
            decision(
                primary_intent,
                False,
                entities_from_rules(lowered),
                1.0,
                "safari_knowledge",
            ),
            "rules",
        )
    if not llm:
        return (
            decision("general", False, entities_from_rules(lowered), 0.0, "safari_knowledge"),
            "rules_fallback",
        )

    try:
        structured_llm = llm.with_structured_output(SupervisorDecision)
        response = await structured_llm.ainvoke(supervisor_messages(transcript))
        return coerce_supervisor_decision(response), "llm_structured"
    except Exception:
        pass

    try:
        response = await llm.ainvoke(supervisor_messages(transcript))
        return parse_supervisor_payload(str(response.content)), "llm_json"
    except Exception:
        return (
            decision("general", False, entities_from_rules(lowered), 0.0, "safari_knowledge"),
            "llm_failed",
        )
