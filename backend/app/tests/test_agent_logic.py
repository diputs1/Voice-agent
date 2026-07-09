from app.agents.logic.intent import (
    parse_supervisor_payload,
    primary_intent_from_rules,
    route_from_decision,
)
from app.agents.logic.query_rewrite import heuristic_rewrite, looks_like_follow_up
from app.agents.nodes.retry_query import expanded_search_query


def test_parse_supervisor_payload_accepts_legacy_intent_shape():
    decision = parse_supervisor_payload('{"intent":"small_talk","confidence":0.8}')

    assert decision.primary_intent == "small_talk"
    assert decision.legacy_intent == "small_talk"
    assert route_from_decision(decision) == "direct_response"


def test_rule_primary_intent_detects_reptile_question():
    assert primary_intent_from_rules("có khu bò sát không?") == "animal_info"


def test_heuristic_rewrite_uses_latest_thread_turn_for_follow_up():
    history = [{"transcript": "Tôi hỏi về khu tham quan Safari", "answer": "Có khu bò sát."}]

    assert looks_like_follow_up("thế còn khu bò sát thì sao?")
    assert heuristic_rewrite("thế còn khu bò sát thì sao?", history) == (
        "Tôi hỏi về khu tham quan Safari. Câu hỏi tiếp theo: thế còn khu bò sát thì sao?"
    )


def test_expanded_search_query_uses_primary_intent_terms():
    query = expanded_search_query(
        {
            "search_query": "Có khu bò sát không?",
            "transcript": "Có khu bò sát không?",
            "primary_intent": "animal_info",
            "intent_entities": [],
        }
    )

    assert "Vinpearl Safari Phú Quốc" in query
    assert "động vật loài thú khu tham quan" in query
