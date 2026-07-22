import pytest

from app.api.schemas import VoiceAgentKnowledgeRequest
from app.evals.elevenlabs_tests import build_test_manifest, normalize_run_tests_response
from app.evals.voice_conversation import (
    evaluate_voice_conversation,
    extract_tool_calls,
    extract_tool_results,
)
from app.evals.voice_tool_harness import load_voice_cases, run_local_voice_harness


def test_voice_agent_default_site_id_is_unscoped():
    payload = VoiceAgentKnowledgeRequest.model_validate(
        {"parameters": {"query": "giờ mở cửa", "site_id": " default ", "limit": 3}}
    )

    assert payload.site_id is None


@pytest.mark.asyncio
async def test_local_voice_harness_runs_seed_cases():
    cases = [
        case
        for case in load_voice_cases()
        if case["id"] in {"schedule_basic", "follow_up_price_child", "small_talk_no_tool"}
    ]

    report = await run_local_voice_harness(cases)

    assert report["passed"] is True
    assert report["case_count"] == 3
    assert report["executable_case_count"] == 2
    follow_up = next(item for item in report["results"] if item["id"] == "follow_up_price_child")
    assert follow_up["rewrite_source"] == "voice_memory_heuristic"
    assert "còn trẻ em thì sao?" in follow_up["resolved_query"]


def test_elevenlabs_manifest_covers_tool_reply_and_simulation_tests():
    cases = [
        {
            "id": "follow_up",
            "turns": ["Giá vé người lớn?", "còn trẻ em thì sao?"],
            "requires_tool_call": True,
            "expected_tool": "search_vinpearl_safari_knowledge",
            "expected_answer_substrings": ["kiểm tra"],
            "forbidden_answer_substrings": ["chắc chắn là"],
            "time_sensitive": True,
        }
    ]

    manifest = build_test_manifest(cases)

    assert {test["type"] for test in manifest["tests"]} == {"tool", "llm", "simulation"}
    assert "search_vinpearl_safari_knowledge" in manifest["tests"][0]["expected_tool"]
    assert "không tự bịa" not in manifest["tests"][1]["success_condition"]


def test_normalize_elevenlabs_run_tests_response_extracts_tool_results():
    raw = {
        "id": "invocation-1",
        "agent_id": "agent-1",
        "repeat_count": 1,
        "test_runs": [
            {
                "test_run_id": "run-1",
                "test_id": "test-1",
                "test_name": "tool call",
                "status": "done",
                "condition_result": {"result": "success"},
                "agent_responses": [
                    {
                        "role": "agent",
                        "message": "Safari mở cửa từ 09:00 đến 16:00.",
                        "tool_calls": [
                            {
                                "tool_name": "search_vinpearl_safari_knowledge",
                                "params_as_json": '{"query":"giờ mở cửa"}',
                            }
                        ],
                        "tool_results": [
                            {
                                "tool_name": "search_vinpearl_safari_knowledge",
                                "tool_latency_secs": 0.4,
                                "is_error": False,
                            }
                        ],
                    }
                ],
            }
        ],
    }

    report = normalize_run_tests_response(raw)

    assert report["passed"] is True
    assert report["tool_latency_p95_secs"] == 0.4
    assert report["results"][0]["tool_calls"][0]["tool_name"] == "search_vinpearl_safari_knowledge"


def test_voice_conversation_eval_checks_required_tool_call():
    conversation = {
        "conversation_id": "conv-1",
        "transcript": [
            {
                "role": "user",
                "message": "Vinpearl Safari mở cửa mấy giờ?",
                "tool_calls": [{"tool_name": "search_vinpearl_safari_knowledge"}],
                "tool_results": [
                    {
                        "tool_name": "search_vinpearl_safari_knowledge",
                        "tool_latency_secs": 0.2,
                        "is_error": False,
                    }
                ],
            },
            {"role": "agent", "message": "Safari mở cửa từ 09:00 đến 16:00."},
        ],
    }
    cases = [
        {
            "id": "schedule_basic",
            "transcript": "Vinpearl Safari mở cửa mấy giờ?",
            "requires_tool_call": True,
            "expected_tool": "search_vinpearl_safari_knowledge",
            "required_substrings": ["09:00", "16:00"],
        }
    ]

    report = evaluate_voice_conversation(conversation, cases)

    assert extract_tool_calls(conversation)
    assert extract_tool_results(conversation)
    assert report["passed"] is True
    assert report["tool_call_count"] == 1
    assert report["results"][0]["tool_call_count"] == 1
