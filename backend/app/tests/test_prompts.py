import pytest

from app.prompts import _parse_prompt_text, load_prompt


def test_load_prompt_reads_version_and_body():
    prompt = load_prompt("voice_answer")

    assert prompt.version.startswith("voice_answer@")
    assert "Chỉ trả lời dựa trên context" in prompt.template


def test_missing_prompt_file_fails_clearly():
    with pytest.raises(FileNotFoundError, match="Prompt file not found"):
        load_prompt("missing_prompt")


def test_prompt_missing_version_header_fails_clearly():
    with pytest.raises(ValueError, match="missing a prompt_version header"):
        _parse_prompt_text("voice_answer", "Body without a header")
