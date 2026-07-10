import os

from app.core.config import Settings
from app.core.observability import configure_langsmith


def test_configure_langsmith_exports_settings_to_environment(monkeypatch):
    for key in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)

    status = configure_langsmith(
        Settings(
            langsmith_tracing=True,
            langsmith_api_key="ls-test",
            langsmith_project="vin-agent-test",
        )
    )

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-test"
    assert os.environ["LANGSMITH_PROJECT"] == "vin-agent-test"
    assert status["enabled"] is True
    assert status["api_key_configured"] is True
