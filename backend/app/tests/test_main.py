import pytest
from fastapi import HTTPException

from app import main


@pytest.mark.asyncio
async def test_health_reports_ok_for_primary_kb():
    main.app.state.kb_status = {"provider": "postgres", "fallback": False, "fallback_reason": None}

    response = await main.health()

    assert response["status"] == "ok"
    assert response["kb"]["provider"] == "postgres"
    assert response["kb"]["fallback"] is False


@pytest.mark.asyncio
async def test_health_reports_degraded_for_kb_fallback():
    main.app.state.kb_status = {
        "provider": "memory",
        "fallback": True,
        "fallback_reason": "connection failed",
    }

    response = await main.health()

    assert response["status"] == "degraded"
    assert response["kb"]["provider"] == "memory"
    assert response["kb"]["fallback"] is True


@pytest.mark.asyncio
async def test_admin_auth_allows_dev_when_key_is_unset(monkeypatch):
    monkeypatch.setattr(main.settings, "app_env", "development")
    monkeypatch.setattr(main.settings, "admin_api_key", None)

    await main.require_admin_api_key()


@pytest.mark.asyncio
async def test_admin_auth_rejects_missing_or_wrong_key_when_configured(monkeypatch):
    monkeypatch.setattr(main.settings, "app_env", "development")
    monkeypatch.setattr(main.settings, "admin_api_key", "secret-admin-key")

    with pytest.raises(HTTPException) as missing:
        await main.require_admin_api_key()
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        await main.require_admin_api_key("wrong")
    assert wrong.value.status_code == 401

    await main.require_admin_api_key("secret-admin-key")


@pytest.mark.asyncio
async def test_admin_auth_fails_closed_outside_dev_when_key_is_unset(monkeypatch):
    monkeypatch.setattr(main.settings, "app_env", "production")
    monkeypatch.setattr(main.settings, "admin_api_key", None)

    with pytest.raises(HTTPException) as exc:
        await main.require_admin_api_key()

    assert exc.value.status_code == 500
    assert "ADMIN_API_KEY" in exc.value.detail
