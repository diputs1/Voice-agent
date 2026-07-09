import pytest
from fastapi import BackgroundTasks
from fastapi import HTTPException

from app import main
from app.schemas import CrawlRequest


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


@pytest.mark.asyncio
async def test_crawl_creates_queued_job_in_store(monkeypatch):
    store = FakeCrawlJobStore()
    main.app.state.crawl_job_store = store
    main.app.state.firecrawl_client = object()

    response = await main.crawl(
        CrawlRequest(url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/"),
        BackgroundTasks(),
    )

    assert response.status == "queued"
    assert response.job_id in store.jobs
    assert store.jobs[response.job_id]["status"] == "queued"
    assert store.jobs[response.job_id]["url"] == "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/"


@pytest.mark.asyncio
async def test_get_crawl_job_returns_404_for_missing_store_job():
    main.app.state.crawl_job_store = FakeCrawlJobStore()

    with pytest.raises(HTTPException) as exc:
        await main.get_crawl_job("missing")

    assert exc.value.status_code == 404


class FakeCrawlJobStore:
    def __init__(self) -> None:
        self.jobs = {}

    async def create(self, *, job_id, url, payload, initial):
        del payload
        self.jobs[job_id] = {**initial, "job_id": job_id, "url": url}

    async def update(self, job_id, updates):
        self.jobs[job_id].update(updates)

    async def get(self, job_id):
        return self.jobs.get(job_id)
