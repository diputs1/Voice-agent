import pytest

from app.crawl_jobs import InMemoryCrawlJobStore


@pytest.mark.asyncio
async def test_in_memory_crawl_job_store_create_get_update():
    store = InMemoryCrawlJobStore()
    await store.ensure_ready()

    await store.create(
        job_id="job-1",
        url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
        payload={"max_depth": 2},
        initial={"status": "queued", "pages_crawled": 0},
    )
    await store.update("job-1", {"status": "running", "pages_crawled": 1})

    job = await store.get("job-1")

    assert job["job_id"] == "job-1"
    assert job["status"] == "running"
    assert job["pages_crawled"] == 1
    assert job["url"] == "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/"
    assert job["created_at"]
    assert job["updated_at"]
