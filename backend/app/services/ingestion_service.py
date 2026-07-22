from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from urllib.parse import urlparse

from fastapi import BackgroundTasks, HTTPException

from app.core.config import Settings
from app.crawling.jobs import CrawlJobStore
from app.crawling.firecrawl import FirecrawlClient, firecrawl_result_to_chunks, scrape_result_to_chunks
from app.knowledge.kb import IngestionSink
from app.crawling.link_discovery import LinkDiscoveryAgent, discovery_metadata_for
from app.api.schemas import CrawlJobResponse, CrawlRequest, IngestRequest
from app.core.sites import canonical_root_url, site_id_for_url

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        kb: IngestionSink,
        crawl_job_store: CrawlJobStore,
        firecrawl_client: FirecrawlClient | None,
    ) -> None:
        self.settings = settings
        self.kb = kb
        self.crawl_job_store = crawl_job_store
        self.firecrawl_client = firecrawl_client

    async def ingest_vinwonders(self, payload: IngestRequest | None = None) -> dict[str, int | str]:
        if not self.firecrawl_client:
            raise HTTPException(status_code=501, detail="FIRECRAWL_API_KEY is not configured")

        url = payload.url if payload and payload.url else self.settings.vinwonders_source_url
        site_id = site_id_for_url(url)
        chunks, summary = await self.collect_firecrawl_chunks(
            url=url,
            max_depth=2,
            max_pages=40,
            crawl_job_id=None,
            site_id=site_id,
            include_subdomains=False,
            exclude_patterns=[],
        )
        inserted = await self.kb.upsert_chunks(chunks)
        return {
            "source_url": url,
            "provider": "firecrawl",
            "pages_crawled": summary["pages_crawled"],
            "chunks_seen": len(chunks),
            "chunks_inserted": inserted,
        }

    async def crawl(
        self,
        payload: CrawlRequest,
        background_tasks: BackgroundTasks,
    ) -> CrawlJobResponse:
        if not self.firecrawl_client:
            raise HTTPException(status_code=501, detail="FIRECRAWL_API_KEY is not configured")
        job_id = str(uuid.uuid4())
        url = str(payload.url)
        site_id = site_id_for_url(url)
        initial = {
            "job_id": job_id,
            "status": "queued",
            "url": url,
            "site_id": site_id,
            "root_url": canonical_root_url(url),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "pages_crawled": 0,
            "pages_failed": 0,
            "pages_skipped": 0,
            "chunks_seen": 0,
            "chunks_inserted": 0,
            "errors": [],
            "discovered_urls": [],
            "candidate_urls": [],
            "selected_urls": [],
            "skipped_urls": [],
            "skip_reasons": {},
        }
        await self.crawl_job_store.create(
            job_id=job_id,
            url=url,
            payload=payload.model_dump(mode="json"),
            initial=initial,
        )
        background_tasks.add_task(self.run_crawl_job, job_id, payload)
        return CrawlJobResponse(job_id=job_id, status="queued", url=url, site_id=site_id)

    async def get_crawl_job(self, job_id: str) -> dict[str, object]:
        job = await self.crawl_job_store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        return job

    async def run_crawl_job(self, job_id: str, payload: CrawlRequest) -> None:
        await self.update_crawl_job(job_id, {"status": "running", "updated_at": utc_now()})
        try:
            url = str(payload.url)
            site_id = site_id_for_url(url)
            if hasattr(self.kb, "upsert_site"):
                await self.kb.upsert_site(  # type: ignore[attr-defined]
                    site_id=site_id,
                    root_url=canonical_root_url(url),
                    allowed_domains=[urlparse(url).netloc.lower()],
                    crawl_policy=payload.model_dump(mode="json"),
                )
            chunks, summary = await self.collect_firecrawl_chunks(
                url=url,
                max_depth=payload.max_depth,
                crawl_job_id=job_id,
                max_pages=payload.max_pages,
                site_id=site_id,
                include_subdomains=payload.include_subdomains,
                exclude_patterns=payload.exclude_patterns,
            )
            inserted = await self.kb.upsert_chunks(chunks)
            await self.update_crawl_job(
                job_id,
                {
                    "status": "completed",
                    "updated_at": utc_now(),
                    "provider": "firecrawl",
                    "site_id": site_id,
                    "root_url": canonical_root_url(url),
                    "firecrawl_id": summary["firecrawl_id"],
                    "started_at": summary["started_at"],
                    "finished_at": summary["finished_at"],
                    "pages_seen": summary["pages_seen"],
                    "pages_crawled": summary["pages_crawled"],
                    "pages_failed": summary["pages_failed"],
                    "pages_skipped": 0,
                    "credits_used": summary["credits_used"],
                    "chunks_seen": len(chunks),
                    "chunks_inserted": inserted,
                    "errors": summary["errors"],
                    "discovered_urls": summary["discovered_urls"],
                    "candidate_urls": summary["candidate_urls"],
                    "selected_urls": summary["selected_urls"],
                    "skipped_urls": summary["skipped_urls"],
                    "skip_reasons": summary["skip_reasons"],
                },
            )
        except Exception as exc:
            logger.exception("Crawl job failed: %s", job_id)
            await self.update_crawl_job(
                job_id,
                {"status": "failed", "updated_at": utc_now(), "error": str(exc)},
            )

    async def collect_firecrawl_chunks(
        self,
        *,
        url: str,
        max_depth: int,
        max_pages: int,
        crawl_job_id: str | None,
        site_id: str | None,
        include_subdomains: bool,
        exclude_patterns: list[str],
    ) -> tuple[list, dict[str, object]]:
        if not self.firecrawl_client:
            raise HTTPException(status_code=501, detail="FIRECRAWL_API_KEY is not configured")

        discovery_budget = reserved_discovery_budget(max_pages)
        initial_crawl_limit = max(1, max_pages - discovery_budget)
        firecrawl_job = await self.firecrawl_client.start_crawl(
            url=url,
            max_depth=max_depth,
            limit=initial_crawl_limit,
            include_paths=default_include_paths(url),
            exclude_paths=default_exclude_paths(exclude_patterns),
            allow_subdomains=include_subdomains,
        )
        if crawl_job_id:
            await self.update_crawl_job(
                crawl_job_id,
                {
                    "status": "waiting_firecrawl",
                    "updated_at": utc_now(),
                    "firecrawl_id": str(firecrawl_job["id"]),
                },
            )
        firecrawl_result = await self.firecrawl_client.wait_for_crawl(
            str(firecrawl_job["id"]),
            timeout_seconds=240.0,
        )
        map_urls: list[str] = []
        map_error: str | None = None
        try:
            map_result = await self.firecrawl_client.map(
                url=url,
                limit=map_discovery_limit(max_pages),
                include_subdomains=include_subdomains,
            )
            map_urls = map_result_urls(map_result)
        except Exception as exc:
            map_error = str(exc)
            logger.warning("Firecrawl map discovery failed for %s: %s", url, exc)
        if crawl_job_id:
            await self.update_crawl_job(
                crawl_job_id,
                {
                    "status": "processing_firecrawl_result",
                    "updated_at": utc_now(),
                    "pages_seen": firecrawl_result.get("total", 0),
                    "pages_crawled": int(firecrawl_result.get("completed") or 0),
                    "credits_used": int(firecrawl_result.get("creditsUsed") or 0),
                    "map_url_count": len(map_urls),
                    "map_error": map_error,
                },
            )
        discovered_urls = [
            item.get("metadata", {}).get("sourceURL") or item.get("metadata", {}).get("url")
            for item in firecrawl_result.get("data", [])
            if item.get("metadata")
        ]
        allowed_domains = [urlparse(url).netloc.lower()]
        discovery_agent = LinkDiscoveryAgent(
            seed_url=url,
            allowed_domains=allowed_domains,
            include_subdomains=include_subdomains,
            exclude_patterns=exclude_patterns,
        )

        async def on_discovery_progress(update: dict[str, object]) -> None:
            if not crawl_job_id:
                return
            await self.update_crawl_job(
                crawl_job_id,
                {
                    "status": "scraping_discovered_urls",
                    "updated_at": utc_now(),
                    **update,
                },
            )

        initial_completed = int(firecrawl_result.get("completed") or 0)
        discovery, scraped_pages = await discovery_agent.discover_and_scrape(
            firecrawl_result,
            self.firecrawl_client,
            max_scrapes=max(0, max_pages - initial_completed),
            extra_urls=map_urls,
            on_progress=on_discovery_progress,
        )
        selected_urls = [item.url for item in discovery.selected_urls]
        selected_metadata = {
            discovered.url: discovery_metadata_for(discovered) for discovered in discovery.selected_urls
        }
        chunks = firecrawl_result_to_chunks(
            result=firecrawl_result,
            seed_url=url,
            crawl_job_id=crawl_job_id,
            site_id=site_id,
            url_metadata=selected_metadata,
        )

        for page in scraped_pages:
            scrape_url = page.discovered_url.url
            scrape_chunks = scrape_result_to_chunks(
                result=page.result,
                seed_url=url,
                crawl_job_id=crawl_job_id,
                site_id=site_id,
                url_metadata={scrape_url: discovery_metadata_for(page.discovered_url)},
            )
            if scrape_chunks:
                chunks.extend(scrape_chunks)
                discovered_urls.append(scrape_url)
                if crawl_job_id:
                    await self.update_crawl_job(
                        crawl_job_id,
                        {
                            "updated_at": utc_now(),
                            "pages_crawled": initial_completed + len(discovery.scraped_urls),
                            "chunks_seen": len(chunks),
                            "scraped_count": len(discovery.scraped_urls),
                        },
                    )

        summary: dict[str, object] = {
            "firecrawl_id": str(firecrawl_job["id"]),
            "started_at": firecrawl_result.get("createdAt"),
            "finished_at": firecrawl_result.get("completedAt"),
            "pages_seen": firecrawl_result.get("total", 0),
            "pages_crawled": initial_completed + len(discovery.scraped_urls),
            "pages_failed": len(discovery.scrape_errors),
            "credits_used": int(firecrawl_result.get("creditsUsed") or 0) + len(discovery.scraped_urls),
            "errors": discovery.scrape_errors,
            "discovered_urls": discovered_urls,
            "candidate_urls": discovery.candidate_urls,
            "map_url_count": len(map_urls),
            "map_error": map_error,
            "selected_urls": selected_urls,
            "skipped_urls": discovery.skipped_urls,
            "skip_reasons": discovery.skip_reasons,
        }
        return chunks, summary

    async def update_crawl_job(self, job_id: str, updates: dict[str, object]) -> None:
        await self.crawl_job_store.update(job_id, updates)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_include_paths(url: str) -> list[str]:
    lowered = url.lower()
    if "vinwonders.com" not in lowered:
        return []
    language = "/en/" if "/en/" in lowered else "/vi/"
    return [
        f".*{language}vinpearl-safari-phu-quoc.*",
        f".*{language}wonderpedia.*",
        f".*{language}promotions.*",
        f".*{language}uu-dai.*",
    ]


def reserved_discovery_budget(max_pages: int) -> int:
    if max_pages <= 4:
        return 0
    return min(12, max(2, max_pages // 4))


def map_discovery_limit(max_pages: int) -> int:
    return min(5000, max(100, max_pages * 20))


def map_result_urls(result: dict[str, object]) -> list[str]:
    urls: list[str] = []
    for item in result.get("links") or []:
        if isinstance(item, dict):
            url = item.get("url")
        else:
            url = item
        if url:
            urls.append(str(url))
    return urls


def default_exclude_paths(extra_patterns: list[str] | None = None) -> list[str]:
    return [
        ".*\\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|mp4|mov|zip)$",
        ".*login.*",
        ".*register.*",
        ".*dang-nhap.*",
        ".*dang-ky.*",
        ".*checkout.*",
        ".*cart.*",
    ] + list(extra_patterns or [])
