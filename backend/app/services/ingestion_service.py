from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import BackgroundTasks, HTTPException

from app.config import Settings
from app.crawl_jobs import CrawlJobStore
from app.firecrawl import FirecrawlClient, firecrawl_result_to_chunks, scrape_result_to_chunks
from app.ingestion import fetch_vinwonders_chunks
from app.kb import IngestionSink
from app.link_discovery import LinkDiscoveryAgent, discovery_metadata_for, normalize_url
from app.schemas import CrawlJobResponse, CrawlRequest, IngestRequest

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
        url = payload.url if payload and payload.url else self.settings.vinwonders_source_url
        if self.firecrawl_client:
            chunks, summary = await self.collect_firecrawl_chunks(
                url=url,
                max_depth=2,
                max_pages=40,
                crawl_job_id=None,
            )
            inserted = await self.kb.upsert_chunks(chunks)
            return {
                "source_url": url,
                "provider": "firecrawl",
                "pages_crawled": summary["pages_crawled"],
                "chunks_seen": len(chunks),
                "chunks_inserted": inserted,
            }

        chunks = await fetch_vinwonders_chunks(url)
        inserted = await self.kb.upsert_chunks(chunks)
        return {
            "source_url": url,
            "provider": "fallback_http",
            "warning": "FIRECRAWL_API_KEY is not configured; used one-page fallback ingestion",
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
        initial = {
            "job_id": job_id,
            "status": "queued",
            "url": str(payload.url),
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
            url=str(payload.url),
            payload=payload.model_dump(mode="json"),
            initial=initial,
        )
        background_tasks.add_task(self.run_crawl_job, job_id, payload)
        return CrawlJobResponse(job_id=job_id, status="queued", url=str(payload.url))

    async def get_crawl_job(self, job_id: str) -> dict[str, object]:
        job = await self.crawl_job_store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        return job

    async def run_crawl_job(self, job_id: str, payload: CrawlRequest) -> None:
        await self.update_crawl_job(job_id, {"status": "running", "updated_at": utc_now()})
        try:
            chunks, summary = await self.collect_firecrawl_chunks(
                url=str(payload.url),
                max_depth=payload.max_depth,
                crawl_job_id=job_id,
                max_pages=payload.max_pages,
            )
            inserted = await self.kb.upsert_chunks(chunks)
            await self.update_crawl_job(
                job_id,
                {
                    "status": "completed",
                    "updated_at": utc_now(),
                    "provider": "firecrawl",
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
    ) -> tuple[list, dict[str, object]]:
        if not self.firecrawl_client:
            raise HTTPException(status_code=501, detail="FIRECRAWL_API_KEY is not configured")

        firecrawl_job = await self.firecrawl_client.start_crawl(
            url=url,
            max_depth=max_depth,
            limit=max_pages,
            include_paths=default_include_paths(url),
            exclude_paths=default_exclude_paths(),
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
        if crawl_job_id:
            await self.update_crawl_job(
                crawl_job_id,
                {
                    "status": "processing_firecrawl_result",
                    "updated_at": utc_now(),
                    "pages_seen": firecrawl_result.get("total", 0),
                    "pages_crawled": int(firecrawl_result.get("completed") or 0),
                    "credits_used": int(firecrawl_result.get("creditsUsed") or 0),
                },
            )
        discovered_urls = [
            item.get("metadata", {}).get("sourceURL") or item.get("metadata", {}).get("url")
            for item in firecrawl_result.get("data", [])
            if item.get("metadata")
        ]
        discovered_set = {str(item) for item in discovered_urls if item}
        normalized_discovered_set = {
            normalized for item in discovered_set if (normalized := normalize_url(item))
        }
        errors: list[dict[str, str]] = []
        scraped_count = 0
        discovery = LinkDiscoveryAgent(
            seed_url=url,
            fallback_urls=strategic_firecrawl_urls(url),
        ).discover_from_firecrawl_result(firecrawl_result)
        selected_urls = [item.url for item in discovery.selected_urls]
        selected_metadata = {
            discovered.url: discovery_metadata_for(discovered) for discovered in discovery.selected_urls
        }
        chunks = firecrawl_result_to_chunks(
            result=firecrawl_result,
            seed_url=url,
            crawl_job_id=crawl_job_id,
            url_metadata=selected_metadata,
        )

        for discovered in discovery.selected_urls:
            scrape_url = discovered.url
            if scrape_url in normalized_discovered_set:
                continue
            try:
                if crawl_job_id:
                    await self.update_crawl_job(
                        crawl_job_id,
                        {
                            "status": "scraping_discovered_urls",
                            "updated_at": utc_now(),
                            "current_scrape_url": scrape_url,
                            "selected_urls": selected_urls,
                            "candidate_urls": discovery.candidate_urls,
                            "skipped_urls": discovery.skipped_urls,
                            "skip_reasons": discovery.skip_reasons,
                            "scraped_count": scraped_count,
                        },
                    )
                scrape_result = await self.firecrawl_client.scrape(scrape_url)
                scrape_chunks = scrape_result_to_chunks(
                    result=scrape_result,
                    seed_url=url,
                    crawl_job_id=crawl_job_id,
                    url_metadata={scrape_url: selected_metadata[scrape_url]},
                )
                if scrape_chunks:
                    chunks.extend(scrape_chunks)
                    discovered_urls.append(scrape_url)
                    scraped_count += 1
                    if crawl_job_id:
                        await self.update_crawl_job(
                            crawl_job_id,
                            {
                                "updated_at": utc_now(),
                                "pages_crawled": int(firecrawl_result.get("completed") or 0)
                                + scraped_count,
                                "chunks_seen": len(chunks),
                                "scraped_count": scraped_count,
                            },
                        )
            except Exception as exc:
                errors.append({"url": scrape_url, "error": str(exc)})
                if crawl_job_id:
                    await self.update_crawl_job(
                        crawl_job_id,
                        {
                            "updated_at": utc_now(),
                            "pages_failed": len(errors),
                            "errors": errors,
                        },
                    )

        summary: dict[str, object] = {
            "firecrawl_id": str(firecrawl_job["id"]),
            "started_at": firecrawl_result.get("createdAt"),
            "finished_at": firecrawl_result.get("completedAt"),
            "pages_seen": firecrawl_result.get("total", 0),
            "pages_crawled": int(firecrawl_result.get("completed") or 0) + scraped_count,
            "pages_failed": len(errors),
            "credits_used": int(firecrawl_result.get("creditsUsed") or 0) + scraped_count,
            "errors": errors,
            "discovered_urls": discovered_urls,
            "candidate_urls": discovery.candidate_urls,
            "selected_urls": selected_urls,
            "skipped_urls": discovery.skipped_urls,
            "skip_reasons": discovery.skip_reasons,
        }
        return chunks, summary

    async def update_crawl_job(self, job_id: str, updates: dict[str, object]) -> None:
        await self.crawl_job_store.update(job_id, updates)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def strategic_firecrawl_urls(url: str) -> list[str]:
    if "vinwonders.com" not in url:
        return []
    return [
        "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
        "https://vinwonders.com/vi/vinpearl-safari-phu-quoc-gia-ve-va-quy-dinh/",
        "https://vinwonders.com/vi/promotions/",
        "https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub/",
        "https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/",
        "https://vinwonders.com/vi/wonderpedia/news/gia-ve-vinpearl-safari-phu-quoc/",
        "https://vinwonders.com/vi/wonderpedia/news/voucher-vinpearl-safari-phu-quoc/",
        "https://vinwonders.com/en/vinpearl-safari-phu-quoc/",
        "https://vinwonders.com/en/vinpearl-safari-phu-quoc-price-and-regulations/",
        "https://vinwonders.com/en/promotions/",
        "https://vinwonders.com/en/terms-and-conditions/",
    ]


def default_include_paths(url: str) -> list[str]:
    if "/vi/" not in url and "/en/" not in url:
        return []
    return [
        "vi/.*vinpearl-safari-phu-quoc.*",
        "en/.*vinpearl-safari-phu-quoc.*",
        "vi/promotions.*",
        "en/promotions.*",
        "vi/uu-dai/.*",
        "en/uu-dai/.*",
        "vi/wonderpedia/news/.*(vinpearl-safari-phu-quoc|voucher|combo|affiliate|vinclub|gia-ve).*",
        "en/wonderpedia/news/.*(vinpearl-safari-phu-quoc|voucher|combo|affiliate|vinclub|ticket|price).*",
    ]


def default_exclude_paths() -> list[str]:
    return [
        ".*\\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|mp4|mov|zip)$",
        ".*/(ko|zh|ru)/.*",
        ".*login.*",
    ]
