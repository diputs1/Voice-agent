from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request

from app.api.schemas import CrawlJobResponse, CrawlRequest, IngestRequest
from app.services.ingestion_service import IngestionService


async def require_admin_api_key(
    request: Request,
    x_admin_api_key: str | None = Header(default=None),
) -> None:
    settings = request.app.state.settings
    configured_key = settings.admin_api_key
    if configured_key:
        if x_admin_api_key != configured_key:
            raise HTTPException(status_code=401, detail="Invalid admin API key")
        return

    if settings.app_env.lower() not in {"development", "dev", "test", "testing", "local"}:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY is not configured")


router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_api_key)])


@router.post("/ingest-vinwonders")
async def ingest_vinwonders(
    request: Request,
    payload: IngestRequest | None = None,
) -> dict[str, int | str]:
    service: IngestionService = request.app.state.ingestion_service
    return await service.ingest_vinwonders(payload)


@router.post("/crawl", response_model=CrawlJobResponse)
async def crawl(
    request: Request,
    payload: CrawlRequest,
    background_tasks: BackgroundTasks,
) -> CrawlJobResponse:
    service: IngestionService = request.app.state.ingestion_service
    return await service.crawl(payload, background_tasks)


@router.get("/crawl-jobs/{job_id}")
async def get_crawl_job(request: Request, job_id: str) -> dict[str, object]:
    service: IngestionService = request.app.state.ingestion_service
    return await service.get_crawl_job(job_id)


@router.get("/sites/{site_id}/status")
async def get_site_status(request: Request, site_id: str) -> dict[str, object]:
    kb = request.app.state.kb
    if not hasattr(kb, "site_status"):
        return {"site_id": site_id, "document_count": None, "category_counts": {}}
    return await kb.site_status(site_id)


@router.get("/kb/status")
async def get_kb_status(request: Request) -> dict[str, object]:
    kb = request.app.state.kb
    base = dict(getattr(request.app.state, "kb_status", {}))
    if hasattr(kb, "site_status"):
        base.update(await kb.site_status(None))
    return base
