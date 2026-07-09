from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row


class CrawlJobStore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def create(
        self,
        *,
        job_id: str,
        url: str,
        payload: dict[str, Any],
        initial: dict[str, Any],
    ) -> None: ...

    async def update(self, job_id: str, updates: dict[str, Any]) -> None: ...

    async def get(self, job_id: str) -> dict[str, Any] | None: ...


class InMemoryCrawlJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    async def ensure_ready(self) -> None:
        return None

    async def create(
        self,
        *,
        job_id: str,
        url: str,
        payload: dict[str, Any],
        initial: dict[str, Any],
    ) -> None:
        del payload
        now = _utc_now()
        self.jobs[job_id] = {
            **initial,
            "job_id": job_id,
            "url": url,
            "created_at": initial.get("created_at") or now,
            "updated_at": initial.get("updated_at") or now,
        }

    async def update(self, job_id: str, updates: dict[str, Any]) -> None:
        if job_id not in self.jobs:
            return
        self.jobs[job_id].update({**updates, "updated_at": updates.get("updated_at") or _utc_now()})

    async def get(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        return dict(job) if job else None


class PostgresCrawlJobStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    async def ensure_ready(self) -> None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS crawl_jobs (
                  job_id TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  url TEXT NOT NULL,
                  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                  result JSONB NOT NULL DEFAULT '{}'::jsonb,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS crawl_jobs_status_idx ON crawl_jobs (status, updated_at DESC)"
            )

    async def create(
        self,
        *,
        job_id: str,
        url: str,
        payload: dict[str, Any],
        initial: dict[str, Any],
    ) -> None:
        status = str(initial.get("status") or "queued")
        result = {key: value for key, value in initial.items() if key not in {"job_id", "status", "url"}}
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO crawl_jobs (job_id, status, url, payload, result)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (job_id) DO UPDATE SET
                  status = EXCLUDED.status,
                  url = EXCLUDED.url,
                  payload = EXCLUDED.payload,
                  result = EXCLUDED.result,
                  updated_at = now()
                """,
                (
                    job_id,
                    status,
                    url,
                    json.dumps(_json_safe(payload), ensure_ascii=False),
                    json.dumps(_json_safe(result), ensure_ascii=False),
                ),
            )

    async def update(self, job_id: str, updates: dict[str, Any]) -> None:
        current = await self.get(job_id)
        if not current:
            return
        status = str(updates.get("status") or current.get("status") or "queued")
        url = str(updates.get("url") or current.get("url") or "")
        result = {
            key: value
            for key, value in {**current, **updates}.items()
            if key not in {"job_id", "status", "url", "payload"}
        }
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute(
                """
                UPDATE crawl_jobs
                SET status = %s, url = %s, result = %s::jsonb, updated_at = now()
                WHERE job_id = %s
                """,
                (status, url, json.dumps(_json_safe(result), ensure_ascii=False), job_id),
            )

    async def get(self, job_id: str) -> dict[str, Any] | None:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                SELECT job_id, status, url, payload, result, created_at::text, updated_at::text
                FROM crawl_jobs
                WHERE job_id = %s
                """,
                (job_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row.get("result") or {})
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "url": row["url"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            **result,
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
