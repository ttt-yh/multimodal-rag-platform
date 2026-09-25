"""Durable lifecycle operations for the first single-worker implementation.

The worker owns a short lease rather than a process-local queue. PostgreSQL
therefore remains the source of truth when a process stops between two stages.
These functions only change task state; parsing and quality decisions remain in
the application layer.
"""
from __future__ import annotations

from datetime import datetime

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


def _validate_lease_seconds(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 10 <= value <= 3600:
        raise AppError("invalid_lease", "任务租约必须是10到3600秒之间的整数", 422)
    return value


def claim_next_job(settings: Settings, worker_id: str, *, lease_seconds: int = 300,
                   job_id: str | None = None) -> dict | None:
    """Claim one job using row-level locking; optionally restrict to one job."""
    if not worker_id or len(worker_id) > 128:
        raise AppError("invalid_worker_id", "worker编号不能为空且长度不能超过128", 422)
    lease_seconds = _validate_lease_seconds(lease_seconds)
    with connection(settings) as conn:
        row = conn.execute(
            """WITH candidate AS (
                   SELECT job_id FROM mrag.ingestion_jobs
                   WHERE (%s::text IS NULL OR job_id=%s)
                     AND (status='pending' OR (status='running' AND lease_expires_at < now()))
                   ORDER BY created_at, job_id
                   FOR UPDATE SKIP LOCKED LIMIT 1
               )
               UPDATE mrag.ingestion_jobs AS j
               SET status='running', stage='claimed', lease_owner=%s,
                   lease_expires_at=now() + (%s * interval '1 second'),
                   heartbeat_at=now(), updated_at=now(), attempts=j.attempts+1
               FROM candidate
               WHERE j.job_id=candidate.job_id
               RETURNING j.job_id,j.document_id,j.version_id,j.processing_version_id,
                         j.operation,j.status,j.stage,j.attempts,j.lease_expires_at""",
            (job_id, job_id, worker_id, lease_seconds),
        ).fetchone()
    if not row:
        return None
    return dict(zip(("job_id", "document_id", "version_id", "processing_version_id",
                     "operation", "status", "stage", "attempts", "lease_expires_at"), row))


def heartbeat_job(settings: Settings, job_id: str, worker_id: str, *, lease_seconds: int = 300) -> datetime:
    lease_seconds = _validate_lease_seconds(lease_seconds)
    with connection(settings) as conn:
        row = conn.execute(
            """UPDATE mrag.ingestion_jobs
               SET heartbeat_at=now(), lease_expires_at=now() + (%s * interval '1 second'), updated_at=now()
               WHERE job_id=%s AND status='running' AND lease_owner=%s
               RETURNING lease_expires_at""",
            (lease_seconds, job_id, worker_id),
        ).fetchone()
    if not row:
        raise AppError("job_lease_lost", "任务不存在、已被其他worker接管或不在运行状态", 409)
    return row[0]


def finish_job(settings: Settings, job_id: str, worker_id: str, *, stage: str = "completed") -> None:
    with connection(settings) as conn:
        changed = conn.execute(
            """UPDATE mrag.ingestion_jobs
               SET status='succeeded', stage=%s, lease_owner=NULL, lease_expires_at=NULL,
                   heartbeat_at=now(), completed_at=now(), updated_at=now()
               WHERE job_id=%s AND status='running' AND lease_owner=%s""",
            (stage, job_id, worker_id),
        ).rowcount
    if changed != 1:
        raise AppError("job_lease_lost", "任务不存在、已被其他worker接管或不在运行状态", 409)


def fail_job(settings: Settings, job_id: str, worker_id: str, error_code: str, *, stage: str = "failed") -> None:
    if not error_code or len(error_code) > 128:
        raise AppError("invalid_error_code", "错误编号不能为空且长度不能超过128", 422)
    with connection(settings) as conn:
        changed = conn.execute(
            """UPDATE mrag.ingestion_jobs
               SET status='failed', stage=%s, error_code=%s, lease_owner=NULL,
                   lease_expires_at=NULL, heartbeat_at=now(), updated_at=now()
               WHERE job_id=%s AND status='running' AND lease_owner=%s""",
            (stage, error_code, job_id, worker_id),
        ).rowcount
    if changed != 1:
        raise AppError("job_lease_lost", "任务不存在、已被其他worker接管或不在运行状态", 409)
