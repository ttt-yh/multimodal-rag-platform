"""Read task state and record human quality decisions."""
from __future__ import annotations

import re
from uuid import uuid4

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _check_id(value: str, field: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise AppError("invalid_identifier", f"{field}格式不正确", 422)


def get_job(settings: Settings, job_id: str) -> dict:
    _check_id(job_id, "job_id")
    with connection(settings, read_only=True) as conn:
        row = conn.execute(
            """SELECT job_id,document_id,version_id,processing_version_id,operation,status,stage,
                      quality_status,error_code,attempts,created_at,updated_at
               FROM mrag.ingestion_jobs WHERE job_id=%s""", (job_id,)
        ).fetchone()
    if not row:
        raise AppError("job_not_found", "入库任务不存在", 404)
    return dict(zip(("job_id", "document_id", "version_id", "processing_version_id", "operation", "status",
                     "stage", "quality_status", "error_code", "attempts", "created_at", "updated_at"), row))


def record_quality_review(settings: Settings, processing_version_id: str, reviewer: str,
                          decision: str, notes: str = "") -> dict:
    _check_id(processing_version_id, "processing_version_id")
    if not reviewer or len(reviewer) > 128 or not notes or len(notes) > 4000:
        raise AppError("invalid_review", "审核人不能为空，审核说明长度必须在1到4000之间", 422)
    if decision not in {"approved", "rejected"}:
        raise AppError("invalid_review", "审核结论只能是approved或rejected", 422)
    review_id = "review_" + uuid4().hex
    quality = "approved" if decision == "approved" else "blocked"
    with connection(settings) as conn:
        exists = conn.execute(
            "SELECT release_status FROM mrag.processing_versions WHERE processing_version_id=%s",
            (processing_version_id,),
        ).fetchone()
        if not exists:
            raise AppError("processing_version_not_found", "处理版本不存在", 404)
        conn.execute(
            "INSERT INTO mrag.quality_reviews(review_id,processing_version_id,reviewer,decision,notes) "
            "VALUES (%s,%s,%s,%s,%s)",
            (review_id, processing_version_id, reviewer, decision, notes),
        )
        row = conn.execute(
            "UPDATE mrag.processing_versions SET quality_status=%s,updated_at=now() "
            "WHERE processing_version_id=%s RETURNING release_status",
            (quality, processing_version_id),
        ).fetchone()
    return {"review_id": review_id, "processing_version_id": processing_version_id, "reviewer": reviewer,
            "decision": decision, "quality_status": quality, "release_status": row[0]}
