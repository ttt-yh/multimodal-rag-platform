"""Auditable document update/retirement operations.

Retirement is deliberately a two-step operation: mark source and processing
versions retired, then build and activate an index whose scope excludes them.
Until the new index is active, the old immutable index remains readable.
"""
from __future__ import annotations

import re
from uuid import uuid4

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings

_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def retire_document(settings: Settings, document_id: str, reviewer: str, notes: str) -> dict:
    if not _ID.fullmatch(document_id or ""):
        raise AppError("invalid_identifier", "document_id格式不正确", 422)
    if not reviewer or len(reviewer) > 128 or not notes or len(notes) > 4000:
        raise AppError("invalid_retirement", "操作人不能为空，说明长度必须在1到4000之间", 422)
    retirement_id = "retire_" + uuid4().hex
    with connection(settings) as conn:
        document = conn.execute(
            "SELECT document_id,title FROM mrag.documents WHERE document_id=%s FOR UPDATE",
            (document_id,),
        ).fetchone()
        if not document:
            raise AppError("document_not_found", "文档不存在", 404)
        versions = conn.execute(
            "SELECT version_id FROM mrag.document_versions WHERE document_id=%s",
            (document_id,),
        ).fetchall()
        if not versions:
            raise AppError("document_has_no_version", "文档没有可下线的版本", 409)
        conn.execute(
            """INSERT INTO mrag.document_retirements(retirement_id,document_id,reviewer,notes)
               VALUES (%s,%s,%s,%s)""",
            (retirement_id, document_id, reviewer, notes),
        )
        changed_versions = conn.execute(
            """UPDATE mrag.document_versions SET status='retired'
               WHERE document_id=%s AND status <> 'retired' RETURNING version_id""",
            (document_id,),
        ).fetchall()
        changed_processing = conn.execute(
            """UPDATE mrag.processing_versions SET release_status='retired',updated_at=now()
               WHERE document_id=%s AND release_status <> 'retired'
               RETURNING processing_version_id""",
            (document_id,),
        ).fetchall()
    return {
        "status": "retired", "document_id": document_id, "title": document[1],
        "retirement_id": retirement_id, "reviewer": reviewer, "notes": notes,
        "retired_version_count": len(changed_versions),
        "retired_processing_version_count": len(changed_processing),
        "requires_index_rebuild": True,
    }
