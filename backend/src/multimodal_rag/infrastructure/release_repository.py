"""Read-side views for quality review and index release workbench."""
from __future__ import annotations

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


def list_review_queue(settings: Settings, *, state: str = "pending", limit: int = 50, offset: int = 0) -> dict:
    if state not in {"pending", "approved", "all"}:
        raise AppError("invalid_review_filter", "审核队列筛选值不正确", 422)
    if not 1 <= limit <= 100 or offset < 0:
        raise AppError("invalid_pagination", "分页参数不正确", 422)
    with connection(settings, read_only=True) as conn:
        quality_sql = {
            "pending": "pv.quality_status IN ('candidate','needs_review','blocked')",
            "approved": "pv.quality_status='approved'",
            "all": "TRUE",
        }[state]
        rows = conn.execute(
            f"""SELECT pv.processing_version_id,pv.document_id,d.title,d.source_path,d.format,
                      pv.version_id,pv.quality_status,pv.release_status,pv.element_count,
                      pv.warning_count,pv.updated_at,COALESCE(ch.chunk_count,0)
                 FROM mrag.processing_versions pv
                 JOIN mrag.documents d ON d.document_id=pv.document_id
                 LEFT JOIN LATERAL (
                   SELECT COUNT(*)::int AS chunk_count FROM mrag.chunks c
                   WHERE c.processing_version_id=pv.processing_version_id
                 ) ch ON TRUE
                WHERE {quality_sql}
                ORDER BY pv.updated_at DESC,pv.processing_version_id
                LIMIT %s OFFSET %s""",
            (limit, offset),
        ).fetchall()
        total = conn.execute(f"SELECT COUNT(*) FROM mrag.processing_versions WHERE {quality_sql}").fetchone()[0]
    items = [{
        "processing_version_id": row[0], "document_id": row[1], "title": row[2],
        "source_path": row[3], "format": row[4], "version_id": row[5],
        "quality_status": row[6], "release_status": row[7], "element_count": row[8],
        "warning_count": row[9], "updated_at": row[10].isoformat() if row[10] else None,
        "chunk_count": row[11],
    } for row in rows]
    return {"items": items, "filter": state, "pagination": {"limit": limit, "offset": offset,
            "returned": len(items), "total": total}}


def list_indexes(settings: Settings, *, status: str = "all", limit: int = 50) -> dict:
    if status not in {"all", "draft", "active", "retired", "failed"}:
        raise AppError("invalid_index_filter", "索引状态筛选值不正确", 422)
    if not 1 <= limit <= 100:
        raise AppError("invalid_pagination", "分页参数不正确", 422)
    where = "" if status == "all" else "WHERE iv.status=%s"
    params = () if status == "all" else (status,)
    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            f"""SELECT iv.index_version_id,iv.status,iv.processing_version_id,
                       iv.embedding_model,iv.embedding_dimensions,iv.bm25_version,
                       iv.chunk_count,iv.vector_count,iv.keyword_count,iv.artifact_path,
                       iv.created_at,iv.updated_at
                  FROM mrag.index_versions iv {where}
                 ORDER BY (iv.status='active') DESC,iv.updated_at DESC
                 LIMIT %s""", (*params, limit)
        ).fetchall()
    items = [{
        "index_version_id": row[0], "status": row[1], "processing_version_id": row[2],
        "embedding_model": row[3], "embedding_dimensions": row[4], "bm25_version": row[5],
        "chunk_count": row[6], "vector_count": row[7], "keyword_count": row[8],
        "artifact_path": row[9], "created_at": row[10].isoformat() if row[10] else None,
        "updated_at": row[11].isoformat() if row[11] else None,
    } for row in rows]
    return {"items": items, "filter": status}
