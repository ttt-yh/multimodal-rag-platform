"""Read-only views used by the knowledge-base workbench."""
from __future__ import annotations

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


VALID_FILTERS = {"all", "active", "draft", "review"}


def list_knowledge(settings: Settings, *, status: str = "all", limit: int = 50,
                   offset: int = 0) -> dict:
    if status not in VALID_FILTERS:
        raise AppError("invalid_knowledge_filter", "知识库状态筛选值不正确", 422)
    if not 1 <= limit <= 100 or offset < 0:
        raise AppError("invalid_pagination", "分页参数不正确", 422)

    status_sql = ""
    params: list[object] = []
    if status == "active":
        status_sql = "WHERE COALESCE(iv.status, pv.release_status, dv.status) = 'active'"
    elif status == "draft":
        status_sql = "WHERE COALESCE(iv.status, pv.release_status, dv.status) IN ('draft','retired')"
    elif status == "review":
        status_sql = "WHERE pv.quality_status IN ('candidate','needs_review','blocked')"

    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            f"""SELECT d.document_id,d.title,d.source_path,d.format,d.knowledge_base,
                       dv.version_id,dv.status,dv.source_revision,dv.created_at,
                       pv.processing_version_id,pv.quality_status,pv.release_status,
                       pv.element_count,pv.warning_count,pv.updated_at,
                       COALESCE(ch.chunk_count,0),iv.index_version_id,iv.status,
                       iv.embedding_model,iv.embedding_dimensions,iv.bm25_version
                  FROM mrag.documents d
                  LEFT JOIN LATERAL (
                    SELECT version_id,status,source_revision,created_at
                    FROM mrag.document_versions x
                    WHERE x.document_id=d.document_id
                    ORDER BY (x.status='active') DESC,x.created_at DESC
                    LIMIT 1
                  ) dv ON TRUE
                  LEFT JOIN LATERAL (
                    SELECT processing_version_id,quality_status,release_status,
                           element_count,warning_count,updated_at
                    FROM mrag.processing_versions x
                    WHERE x.document_id=d.document_id
                    ORDER BY (x.release_status='active') DESC,x.updated_at DESC
                    LIMIT 1
                  ) pv ON TRUE
                  LEFT JOIN LATERAL (
                    SELECT processing_version_id,COUNT(*)::int AS chunk_count
                    FROM mrag.chunks x
                    WHERE x.processing_version_id=pv.processing_version_id
                    GROUP BY processing_version_id
                  ) ch ON TRUE
                  LEFT JOIN LATERAL (
                    SELECT index_version_id,status,embedding_model,embedding_dimensions,bm25_version
                    FROM mrag.index_versions x
                    WHERE x.processing_version_id=pv.processing_version_id
                    ORDER BY (x.status='active') DESC,x.updated_at DESC
                    LIMIT 1
                  ) iv ON TRUE
                  {status_sql}
                  ORDER BY COALESCE(pv.updated_at,d.created_at) DESC,d.document_id
                  LIMIT %s OFFSET %s""",
            (*params, limit, offset),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM mrag.documents").fetchone()[0]
        approved = conn.execute(
            "SELECT COUNT(*) FROM mrag.processing_versions WHERE quality_status='approved'"
        ).fetchone()[0]
        active_indexes = conn.execute(
            "SELECT COUNT(*) FROM mrag.index_versions WHERE status='active'"
        ).fetchone()[0]

    documents = []
    for row in rows:
        documents.append({
            "document_id": row[0], "title": row[1], "source_path": row[2],
            "format": row[3], "knowledge_base": row[4],
            "version": {"version_id": row[5], "status": row[6],
                        "source_revision": row[7], "created_at": row[8].isoformat() if row[8] else None},
            "processing": {"processing_version_id": row[9], "quality_status": row[10],
                           "release_status": row[11], "element_count": row[12],
                           "warning_count": row[13], "updated_at": row[14].isoformat() if row[14] else None},
            "chunk_count": row[15],
            "index": {"index_version_id": row[16], "status": row[17],
                      "embedding_model": row[18], "embedding_dimensions": row[19],
                      "bm25_version": row[20]},
        })
    return {"documents": documents, "summary": {
        "total_documents": total, "approved_processing_versions": approved,
        "active_indexes": active_indexes,
    }, "pagination": {"limit": limit, "offset": offset, "returned": len(documents), "total": total},
            "filter": status}
