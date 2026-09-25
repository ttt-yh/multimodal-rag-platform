"""Index activation gate and read-side version selection."""
from __future__ import annotations

from pathlib import Path

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


def _processing_scope(manifest: dict | None, fallback_id: str) -> set[str]:
    values = manifest.get("processing_version_ids") if isinstance(manifest, dict) else None
    return {str(value) for value in (values or [fallback_id])}


def activate_index(settings: Settings, index_version_id: str) -> dict:
    with connection(settings) as conn:
        row = conn.execute(
            """SELECT iv.processing_version_id,iv.status,iv.chunk_count,iv.vector_count,iv.keyword_count,
                      iv.artifact_path,pv.quality_status,pv.release_status,pv.document_id,pv.version_id,
                      iv.manifest
               FROM mrag.index_versions iv JOIN mrag.processing_versions pv
                 ON pv.processing_version_id=iv.processing_version_id
               WHERE iv.index_version_id=%s FOR UPDATE""", (index_version_id,)
        ).fetchone()
        if not row:
            raise AppError("index_not_found", "索引版本不存在", 404)
        processing_id, status, chunks, vectors, keywords, artifact, quality, release, document_id, version_id, manifest = row
        scope_ids = sorted(_processing_scope(manifest, processing_id))
        approved = conn.execute(
            "SELECT processing_version_id FROM mrag.processing_versions "
            "WHERE processing_version_id = ANY(%s) AND quality_status='approved' "
            "AND release_status <> 'retired'",
            (scope_ids,),
        ).fetchall()
        if len(approved) != len(scope_ids):
            raise AppError("quality_not_approved", "索引范围内仍有处理版本未通过人工审核，不能激活", 409)
        if not chunks or vectors != chunks or keywords != chunks:
            raise AppError("index_incomplete", "向量、关键词索引与Chunk数量不一致，不能激活", 409)
        current = conn.execute(
            "SELECT index_version_id,processing_version_id,manifest FROM mrag.index_versions "
            "WHERE status='active' AND index_version_id<>%s FOR UPDATE", (index_version_id,)
        ).fetchone()
        if current:
            current_scope = _processing_scope(current[2], current[1])
            omitted = sorted(current_scope - set(scope_ids))
            if omitted:
                retired = conn.execute(
                    """SELECT pv.processing_version_id
                       FROM mrag.processing_versions pv
                       JOIN mrag.document_versions dv
                         ON dv.document_id=pv.document_id AND dv.version_id=pv.version_id
                       WHERE pv.processing_version_id = ANY(%s) AND dv.status='retired'""",
                    (omitted,),
                ).fetchall()
                if {row[0] for row in retired} != set(omitted):
                    raise AppError(
                        "index_scope_regression",
                        "待激活索引未包含当前知识范围；只有显式下线的文档才能从索引移除",
                        409,
                    )
        artifact_path = (settings.project_root / artifact).resolve()
        if not artifact_path.is_relative_to(settings.project_root) or not artifact_path.is_dir():
            raise AppError("index_artifact_missing", "索引产物目录不存在或不在项目目录内", 409)
        # 一个数据库只允许一个当前可读索引；多文档索引的锚点不同，不能只按
        # processing_version_id 下线旧版本。
        conn.execute("UPDATE mrag.index_versions SET status='retired',updated_at=now() "
                     "WHERE status='active' AND index_version_id<>%s", (index_version_id,))
        conn.execute("UPDATE mrag.index_versions SET status='active',updated_at=now() "
                     "WHERE index_version_id=%s", (index_version_id,))
        conn.execute("UPDATE mrag.processing_versions SET release_status='active',updated_at=now() "
                     "WHERE processing_version_id = ANY(%s)", (scope_ids,))
        conn.execute(
            """UPDATE mrag.document_versions dv SET status='retired'
               WHERE dv.status='active' AND EXISTS (
                   SELECT 1 FROM mrag.processing_versions pv
                   WHERE pv.processing_version_id = ANY(%s)
                     AND pv.document_id = dv.document_id
               )""", (scope_ids,))
        conn.execute(
            """UPDATE mrag.document_versions dv SET status='active'
               WHERE dv.version_id IN (
                   SELECT pv.version_id FROM mrag.processing_versions pv
                   WHERE pv.processing_version_id = ANY(%s)
               )""", (scope_ids,))
    return {"status": "active", "index_version_id": index_version_id, "processing_version_id": processing_id}
