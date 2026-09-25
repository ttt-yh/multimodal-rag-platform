"""Persistence for index build metadata and active-version selection."""
from __future__ import annotations

from psycopg.types.json import Jsonb

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


def get_active_index(settings: Settings) -> dict:
    """Return the currently published index without opening its artifacts.

    The database is the source of truth for publication.  Artifact existence
    and path validation happen in the application service immediately before
    reading Chroma/BM25, so a stale database row cannot silently be used.
    """
    with connection(settings, read_only=True) as conn:
        row = conn.execute(
            """SELECT index_version_id,processing_version_id,embedding_model,
                      embedding_dimensions,bm25_version,chunk_count,vector_count,
                      keyword_count,artifact_path,status,manifest
               FROM mrag.index_versions
               WHERE status='active'
               ORDER BY updated_at DESC, created_at DESC
               LIMIT 1"""
        ).fetchone()
    if not row:
        raise AppError("no_active_index", "当前没有已激活的知识索引", 409)
    return {
        "index_version_id": row[0], "processing_version_id": row[1],
        "embedding_model": row[2], "embedding_dimensions": row[3],
        "bm25_version": row[4], "chunk_count": row[5],
        "vector_count": row[6], "keyword_count": row[7],
        "artifact_path": row[8], "status": row[9], "manifest": row[10],
    }


def persist_index_manifest(settings: Settings, manifest: dict, artifact_path: str) -> None:
    embedding = manifest["embedding"]
    keyword = manifest["keyword"]
    with connection(settings) as conn:
        conn.execute(
            """INSERT INTO mrag.index_versions(index_version_id,processing_version_id,embedding_model,
                 embedding_dimensions,bm25_version,chunk_count,vector_count,keyword_count,artifact_path,status,manifest)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'draft',%s)
               ON CONFLICT (index_version_id) DO UPDATE SET manifest=EXCLUDED.manifest,
                 artifact_path=EXCLUDED.artifact_path,chunk_count=EXCLUDED.chunk_count,
                 vector_count=EXCLUDED.vector_count,keyword_count=EXCLUDED.keyword_count,
                 updated_at=now()""",
            (manifest["index_version_id"], manifest["processing_version_id"], embedding["model"],
             embedding["dimensions"], keyword["version"], manifest["chunk_count"],
             manifest["vector_count"], manifest["keyword_count"], artifact_path, Jsonb(manifest)),
        )
