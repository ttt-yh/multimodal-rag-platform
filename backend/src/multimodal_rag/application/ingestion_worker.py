"""Small, restartable Markdown/TXT ingestion worker.

The worker deliberately has one responsibility: turn a queued local document
into a persisted preview and quality decision. Index construction is a later
stage and is therefore not hidden inside this loop.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.chunking import split_elements
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.chunk_repository import persist_chunks
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.documents import WhitelistDocumentReader
from multimodal_rag.infrastructure.ingestion_repository import _processing_id, _profile_hash, persist_preview
from multimodal_rag.infrastructure.settings import Settings
from multimodal_rag.infrastructure.text_parser import PARSER_VERSION
from multimodal_rag.infrastructure.worker_repository import claim_next_job, fail_job, finish_job
from psycopg.types.json import Jsonb


CHUNKING_VERSION = "structure-recursive-v1"
MAX_CHARACTERS = 1200
OVERLAP_CHARACTERS = 160
MIN_CHARACTERS = 100


def enqueue_document(settings: Settings, document_id: str, *, profile: dict[str, Any] | None = None) -> dict:
    """Create one pending job from the local whitelist without parsing it yet."""
    entry, _ = WhitelistDocumentReader(
        settings.project_root, settings.max_document_bytes, settings.ingestion_manifest
    ).read(document_id)
    profile = dict(profile or {})
    profile.setdefault("worker_version", "local-text-worker-v1")
    profile.setdefault("parser_version", PARSER_VERSION)
    profile.setdefault("chunking_version", CHUNKING_VERSION)
    profile.setdefault("max_characters", MAX_CHARACTERS)
    profile.setdefault("overlap_characters", OVERLAP_CHARACTERS)
    profile.setdefault("min_characters", MIN_CHARACTERS)
    # 与 persist_preview 的质量策略保持一致，确保队列中的 processing_id
    # 与实际落库的 processing_version_id 完全相同。
    profile.setdefault("quality_policy", "parser-quality-v1")
    profile_hash = _profile_hash(profile)
    version_id = "ver_" + hashlib.sha256(f"{entry.document_id}:{entry.sha256}".encode()).hexdigest()
    processing_id = _processing_id(version_id, PARSER_VERSION, profile_hash)
    job_id = "job_" + hashlib.sha256(("ingest:" + processing_id).encode()).hexdigest()[:48]
    idempotency_key = "ingest:" + processing_id
    with connection(settings) as conn:
        conn.execute(
            """INSERT INTO mrag.documents(document_id,knowledge_base,title,source_path,source_family,format,license)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (document_id) DO UPDATE SET title=EXCLUDED.title,source_path=EXCLUDED.source_path,
                 knowledge_base=EXCLUDED.knowledge_base,source_family=EXCLUDED.source_family,
                 format=EXCLUDED.format,license=EXCLUDED.license""",
            (entry.document_id, entry.knowledge_base, entry.title, entry.path, entry.source_family,
             entry.format, entry.license),
        )
        conn.execute(
            """INSERT INTO mrag.document_versions(version_id,document_id,content_sha256,source_revision,status)
               VALUES (%s,%s,%s,%s,'draft') ON CONFLICT (document_id,content_sha256) DO NOTHING""",
            (version_id, entry.document_id, entry.sha256, entry.source_revision),
        )
        conn.execute(
            """INSERT INTO mrag.ingestion_jobs(job_id,document_id,version_id,operation,status,stage,
                 idempotency_key,processing_profile)
               VALUES (%s,%s,%s,'ingest','pending','queued',%s,%s)
               ON CONFLICT (idempotency_key) DO NOTHING""",
            (job_id, entry.document_id, version_id, idempotency_key, Jsonb(profile)),
        )
    return {"job_id": job_id, "document_id": entry.document_id, "version_id": version_id,
            "processing_version_id": processing_id, "idempotency_key": idempotency_key,
            "profile": profile}


def run_once(settings: Settings, worker_id: str, *, lease_seconds: int = 300,
             job_id: str | None = None) -> dict | None:
    """Process at most one queued job; return None when the queue is empty."""
    job = claim_next_job(settings, worker_id, lease_seconds=lease_seconds, job_id=job_id)
    if job is None:
        return None
    try:
        with connection(settings, read_only=True) as conn:
            row = conn.execute("SELECT processing_profile FROM mrag.ingestion_jobs WHERE job_id=%s",
                               (job["job_id"],)).fetchone()
        profile = row[0] if row and isinstance(row[0], dict) else {}
        preview = preview_document(job["document_id"], settings)
        assessment = {"status": "candidate", "policy_version": "worker-v1"}
        rows = [{"element": e.model_dump(), "index_eligible": bool(e.raw_text.strip() or e.image_ref)}
                for e in preview.elements]
        from multimodal_rag.core.quality import assess_parser_result
        assessment = assess_parser_result({"elements": rows,
            "warnings": [{"code": warning} for warning in preview.warnings],
            "artifact": {"missing_image_references": 0}})
        persisted = persist_preview(settings, preview, assessment, profile=profile, job_id=job["job_id"])
        # 元素表中的 ID 以 processing_version_id 为命名空间稳定生成；解析器自身的
        # element_id 只描述原始预览，不能直接作为处理版本之间的持久化身份。
        elements = [element.model_copy(update={
            "element_id": "el_" + hashlib.sha256(
                f"{persisted.processing_version_id}:{element.order}".encode()
            ).hexdigest()[:24],
            "processing_version_id": persisted.processing_version_id,
        }) for element in preview.elements]
        chunks = split_elements(
            elements,
            persisted.processing_version_id,
            max_characters=int(profile.get("max_characters", MAX_CHARACTERS)),
            overlap_characters=int(profile.get("overlap_characters", OVERLAP_CHARACTERS)),
            min_characters=int(profile.get("min_characters", MIN_CHARACTERS)),
        )
        chunks_persisted = persist_chunks(settings, chunks)
        finish_job(settings, job["job_id"], worker_id, stage="chunked")
        return {"status": "succeeded", "job": persisted.model_dump(), "quality": assessment,
                "chunk_count": len(chunks), "chunks_persisted": chunks_persisted}
    except AppError as exc:
        try:
            fail_job(settings, job["job_id"], worker_id, exc.code)
        except AppError:
            pass
        return {"status": "failed", "job_id": job["job_id"], "error_code": exc.code}
    except Exception:
        try:
            fail_job(settings, job["job_id"], worker_id, "worker_error")
        except AppError:
            pass
        return {"status": "failed", "job_id": job["job_id"], "error_code": "worker_error"}
