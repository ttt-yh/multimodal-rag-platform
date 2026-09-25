"""PostgreSQL persistence for the first ingestion slice.

The repository stores source truth and parser output only. It does not build
embeddings, Chroma, BM25, or activate a knowledge index. Quality is persisted
separately from technical job completion so a successful parse cannot silently
become published knowledge.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from psycopg.types.json import Jsonb

from multimodal_rag.core.errors import AppError
from multimodal_rag.core.models import IngestionPersistenceResult, PreviewResult
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


QUALITY_STATES = {"candidate", "needs_review", "blocked", "approved"}


def _profile_hash(profile: dict[str, Any]) -> str:
    encoded = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _processing_id(version_id: str, parser_version: str, profile_sha256: str) -> str:
    value = f"{version_id}:{parser_version}:{profile_sha256}".encode()
    return "proc_" + hashlib.sha256(value).hexdigest()[:48]


def persist_preview(settings: Settings, preview: PreviewResult, assessment: dict,
                    *, profile: dict[str, Any] | None = None,
                    job_id: str | None = None,
                    element_metadata: list[dict[str, Any]] | None = None) -> IngestionPersistenceResult:
    """Persist one deterministic local parse; repeat calls are idempotent.

    `profile` must describe processing inputs, not answers or evaluation labels.
    The assessment is a screening result, never an approval decision.
    """
    quality = assessment.get("status")
    if quality not in QUALITY_STATES - {"approved"}:
        raise AppError("invalid_quality_status", "解析质量状态不允许直接批准或格式不正确", 422)
    profile = dict(profile or {})
    profile.setdefault("quality_policy", assessment.get("policy_version"))
    profile.setdefault("parser_version", preview.parser_version)
    profile_sha = _profile_hash(profile)
    processing_id = _processing_id(preview.version.version_id, preview.parser_version, profile_sha)
    job_id = job_id or ("job_" + hashlib.sha256(("ingest:" + processing_id).encode()).hexdigest()[:48])
    idempotent = False
    with connection(settings) as conn:
        existing = conn.execute(
            "SELECT processing_version_id,element_count,quality_status,release_status "
            "FROM mrag.processing_versions WHERE processing_version_id=%s", (processing_id,)
        ).fetchone()
        if existing:
            idempotent = True
            if existing[2] != quality:
                raise AppError("quality_status_conflict", "同一处理版本的质量状态发生变化，请建立审核事件", 409)
            count = existing[1]
            release = existing[3]
        else:
            conn.execute(
                "INSERT INTO mrag.documents(document_id,knowledge_base,title,source_path,source_family,format,license) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (document_id) DO UPDATE SET "
                "knowledge_base=EXCLUDED.knowledge_base,title=EXCLUDED.title,source_path=EXCLUDED.source_path,"
                "source_family=EXCLUDED.source_family,format=EXCLUDED.format,license=EXCLUDED.license",
                (preview.document.document_id, preview.document.knowledge_base, preview.document.title,
                 preview.document.source_path, preview.document.source_family, preview.document.format,
                 preview.document.license),
            )
            conn.execute(
                "INSERT INTO mrag.document_versions(version_id,document_id,content_sha256,source_revision,status) "
                "VALUES (%s,%s,%s,%s,'draft') ON CONFLICT (document_id,content_sha256) DO NOTHING",
                (preview.version.version_id, preview.document.document_id, preview.version.content_sha256,
                 preview.version.source_revision),
            )
            actual_version = conn.execute(
                "SELECT version_id FROM mrag.document_versions WHERE document_id=%s AND content_sha256=%s",
                (preview.document.document_id, preview.version.content_sha256),
            ).fetchone()
            if not actual_version or actual_version[0] != preview.version.version_id:
                raise AppError("content_version_conflict", "同内容对应的版本编号与输入不一致", 409)
            conn.execute(
                "INSERT INTO mrag.processing_versions(processing_version_id,document_id,version_id,parser_version,"
                "profile_sha256,processing_profile,quality_status,release_status,element_count,warning_count) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,'draft',%s,%s)",
                (processing_id, preview.document.document_id, preview.version.version_id, preview.parser_version,
                 profile_sha, Jsonb(profile), quality, len(preview.elements), len(preview.warnings)),
            )
            metadata_by_order = {
                int(row["element"]["order"]): row for row in (element_metadata or [])
                if isinstance(row, dict) and isinstance(row.get("element"), dict)
                and type(row["element"].get("order")) is int
            }
            for element in preview.elements:
                identity = hashlib.sha256(f"{processing_id}:{element.order}".encode()).hexdigest()[:24]
                element_id = "el_" + identity
                metadata = metadata_by_order.get(element.order, {})
                default_eligible = bool(element.raw_text.strip() or element.image_ref)
                index_eligible = metadata.get("index_eligible", default_eligible)
                if not isinstance(index_eligible, bool):
                    raise AppError("invalid_element_policy", "元素索引策略格式不正确", 422)
                excluded_reason = metadata.get("excluded_reason")
                if excluded_reason is not None and not isinstance(excluded_reason, str):
                    raise AppError("invalid_element_policy", "元素排除原因格式不正确", 422)
                if not index_eligible and not excluded_reason:
                    excluded_reason = "empty_element"
                conn.execute(
                    "INSERT INTO mrag.elements(element_id,document_id,version_id,processing_version_id,parser_version,"
                    "kind,ordinal,raw_text,heading_path,source,image_ref,generated_description,index_eligible,excluded_reason) VALUES "
                    "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (element_id, element.document_id, element.version_id, processing_id, preview.parser_version,
                     element.kind, element.order, element.raw_text, Jsonb(element.heading_path),
                     Jsonb(element.source.model_dump()), element.image_ref, element.generated_description,
                     index_eligible, excluded_reason),
                )
            count, release = len(preview.elements), "draft"
        updated = conn.execute(
            "UPDATE mrag.ingestion_jobs SET document_id=%s,version_id=%s,processing_version_id=%s,"
            "processing_profile=%s,quality_status=%s,updated_at=now() WHERE job_id=%s",
            (preview.document.document_id, preview.version.version_id, processing_id,
             Jsonb(profile), quality, job_id),
        ).rowcount
        if updated != 1:
            conn.execute(
                "INSERT INTO mrag.ingestion_jobs(job_id,document_id,version_id,processing_version_id,operation,status,stage,"
                "idempotency_key,processing_profile,quality_status) VALUES (%s,%s,%s,%s,'ingest','succeeded',%s,%s,%s,%s) "
                "ON CONFLICT (idempotency_key) DO UPDATE SET processing_version_id=EXCLUDED.processing_version_id,"
                "processing_profile=EXCLUDED.processing_profile,quality_status=EXCLUDED.quality_status,updated_at=now()",
                (job_id, preview.document.document_id, preview.version.version_id, processing_id,
                 "quality_" + quality, "ingest:" + processing_id, Jsonb(profile), quality),
            )
    return IngestionPersistenceResult(document_id=preview.document.document_id, version_id=preview.version.version_id,
        processing_version_id=processing_id, job_id=job_id, quality_status=quality, release_status=release,
        element_count=count, idempotent=idempotent)
