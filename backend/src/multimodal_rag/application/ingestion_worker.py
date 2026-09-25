"""Restartable Markdown/TXT/PDF ingestion worker.

Text sources are processed locally in one step. PDF sources use a durable
MinerU state machine: submit+upload, poll, then normalize+persist. A job stores
only the provider batch id; signed URLs and credentials never enter PostgreSQL.
Index construction remains a later, explicitly approved stage.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
from typing import Any

from pypdf import PdfReader
from psycopg.types.json import Jsonb

from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.chunking import split_elements
from multimodal_rag.core.errors import AppError
from multimodal_rag.core.models import (Document, DocumentVersion, Element,
                                        IngestionJob, PreviewResult)
from multimodal_rag.core.quality import assess_parser_result
from multimodal_rag.infrastructure.chunk_repository import persist_chunks
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.documents import ManifestEntry, WhitelistDocumentReader
from multimodal_rag.infrastructure.http_gateway import HttpGateway
from multimodal_rag.infrastructure.ingestion_repository import (_processing_id,
    _profile_hash, persist_preview)
from multimodal_rag.infrastructure.parser_adapter import ParserAdapter
from multimodal_rag.infrastructure.parser_artifact_repository import persist_parser_artifact
from multimodal_rag.infrastructure.parser_normalization import (NORMALIZER_VERSION,
    materialize_archive_assets, normalize_archive)
from multimodal_rag.infrastructure.settings import Settings
from multimodal_rag.infrastructure.text_parser import PARSER_VERSION
from multimodal_rag.infrastructure.worker_repository import (claim_next_job, defer_job,
    fail_job, finish_job)


CHUNKING_VERSION = "structure-recursive-v1"
MAX_CHARACTERS = 1200
OVERLAP_CHARACTERS = 160
MIN_CHARACTERS = 100


def _reader(settings: Settings) -> WhitelistDocumentReader:
    return WhitelistDocumentReader(settings.project_root, settings.max_document_bytes,
                                   settings.ingestion_manifest, settings.max_pdf_bytes)


def _version_id(entry: ManifestEntry) -> str:
    return "ver_" + hashlib.sha256(f"{entry.document_id}:{entry.sha256}".encode()).hexdigest()


def _pdf_page_count(payload: bytes, settings: Settings) -> int:
    try:
        reader = PdfReader(BytesIO(payload))
        if reader.is_encrypted:
            raise AppError("encrypted_pdf", "不支持加密 PDF 入库", 422)
        count = len(reader.pages)
    except AppError:
        raise
    except Exception:
        raise AppError("invalid_pdf", "PDF 结构无法读取", 422) from None
    if not 1 <= count <= settings.max_pdf_pages:
        raise AppError("pdf_page_limit_exceeded", "PDF 页数超出入库限制", 422)
    return count


def enqueue_document(settings: Settings, document_id: str, *, profile: dict[str, Any] | None = None) -> dict:
    """Create one pending job from the local whitelist without calling a provider."""
    entry, payload = _reader(settings).read_source(document_id)
    parser_version = NORMALIZER_VERSION if entry.format == "pdf" else PARSER_VERSION
    profile = dict(profile or {})
    profile.setdefault("worker_version", "mineru-pdf-worker-v1" if entry.format == "pdf"
                       else "local-text-worker-v1")
    profile.setdefault("parser_version", parser_version)
    profile.setdefault("chunking_version", CHUNKING_VERSION)
    profile.setdefault("max_characters", MAX_CHARACTERS)
    profile.setdefault("overlap_characters", OVERLAP_CHARACTERS)
    profile.setdefault("min_characters", MIN_CHARACTERS)
    profile.setdefault("quality_policy", "parser-quality-v1")
    if entry.format == "pdf":
        profile.setdefault("parser_service", "mineru")
        profile.setdefault("parser_model", settings.parser_model)
        profile.setdefault("page_count", _pdf_page_count(payload, settings))
    profile_hash = _profile_hash(profile)
    version_id = _version_id(entry)
    processing_id = _processing_id(version_id, parser_version, profile_hash)
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
            "format": entry.format, "requires_external_parser": entry.format == "pdf",
            "profile": profile}


def _stable_elements(elements: list[Element], processing_version_id: str) -> list[Element]:
    return [element.model_copy(update={
        "element_id": "el_" + hashlib.sha256(
            f"{processing_version_id}:{element.order}".encode()).hexdigest()[:24],
        "processing_version_id": processing_version_id,
    }) for element in elements]


def _split_and_persist(settings: Settings, elements: list[Element], processing_version_id: str,
                       profile: dict[str, Any]) -> tuple[int, int]:
    chunks = split_elements(
        elements, processing_version_id,
        max_characters=int(profile.get("max_characters", MAX_CHARACTERS)),
        overlap_characters=int(profile.get("overlap_characters", OVERLAP_CHARACTERS)),
        min_characters=int(profile.get("min_characters", MIN_CHARACTERS)),
    )
    return len(chunks), persist_chunks(settings, chunks)


def _run_text(settings: Settings, job: dict, worker_id: str, profile: dict[str, Any]) -> dict:
    preview = preview_document(job["document_id"], settings)
    rows = [{"element": e.model_dump(), "index_eligible": bool(e.raw_text.strip() or e.image_ref)}
            for e in preview.elements]
    assessment = assess_parser_result({"elements": rows,
        "warnings": [{"code": warning} for warning in preview.warnings],
        "artifact": {"missing_image_references": 0}})
    persisted = persist_preview(settings, preview, assessment, profile=profile, job_id=job["job_id"])
    elements = _stable_elements(preview.elements, persisted.processing_version_id)
    chunk_count, chunks_persisted = _split_and_persist(
        settings, elements, persisted.processing_version_id, profile)
    finish_job(settings, job["job_id"], worker_id, stage="chunked")
    return {"status": "succeeded", "job": persisted.model_dump(), "quality": assessment,
            "chunk_count": chunk_count, "chunks_persisted": chunks_persisted}


def _pdf_sample(entry: ManifestEntry, version_id: str, job_id: str, page_count: int) -> dict:
    return {"sample_id": job_id, "page_count": page_count, "pages": [{
        "document_id": entry.document_id, "version_id": version_id,
        "source_path": entry.path, "source_sha256": entry.sha256,
        "source_page": page, "split": "dev", "source_kind": "pdf",
    } for page in range(1, page_count + 1)]}


def _run_pdf(settings: Settings, job: dict, worker_id: str, profile: dict[str, Any],
             entry: ManifestEntry, payload: bytes, external_batch_id: str | None,
             gateway: HttpGateway) -> dict:
    adapter = ParserAdapter(gateway)
    page_count = int(profile.get("page_count") or _pdf_page_count(payload, settings))
    if external_batch_id is None:
        if gateway.budget.max_requests - gateway.budget.used < 2:
            defer_job(settings, job["job_id"], worker_id, stage="awaiting_parser_budget")
            return {"status": "pending", "stage": "awaiting_parser_budget",
                    "required_requests": 2}
        receipt = adapter.request_document_upload(
            f"{entry.document_id}.pdf", job["job_id"], page_count=page_count,
            max_pages=settings.max_pdf_pages)
        adapter.upload_document(receipt["signed_url"], payload, max_bytes=settings.max_pdf_bytes)
        defer_job(settings, job["job_id"], worker_id, stage="parser_submitted",
                  external_batch_id=receipt["batch_id"])
        return {"status": "pending", "stage": "parser_submitted",
                "provider_batch_id": receipt["batch_id"]}

    state = adapter.poll(external_batch_id, batch=True, data_id=job["job_id"])
    if state["state"] == "failed":
        raise AppError("parser_task_failed", "MinerU 解析任务失败", 502)
    if state["state"] != "done":
        stage = "parser_" + state["state"].replace("-", "_")
        defer_job(settings, job["job_id"], worker_id, stage=stage,
                  external_batch_id=external_batch_id)
        return {"status": "pending", "stage": stage, "progress": state.get("progress")}

    archive = adapter.fetch_result_bytes(state["signed_result_url"])
    normalized = normalize_archive(
        archive, _pdf_sample(entry, job["version_id"], job["job_id"], page_count),
        settings.parser_max_unpacked_bytes)
    processing_id = _processing_id(job["version_id"], NORMALIZER_VERSION, _profile_hash(profile))
    materialize_archive_assets(archive, normalized, settings.project_root, processing_id)
    elements = [Element(**row["element"]) for row in normalized["elements"]]
    warning_codes = [str(row.get("code", "parser_warning")) for row in normalized["warnings"]]
    document = Document(document_id=entry.document_id, title=entry.title,
                        knowledge_base=entry.knowledge_base, source_path=entry.path,
                        source_family=entry.source_family, format="pdf", license=entry.license)
    version = DocumentVersion(version_id=job["version_id"], document_id=entry.document_id,
                              content_sha256=entry.sha256, source_revision=entry.source_revision)
    preview = PreviewResult(document=document, version=version,
        job=IngestionJob(job_id=job["job_id"], document_id=entry.document_id,
                         version_id=job["version_id"], operation="ingest",
                         status="running", stage="parser_normalized"),
        elements=elements, warnings=warning_codes, parser_version=NORMALIZER_VERSION,
        source_verified=True)
    assessment = assess_parser_result(normalized)
    persisted = persist_preview(settings, preview, assessment, profile=profile,
        job_id=job["job_id"], element_metadata=normalized["elements"])
    stable = _stable_elements(elements, persisted.processing_version_id)
    eligible_orders = {row["element"]["order"] for row in normalized["elements"]
                       if row.get("index_eligible")}
    eligible = [element for element in stable if element.order in eligible_orders]
    chunk_count, chunks_persisted = _split_and_persist(
        settings, eligible, persisted.processing_version_id, profile)
    artifact = persist_parser_artifact(settings,
        processing_version_id=persisted.processing_version_id, job_id=job["job_id"],
        provider_batch_id=external_batch_id, normalizer_version=NORMALIZER_VERSION,
        archive=archive, normalized=normalized)
    finish_job(settings, job["job_id"], worker_id, stage="chunked")
    return {"status": "succeeded", "job": persisted.model_dump(), "quality": assessment,
            "chunk_count": chunk_count, "chunks_persisted": chunks_persisted,
            "parser_artifact": artifact}


def run_once(settings: Settings, worker_id: str, *, lease_seconds: int = 300,
             job_id: str | None = None, parser_gateway: HttpGateway | None = None) -> dict | None:
    """Advance at most one durable job by one local or external-parser step."""
    job = claim_next_job(settings, worker_id, lease_seconds=lease_seconds, job_id=job_id)
    if job is None:
        return None
    entry: ManifestEntry | None = None
    external_batch_id: str | None = None
    try:
        with connection(settings, read_only=True) as conn:
            row = conn.execute(
                "SELECT processing_profile,external_batch_id FROM mrag.ingestion_jobs WHERE job_id=%s",
                (job["job_id"],)).fetchone()
        profile = row[0] if row and isinstance(row[0], dict) else {}
        external_batch_id = row[1] if row else None
        entry, payload = _reader(settings).read_source(job["document_id"])
        if entry.format != "pdf":
            return _run_text(settings, job, worker_id, profile)
        if parser_gateway is None:
            defer_job(settings, job["job_id"], worker_id, stage="awaiting_parser_confirmation",
                      external_batch_id=external_batch_id)
            return {"status": "pending", "stage": "awaiting_parser_confirmation",
                    "requires_external_parser": True}
        return _run_pdf(settings, job, worker_id, profile, entry, payload,
                        external_batch_id, parser_gateway)
    except AppError as exc:
        resumable = (entry is not None and entry.format == "pdf" and external_batch_id
                     and exc.code in {"call_budget_exhausted", "provider_transport_error",
                                      "provider_http_error"})
        try:
            if resumable:
                defer_job(settings, job["job_id"], worker_id, stage="parser_retry",
                          external_batch_id=external_batch_id)
                return {"status": "pending", "stage": "parser_retry", "error_code": exc.code}
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
