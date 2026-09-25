"""Persist reproducible PDF parser artifacts without storing signed URLs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


def persist_parser_artifact(settings: Settings, *, processing_version_id: str, job_id: str,
                            provider_batch_id: str, normalizer_version: str,
                            archive: bytes, normalized: dict) -> dict:
    root = (settings.project_root / "data/derived/mineru_artifacts" /
            processing_version_id).resolve()
    allowed = (settings.project_root / "data/derived/mineru_artifacts").resolve()
    if not root.is_relative_to(allowed):
        raise AppError("unsafe_artifact_path", "PDF 解析产物路径不安全", 403)
    root.mkdir(parents=True, exist_ok=True)
    archive_hash = hashlib.sha256(archive).hexdigest()
    archive_path = root / "result.zip"
    summary_path = root / "normalized.json"
    if archive_path.exists() and hashlib.sha256(archive_path.read_bytes()).hexdigest() != archive_hash:
        raise AppError("parser_artifact_conflict", "PDF 解析产物与已有版本冲突", 409)
    if not archive_path.exists():
        temporary = archive_path.with_suffix(".zip.tmp")
        temporary.write_bytes(archive)
        temporary.replace(archive_path)
    # The normalized record contains source text and hashes but never a signed
    # URL, API key or provider response headers.
    serialized = json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)
    summary_path.write_text(serialized, encoding="utf-8")
    relative = archive_path.relative_to(settings.project_root.resolve()).as_posix()
    image_count = sum(1 for row in normalized.get("elements", [])
                      if row.get("materialized_asset"))
    with connection(settings) as conn:
        conn.execute(
            """INSERT INTO mrag.parser_artifacts(processing_version_id,job_id,provider_batch_id,
                   normalizer_version,archive_sha256,artifact_path,image_count)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (processing_version_id) DO UPDATE SET
                 archive_sha256=EXCLUDED.archive_sha256,artifact_path=EXCLUDED.artifact_path,
                 image_count=EXCLUDED.image_count""",
            (processing_version_id, job_id, provider_batch_id, normalizer_version,
             archive_hash, relative, image_count),
        )
    return {"archive_sha256": archive_hash, "artifact_path": relative,
            "normalized_path": summary_path.relative_to(settings.project_root.resolve()).as_posix(),
            "image_count": image_count}
