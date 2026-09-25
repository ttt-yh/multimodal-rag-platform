"""Build one real dense/keyword index version from persisted Chunks."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.core.models import Chunk, SourceLocation
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.index_backends import build_bm25, build_chroma, load_chroma_embeddings
from multimodal_rag.infrastructure.index_manifest import build_manifest, write_manifest
from multimodal_rag.infrastructure.index_repository import get_active_index, persist_index_manifest
from multimodal_rag.infrastructure.model_adapters import EmbeddingAdapter
from multimodal_rag.infrastructure.settings import Settings


def load_chunks(settings: Settings, processing_version_id: str) -> list[Chunk]:
    return load_chunks_for_processing_versions(settings, [processing_version_id])


def load_chunks_for_processing_versions(settings: Settings,
                                        processing_version_ids: list[str]) -> list[Chunk]:
    if not processing_version_ids:
        return []
    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            """SELECT chunk_id,document_id,version_id,processing_version_id,ordinal,text,heading_path,
                      element_ids,source_locations,length_unit,estimated_length
               FROM mrag.chunks WHERE processing_version_id = ANY(%s)
               ORDER BY processing_version_id,ordinal""",
            (processing_version_ids,),
        ).fetchall()
    return [Chunk(chunk_id=row[0], document_id=row[1], version_id=row[2], processing_version_id=row[3],
                  ordinal=row[4], text=row[5], heading_path=row[6], element_ids=row[7],
                  source_locations=[SourceLocation(**item) for item in row[8]], length_unit=row[9],
                  estimated_length=row[10]) for row in rows]


def build_real_indexes(settings: Settings, processing_version_id: str, *, max_requests: int = 2) -> dict:
    return build_real_indexes_for_processing_versions(settings, [processing_version_id],
                                                      max_requests=max_requests)


def build_real_indexes_for_processing_versions(settings: Settings,
                                               processing_version_ids: list[str], *,
                                               max_requests: int = 2) -> dict:
    processing_version_ids = sorted(set(processing_version_ids))
    if not processing_version_ids:
        raise ValueError("至少需要一个 processing_version_id")
    # index_versions 旧表以第一个处理版本作为兼容锚点；manifest 保存完整集合。
    anchor_processing_id = processing_version_ids[0]
    chunks = load_chunks_for_processing_versions(settings, processing_version_ids)
    manifest = build_manifest(anchor_processing_id, chunks, embedding_model=settings.embedding_model,
                              dimensions=settings.embedding_dimensions, bm25_version="bm25s-v1",
                              processing_version_ids=processing_version_ids)
    index_id = manifest["index_version_id"]
    artifact_root = settings.project_root / "indexes" / index_id
    gateway = HttpGateway(settings, "embedding", budget=CallBudget(max_requests))
    vectors: list[list[float]] = []
    records: list[dict] = []
    try:
        for start in range(0, len(chunks), 10):
            result = EmbeddingAdapter(gateway).embed([chunk.text for chunk in chunks[start:start + 10]])
            vectors.extend(result["vectors"])
        records = gateway.records
        chroma = build_chroma(artifact_root, index_id, chunks, vectors)
        bm25 = build_bm25(artifact_root, index_id, chunks)
    finally:
        records = list(gateway.records)
        gateway.close()
    manifest.update(vector_count=len(vectors), keyword_count=len(chunks), status="draft",
                    artifact_path=str(artifact_root.relative_to(settings.project_root)))
    manifest_path = write_manifest(settings.project_root, manifest)
    persist_index_manifest(settings, manifest, manifest["artifact_path"])
    report = {"status": "built", "index_version_id": index_id,
              "processing_version_id": anchor_processing_id,
              "processing_version_ids": processing_version_ids,
              "chunk_count": len(chunks),
              "vector_count": len(vectors), "keyword_count": len(chunks),
              "embedding_model": settings.embedding_model, "dimensions": settings.embedding_dimensions,
              "chroma": chroma, "bm25": bm25, "manifest_path": str(manifest_path),
              "http_requests": len(records), "new_external_api_calls": len(records),
              "created_at": datetime.now(timezone.utc).isoformat()}
    report_path = settings.project_root / "evals/results/index-build" / f"{index_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({**report, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def build_incremental_index_from_active(settings: Settings,
                                        added_processing_version_ids: list[str], *,
                                        max_requests: int = 1,
                                        exclude_retired: bool = False) -> dict:
    """Extend the active index while reusing all unchanged dense vectors.

    BM25 and the target Chroma collection are rebuilt as immutable artifacts,
    but only chunks absent from the active collection are sent to Embedding.
    """
    added_ids = sorted(set(added_processing_version_ids))
    if not added_ids and not exclude_retired:
        raise ValueError("至少需要一个新增 processing_version_id")
    active = get_active_index(settings)
    manifest = active.get("manifest") or {}
    active_scope = sorted(set(manifest.get("processing_version_ids") or
                              [active["processing_version_id"]]))
    original_active_scope = list(active_scope)
    if exclude_retired:
        with connection(settings, read_only=True) as conn:
            rows = conn.execute(
                """SELECT pv.processing_version_id
                   FROM mrag.processing_versions pv
                   JOIN mrag.document_versions dv
                     ON dv.document_id=pv.document_id AND dv.version_id=pv.version_id
                   WHERE pv.processing_version_id = ANY(%s)
                     AND pv.release_status <> 'retired'
                     AND dv.status <> 'retired'""",
                (active_scope,),
            ).fetchall()
        retained_scope = sorted(row[0] for row in rows)
        if set(retained_scope) == set(active_scope):
            raise AppError("no_retired_versions", "当前活动索引没有需要下线的文档", 409)
        active_scope = retained_scope
    if set(added_ids) & set(active_scope):
        raise AppError("processing_version_already_active", "新增处理版本已包含在当前活动索引中", 409)
    if (active["embedding_model"] != settings.embedding_model or
            active["embedding_dimensions"] != settings.embedding_dimensions or
            active["bm25_version"] != "bm25s-v1"):
        raise AppError("index_configuration_changed", "索引配置已变化，不能复用旧向量，请执行全量重建", 409)

    processing_ids = sorted(set(active_scope + added_ids))
    # Retired chunks are still read from the old artifact so their vectors can
    # be ignored in the new scope without changing the active index in place.
    active_chunks = load_chunks_for_processing_versions(settings, original_active_scope)
    added_chunks = load_chunks_for_processing_versions(settings, added_ids)
    if not added_chunks and not exclude_retired:
        raise AppError("no_new_chunks", "新增处理版本没有可索引Chunk", 409)
    if len(active_chunks) != active["chunk_count"]:
        raise AppError("active_scope_changed", "当前活动索引的Chunk范围与数据库不一致", 409)
    chunks = load_chunks_for_processing_versions(settings, processing_ids)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise AppError("duplicate_chunk_id", "合并后的索引范围包含重复Chunk编号", 409)

    active_root = (settings.project_root / active["artifact_path"]).resolve()
    if not active_root.is_relative_to(settings.project_root) or not active_root.is_dir():
        raise AppError("index_artifact_missing", "当前活动索引产物不存在", 409)
    reused_all = load_chroma_embeddings(active_root, active["index_version_id"],
                                       [chunk.chunk_id for chunk in active_chunks])
    # A retirement rebuild reads the old collection for reuse, but must drop
    # vectors belonging to versions excluded from the new manifest.
    final_chunk_ids = {chunk.chunk_id for chunk in
                       load_chunks_for_processing_versions(settings, active_scope + added_ids)}
    reused = {chunk_id: vector for chunk_id, vector in reused_all.items()
              if chunk_id in final_chunk_ids}

    gateway = HttpGateway(settings, "embedding", budget=CallBudget(max_requests)) if added_chunks else None
    generated: dict[str, list[float]] = {}
    records: list[dict] = []
    try:
        for start in range(0, len(added_chunks), 10):
            batch = added_chunks[start:start + 10]
            result = EmbeddingAdapter(gateway).embed([chunk.text for chunk in batch])
            generated.update({chunk.chunk_id: vector for chunk, vector in zip(batch, result["vectors"])})
        records = gateway.records if gateway else []
    finally:
        if gateway is not None:
            records = list(gateway.records)
            gateway.close()
    vectors_by_id = {**reused, **generated}
    if set(vectors_by_id) != set(chunk_ids):
        raise AppError("incremental_index_incomplete", "复用向量与新增向量未覆盖完整Chunk范围", 409)
    vectors = [vectors_by_id[chunk.chunk_id] for chunk in chunks]

    target_manifest = build_manifest(
        processing_ids[0], chunks, embedding_model=settings.embedding_model,
        dimensions=settings.embedding_dimensions, bm25_version="bm25s-v1",
        processing_version_ids=processing_ids,
    )
    index_id = target_manifest["index_version_id"]
    artifact_root = settings.project_root / "indexes" / index_id
    chroma = build_chroma(artifact_root, index_id, chunks, vectors)
    bm25 = build_bm25(artifact_root, index_id, chunks)
    target_manifest.update(vector_count=len(vectors), keyword_count=len(chunks), status="draft",
                           artifact_path=str(artifact_root.relative_to(settings.project_root)))
    manifest_path = write_manifest(settings.project_root, target_manifest)
    persist_index_manifest(settings, target_manifest, target_manifest["artifact_path"])
    report = {
        "status": "built", "build_mode": "retire_rebuild" if exclude_retired else "incremental_reuse",
        "index_version_id": index_id,
        "base_index_version_id": active["index_version_id"],
        "processing_version_id": processing_ids[0],
        "processing_version_ids": processing_ids,
        "added_processing_version_ids": added_ids,
        "chunk_count": len(chunks), "vector_count": len(vectors),
        "keyword_count": len(chunks), "reused_vector_count": len(reused),
        "new_vector_count": len(generated), "embedding_model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions, "chroma": chroma, "bm25": bm25,
        "manifest_path": str(manifest_path), "http_requests": len(records),
        "new_external_api_calls": len(records),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = settings.project_root / "evals/results/index-build" / f"{index_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({**report, "records": records}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
