"""将阶段3扩展中文 Markdown 清单写入 PostgreSQL 并持久化结构化 Chunk。

本脚本只做本地解析、数据库入库和切分，不调用 Embedding、Rerank 或 Chat API，
也不会修改任何 active 索引。重复执行依靠现有的文档版本、处理版本、任务和 Chunk
唯一约束保持幂等；后续阶段可使用报告中的 processing_version_id 构建 draft 索引。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.application.ingestion_worker import enqueue_document, run_once
from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.chunking import split_elements
from multimodal_rag.infrastructure.chunk_repository import persist_chunks
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import load_settings


MAX_CHARACTERS = 1200
OVERLAP_CHARACTERS = 160
MIN_CHARACTERS = 100


def _load_manifest(root: Path, relative_path: str) -> list[dict]:
    path = (root / relative_path).resolve()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("document_id"):
                raise ValueError("阶段3白名单包含无效记录")
            rows.append(row)
    if not rows:
        raise ValueError("阶段3白名单为空")
    ids = [row["document_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("阶段3白名单存在重复 document_id")
    return rows


def _persisted_processing(settings, processing_ids: list[str]) -> dict[str, dict]:
    if not processing_ids:
        return {}
    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            """SELECT processing_version_id,quality_status,release_status,element_count,warning_count
               FROM mrag.processing_versions WHERE processing_version_id = ANY(%s)""",
            (processing_ids,),
        ).fetchall()
    return {
        row[0]: {"quality_status": row[1], "release_status": row[2],
                 "element_count": row[3], "warning_count": row[4]}
        for row in rows
    }


def _chunk_quality(settings, processing_ids: list[str]) -> dict:
    if not processing_ids:
        return {"chunk_count": 0, "empty_count": 0, "short_count": 0,
                "duplicate_count": 0, "missing_source_count": 0}
    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            """SELECT chunk_id,text,element_ids,source_locations,estimated_length
               FROM mrag.chunks WHERE processing_version_id = ANY(%s)
               ORDER BY processing_version_id,ordinal""",
            (processing_ids,),
        ).fetchall()
    ids = [row[0] for row in rows]
    return {
        "chunk_count": len(rows),
        "empty_count": sum(not (row[1] or "").strip() for row in rows),
        "short_count": sum((row[4] or 0) < MIN_CHARACTERS for row in rows),
        "duplicate_count": len(ids) - len(set(ids)),
        "missing_source_count": sum(not row[2] or not row[3] for row in rows),
    }


def _reusable_approved_processing(settings, document_id: str) -> dict | None:
    """Reuse a reviewed non-empty processing version for the same source document."""
    with connection(settings, read_only=True) as conn:
        row = conn.execute(
            """SELECT pv.processing_version_id,pv.element_count,count(c.chunk_id)
                 FROM mrag.processing_versions pv
                 JOIN mrag.chunks c ON c.processing_version_id=pv.processing_version_id
                WHERE pv.document_id=%s AND pv.quality_status='approved'
                GROUP BY pv.processing_version_id,pv.element_count,pv.release_status,pv.created_at
                HAVING count(c.chunk_id)>0
                ORDER BY CASE pv.release_status WHEN 'active' THEN 0 WHEN 'draft' THEN 1 ELSE 2 END,
                         pv.created_at DESC
                LIMIT 1""",
            (document_id,),
        ).fetchone()
    if not row:
        return None
    return {"document_id": document_id, "status": "succeeded",
            "processing_version_id": row[0], "element_count": row[1],
            "chunk_count": row[2], "chunks_persisted": 0,
            "idempotent": True, "reused_approved_processing": True}


def _process_document(settings, document_id: str, profile: dict, worker_id: str) -> dict:
    queued = enqueue_document(settings, document_id, profile=profile)
    # 按当前入队任务领取，避免旧队列任务改变当前文档的处理结果。
    result = run_once(settings, worker_id, job_id=queued["job_id"])
    if result is not None and result.get("status") == "failed":
        return {"document_id": document_id, "status": "failed",
                "error_code": result.get("error_code", "worker_error"),
                "processing_version_id": queued["processing_version_id"], "chunks": 0}

    processing_id = queued["processing_version_id"]
    # worker 已经负责保存文档、版本、元素；这里把同一处理版本的元素映射为
    # 数据库中的稳定 element_id，再交给结构切分器生成可追溯 Chunk。
    preview = preview_document(document_id, settings)
    elements = [element.model_copy(update={
        "element_id": "el_" + hashlib.sha256(f"{processing_id}:{element.order}".encode()).hexdigest()[:24],
        "processing_version_id": processing_id,
    }) for element in preview.elements]
    chunks = split_elements(elements, processing_id,
                            max_characters=MAX_CHARACTERS,
                            overlap_characters=OVERLAP_CHARACTERS,
                            min_characters=MIN_CHARACTERS)
    persisted = persist_chunks(settings, chunks)
    return {"document_id": document_id, "status": "succeeded",
            "processing_version_id": processing_id, "element_count": len(elements),
            "chunk_count": len(chunks), "chunks_persisted": persisted,
            "idempotent": bool(result and result.get("job", {}).get("idempotent"))}


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段3B：扩展文本入库与结构化 Chunk 持久化")
    parser.add_argument("--manifest", default="data/manifests/ingestion_chinese_md_stage3.jsonl",
                        help="项目根目录下 data/manifests 中的 JSONL 白名单")
    parser.add_argument("--worker-id", default="stage3b-local-worker")
    parser.add_argument("--report", default="evals/results/stage3/stage3b_ingestion_report.json")
    parser.add_argument("--reuse-approved", action="store_true",
                        help="同一文档已有通过审核且非空的处理版本时直接复用")
    args = parser.parse_args()

    settings = load_settings().model_copy(update={"ingestion_manifest": args.manifest})
    rows = _load_manifest(settings.project_root, args.manifest)
    profile = {"worker_version": "local-text-worker-v1", "parser_version": "local-text-preview-v1",
               "chunking_version": "structure-recursive-v1", "max_characters": MAX_CHARACTERS,
               "overlap_characters": OVERLAP_CHARACTERS, "min_characters": MIN_CHARACTERS,
               "manifest": args.manifest}
    started = time.perf_counter()
    results = []
    for row in rows:
        try:
            reusable = (_reusable_approved_processing(settings, row["document_id"])
                        if args.reuse_approved else None)
            results.append(reusable or _process_document(
                settings, row["document_id"], profile, args.worker_id))
        except Exception as exc:  # 单文档失败只影响当前文档，报告保留可诊断信息
            results.append({"document_id": row["document_id"], "status": "failed",
                            "error_code": type(exc).__name__, "error": str(exc), "chunks": 0})

    processing_ids = [item["processing_version_id"] for item in results
                      if item.get("processing_version_id")]
    processing = _persisted_processing(settings, processing_ids)
    quality = _chunk_quality(settings, processing_ids)
    report = {
        "stage": "3B",
        "status": "succeeded" if all(item["status"] == "succeeded" for item in results) else "partial_failure",
        "manifest": args.manifest,
        "document_count": len(rows),
        "success_count": sum(item["status"] == "succeeded" for item in results),
        "failure_count": sum(item["status"] != "succeeded" for item in results),
        "element_count": sum(item.get("element_count", 0) for item in results),
        "chunk_count": quality["chunk_count"],
        "processing_version_count": len(processing),
        "reused_approved_count": sum(bool(item.get("reused_approved_processing"))
                                       for item in results),
        "processing_versions": processing,
        "chunk_quality": quality,
        "parameters": {"max_characters": MAX_CHARACTERS, "overlap_characters": OVERLAP_CHARACTERS,
                        "min_characters": MIN_CHARACTERS},
        "external_api_calls": 0,
        "active_index_changed": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    target = settings.project_root / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "document_count", "success_count", "failure_count", "element_count",
        "chunk_count", "processing_version_count", "chunk_quality", "external_api_calls",
        "active_index_changed", "elapsed_seconds")}, ensure_ascii=False, indent=2))
    print(f"report_path={target}")
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
