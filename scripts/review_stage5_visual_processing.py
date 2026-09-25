"""Audit and optionally approve Stage 5B processing versions.

This is a document-processing quality review, not a visual QA Gold review.  It
compares persisted elements against a fresh deterministic parse, validates
Chunk invariants, and proves that each designated original image is linked to
at least one Chunk.  It does not call any model or change an active index.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from multimodal_rag.application.preview import preview_document
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.review_repository import record_quality_review
from multimodal_rag.infrastructure.settings import load_settings


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _normalize(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段5B处理版本质量审核")
    parser.add_argument("--apply-approved", action="store_true",
                        help="仅将所有检查通过且仍待审的处理版本登记为approved")
    parser.add_argument("--reviewer", default="agent-assisted-quality-review")
    parser.add_argument("--report", default="evals/results/stage5/stage5b_processing_review.json")
    args = parser.parse_args()

    settings = load_settings().model_copy(update={
        "ingestion_manifest": "data/manifests/ingestion_chinese_visual_stage5.jsonl"})
    root = settings.project_root
    ingestion = json.loads((root / "evals/results/stage5/stage5b_visual_ingestion_report.json")
                           .read_text(encoding="utf-8"))
    candidates = {row["reference_document_ids"][0]: row for row in _jsonl(
        root / "evals/datasets/retrieval_stage5_visual_candidates_v2.jsonl")}
    assets = {row["asset_path"]: row for row in _jsonl(
        root / "data/annotations/tidb_zh/assets.jsonl") if row.get("asset_path")}
    processing_by_document = {
        row["document_id"]: row["processing_version_id"] for row in ingestion["results"]
        if row.get("status") == "succeeded"
    }
    if set(processing_by_document) != set(candidates):
        raise ValueError("入库报告与视觉候选所含文档不一致")

    audits: list[dict] = []
    for document_id, case in candidates.items():
        processing_id = processing_by_document[document_id]
        preview = preview_document(document_id, settings)
        expected = [{
            "element_id": "el_" + hashlib.sha256(
                f"{processing_id}:{element.order}".encode()).hexdigest()[:24],
            "kind": element.kind, "ordinal": element.order,
            "raw_text": element.raw_text, "heading_path": element.heading_path,
            "source": element.source.model_dump(), "image_ref": element.image_ref,
        } for element in preview.elements]
        with connection(settings, read_only=True) as conn:
            pv = conn.execute(
                """SELECT quality_status,release_status,element_count,warning_count
                     FROM mrag.processing_versions WHERE processing_version_id=%s""",
                (processing_id,),).fetchone()
            persisted = conn.execute(
                """SELECT element_id,kind,ordinal,raw_text,heading_path,source,image_ref
                     FROM mrag.elements WHERE processing_version_id=%s ORDER BY ordinal""",
                (processing_id,),).fetchall()
            chunks = conn.execute(
                """SELECT chunk_id,ordinal,text,heading_path,element_ids,source_locations,
                          estimated_length
                     FROM mrag.chunks WHERE processing_version_id=%s ORDER BY ordinal""",
                (processing_id,),).fetchall()
        actual = [{"element_id": row[0], "kind": row[1], "ordinal": row[2],
                   "raw_text": row[3], "heading_path": row[4], "source": row[5],
                   "image_ref": row[6]} for row in persisted]
        element_ids = {row["element_id"] for row in actual}
        chunk_element_ids = {element_id for row in chunks for element_id in row[4]}

        asset = assets.get(case["visual_evidence"]["asset_path"])
        image_matches = [row for row in actual if asset and row["kind"] == "image"
                         and row["image_ref"] == asset["target"]
                         and row["source"].get("line_start") == asset["line"]]
        designated_ids = {row["element_id"] for row in image_matches}
        warnings_allowed = all(
            warning.startswith(("0A 仅进行结构预览", "已跳过 YAML 文档头",
                                "存在 HTML/模板内容"))
            for warning in preview.warnings)
        checks = {
            "processing_version_exists": pv is not None,
            "element_count_matches": bool(pv and pv[2] == len(expected) == len(actual)),
            "elements_exactly_match_fresh_parse": _normalize(actual) == _normalize(expected),
            "element_ordinals_contiguous": [row["ordinal"] for row in actual] == list(range(len(actual))),
            "chunks_non_empty": bool(chunks) and all((row[2] or "").strip() for row in chunks),
            "chunk_ordinals_contiguous": [row[1] for row in chunks] == list(range(len(chunks))),
            "chunk_lengths_match": all(row[6] == len(row[2]) for row in chunks),
            "chunk_element_references_valid": chunk_element_ids <= element_ids,
            "chunk_sources_present": all(bool(row[5]) for row in chunks),
            "designated_original_image_found": len(image_matches) == 1,
            "designated_image_linked_to_chunk": bool(designated_ids & chunk_element_ids),
            "original_image_hash_matches": bool(asset and
                hashlib.sha256((root / asset["asset_path"]).read_bytes()).hexdigest()
                == asset["asset_sha256"]),
            "warnings_in_informational_allowlist": warnings_allowed,
        }
        passed = all(checks.values())
        audits.append({
            "case_id": case["case_id"], "document_id": document_id,
            "processing_version_id": processing_id,
            "quality_status_before": pv[0] if pv else None,
            "release_status": pv[1] if pv else None,
            "element_count": len(actual), "chunk_count": len(chunks),
            "image_element_ids": sorted(designated_ids),
            "warnings": preview.warnings, "checks": checks, "passed": passed,
        })

    applied = []
    failed = [row for row in audits if not row["passed"]]
    if args.apply_approved and not failed:
        for row in audits:
            if row["quality_status_before"] == "approved":
                continue
            notes = (
                "阶段5B文档处理质量审核通过：重新解析结果与持久化元素逐项一致；"
                f"{row['element_count']}个元素、{row['chunk_count']}个Chunk非空且来源完整；"
                "指定原图Element已定位并关联Chunk，资源SHA256一致；解析警告均为信息性。"
                "本结论仅批准文档处理质量，不代表视觉问答Gold审核通过。"
            )
            applied.append(record_quality_review(
                settings, row["processing_version_id"], args.reviewer, "approved", notes))

    report = {
        "status": "approved" if args.apply_approved and not failed else
                  ("audit_passed" if not failed else "audit_failed"),
        "document_count": len(audits), "passed_count": len(audits) - len(failed),
        "failed_count": len(failed),
        "quality_status_before": dict(Counter(row["quality_status_before"] for row in audits)),
        "review_records_created": len(applied), "reviews": applied,
        "external_api_calls": 0, "active_index_changed": False,
        "visual_gold_changed": False,
        "scope_note": "仅审核文档解析、切分、来源和图片关联；不审核视觉问题与答案。",
        "audits": audits,
    }
    target = root / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "document_count", "passed_count", "failed_count",
        "quality_status_before", "review_records_created", "external_api_calls",
        "active_index_changed", "visual_gold_changed")}, ensure_ascii=False, indent=2))
    print(f"report_path={target}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
