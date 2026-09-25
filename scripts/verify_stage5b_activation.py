"""Verify Stage 5B activation and map every visual candidate to active evidence."""
from __future__ import annotations

import argparse
import json

from multimodal_rag.application.visual_evidence import read_active_visual_element
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.settings import load_settings


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段5B激活后核验")
    parser.add_argument("--expected-index", required=True)
    parser.add_argument("--report", default="evals/results/stage5/stage5b_activation_verification.json")
    args = parser.parse_args()
    settings = load_settings()
    root = settings.project_root
    active = get_active_index(settings)
    scope = set((active.get("manifest") or {}).get("processing_version_ids") or [])
    build = json.loads((root / "evals/results/index-build" /
                        f"{args.expected_index}.json").read_text(encoding="utf-8"))
    candidates = _jsonl(root / "evals/datasets/retrieval_stage5_visual_candidates_v2.jsonl")
    assets = {row["asset_path"]: row for row in _jsonl(
        root / "data/annotations/tidb_zh/assets.jsonl") if row.get("asset_path")}
    with connection(settings, read_only=True) as conn:
        indexes = conn.execute(
            """SELECT index_version_id,status,chunk_count,vector_count,keyword_count
                 FROM mrag.index_versions ORDER BY updated_at DESC""").fetchall()
        approved = conn.execute(
            """SELECT count(*)::int FROM mrag.processing_versions
                WHERE processing_version_id=ANY(%s) AND quality_status='approved'""",
            (sorted(scope),)).fetchone()[0]
        released = conn.execute(
            """SELECT count(*)::int FROM mrag.processing_versions
                WHERE processing_version_id=ANY(%s) AND release_status='active'""",
            (sorted(scope),)).fetchone()[0]
        image_rows = conn.execute(
            """SELECT e.document_id,e.element_id,e.image_ref,e.source,c.chunk_id
                 FROM mrag.elements e
                 JOIN mrag.chunks c ON c.processing_version_id=e.processing_version_id
                  AND c.element_ids ? e.element_id
                WHERE e.processing_version_id=ANY(%s) AND e.kind='image'""",
            (sorted(scope),)).fetchall()
    image_index: dict[tuple, list[dict]] = {}
    for document_id, element_id, image_ref, source, chunk_id in image_rows:
        key = (document_id, image_ref, source.get("line_start"))
        image_index.setdefault(key, []).append({"element_id": element_id, "chunk_id": chunk_id})

    mappings = []
    for case in candidates:
        asset = assets[case["visual_evidence"]["asset_path"]]
        document_id = case["reference_document_ids"][0]
        matches = image_index.get((document_id, asset["target"], asset["line"]), [])
        unique_ids = sorted({item["element_id"] for item in matches})
        readable = []
        for element_id in unique_ids:
            content = read_active_visual_element(settings, element_id)
            readable.append(content["sha256"] == case["visual_evidence"]["sha256"])
        mappings.append({
            "case_id": case["case_id"], "document_id": document_id,
            "element_ids": unique_ids,
            "chunk_ids": sorted({item["chunk_id"] for item in matches}),
            "active_image_unique": len(unique_ids) == 1,
            "active_image_hash_matches": readable == [True],
        })
    active_rows = [row for row in indexes if row[1] == "active"]
    old_index = build["base_index_version_id"]
    checks = {
        "exactly_one_active_index": len(active_rows) == 1,
        "expected_index_is_active": active["index_version_id"] == args.expected_index,
        "counts_consistent": active["chunk_count"] == active["vector_count"]
                             == active["keyword_count"] == 1078,
        "scope_count_expected": len(scope) == 69,
        "all_scope_approved": approved == len(scope),
        "all_scope_released": released == len(scope),
        "previous_index_retired": any(row[0] == old_index and row[1] == "retired" for row in indexes),
        "all_visual_candidates_have_unique_active_image": all(
            row["active_image_unique"] for row in mappings),
        "all_active_image_hashes_match": all(
            row["active_image_hash_matches"] for row in mappings),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "active_index_version_id": active["index_version_id"],
        "previous_index_version_id": old_index,
        "chunk_count": active["chunk_count"], "vector_count": active["vector_count"],
        "keyword_count": active["keyword_count"], "processing_version_count": len(scope),
        "visual_candidate_count": len(candidates), "checks": checks,
        "visual_evidence_mappings": mappings,
        "external_api_calls": 0,
    }
    target = root / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "active_index_version_id", "previous_index_version_id", "chunk_count",
        "vector_count", "keyword_count", "processing_version_count",
        "visual_candidate_count", "checks", "external_api_calls")}, ensure_ascii=False, indent=2))
    print(f"report_path={target}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
