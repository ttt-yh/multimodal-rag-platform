"""Bind Stage 5 visual candidates to active Chunk and image Element IDs."""
from __future__ import annotations

import json
from pathlib import Path

from prepare_stage5_visual_expansion import _jsonl, _review_page, _write_jsonl


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evals/datasets/retrieval_stage5_visual_candidates_v2.jsonl"
ACTIVATION = ROOT / "evals/results/stage5/stage5b_activation_verification.json"
TARGET = ROOT / "evals/datasets/retrieval_stage5_visual_mapped_v2.jsonl"
REVIEW = ROOT / "evals/results/stage5/visual_gold_v2_review.html"
REPORT = ROOT / "evals/results/stage5/visual_mapping_summary.json"


def main() -> int:
    rows = _jsonl(SOURCE)
    activation = json.loads(ACTIVATION.read_text(encoding="utf-8"))
    if activation.get("status") != "passed":
        raise ValueError("活动索引核验未通过，不能发布证据映射")
    mappings = {row["case_id"]: row for row in activation["visual_evidence_mappings"]}
    if set(mappings) != {row["case_id"] for row in rows}:
        raise ValueError("候选与活动证据映射编号不一致")
    output = []
    for source in rows:
        row = dict(source)
        mapping = mappings[row["case_id"]]
        if (not mapping["active_image_unique"] or not mapping["active_image_hash_matches"]
                or len(mapping["element_ids"]) != 1 or len(mapping["chunk_ids"]) != 1):
            raise ValueError(f"{row['case_id']}没有唯一且完整的活动图片证据")
        row.update({
            "index_version_id": activation["active_index_version_id"],
            "chunk_ids": mapping["chunk_ids"],
            "evidence_chunk_ids": mapping["chunk_ids"],
            "mapping_status": "mapped_to_active_visual_element",
            "active_visual_mapping": {
                "element_ids": mapping["element_ids"],
                "chunk_ids": mapping["chunk_ids"],
                "asset_path": row["visual_evidence"]["asset_path"],
                "sha256": row["visual_evidence"]["sha256"],
            },
            # Mapping success is not a review decision.
            "review_status": "pending_human_review",
            "gold_eligible": False,
            "evaluation_status": "mapped_visual_candidate_v2_pending_human_review",
        })
        output.append(row)
    _write_jsonl(TARGET, output)
    REVIEW.write_text(_review_page(output), encoding="utf-8")
    report = {
        "status": "mapped_pending_human_review",
        "index_version_id": activation["active_index_version_id"],
        "candidate_count": len(output), "mapped_count": len(output),
        "unique_image_element_count": len({
            row["active_visual_mapping"]["element_ids"][0] for row in output}),
        "unique_chunk_count": len({
            row["active_visual_mapping"]["chunk_ids"][0] for row in output}),
        "gold_eligible_count": sum(bool(row["gold_eligible"]) for row in output),
        "external_api_calls": 0,
        "review_page": REVIEW.relative_to(ROOT).as_posix(),
        "mapped_dataset": TARGET.relative_to(ROOT).as_posix(),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
