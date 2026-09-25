"""Evaluate deterministic title-aware image selection on dev Gold only."""
from __future__ import annotations

import json
import re
from pathlib import Path

from multimodal_rag.application.visual_evidence import locate_visual_evidence_for_title
from multimodal_rag.infrastructure.settings import load_settings


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals/datasets/retrieval_stage5_visual_gold_v2.jsonl"
TARGET = ROOT / "evals/results/stage5/visual_selection_dev_v2.json"
REVISIONS = ROOT / "evals/datasets/qa_stage5_visual_question_revisions_v2.jsonl"
TITLE = re.compile(r"《([^》]+)》")


def main() -> int:
    settings = load_settings()
    revisions = {row["case_id"]: row for row in
                 [json.loads(line) for line in REVISIONS.read_text(encoding="utf-8").splitlines()
                  if line.strip()]}
    cases = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()
             if line.strip() and json.loads(line).get("split") == "dev"]
    details = []
    for case in cases:
        question = revisions.get(case["case_id"], {}).get("proposed_question", case["question"])
        match = TITLE.search(question)
        if not match:
            raise ValueError(f"dev问题缺少文档标题：{case['case_id']}")
        result = locate_visual_evidence_for_title(
            settings, match.group(1), question, max_images=3)
        selected = {row["element_id"] for row in result["candidates"]
                    if row.get("vision_eligible")}
        gold = case["visual_gold"]
        groups = [{row["element_id"]} for row in gold.get("image_elements", [])]
        groups += [{row["element_id"] for row in group}
                   for group in gold.get("alternative_image_groups", [])]
        excluded = {row["element_id"] for row in gold.get("excluded_image_elements", [])}
        recall = sum(bool(group & selected) for group in groups) / len(groups) if groups else 1.0
        details.append({
            "case_id": case["case_id"], "selected_element_ids": sorted(selected),
            "question_revision_simulated": case["case_id"] in revisions,
            "visual_group_recall": recall,
            "excluded_selected": sorted(excluded & selected),
            "scores": [{"element_id": row["element_id"],
                        "score": row.get("visual_relevance_score")}
                       for row in result["candidates"] if row.get("vision_eligible")],
        })
    report = {
        "status": "completed", "split": "dev", "sample_count": len(details),
        "visual_group_recall": sum(row["visual_group_recall"] for row in details) / len(details),
        "excluded_selection_violation_rate": sum(bool(row["excluded_selected"]) for row in details) / len(details),
        "failed_case_ids": [row["case_id"] for row in details
                            if row["visual_group_recall"] < 1 or row["excluded_selected"]],
        "external_api_calls": 0, "test_accessed": False,
        "pending_question_revision_simulation": True, "details": details,
    }
    TARGET.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "split", "sample_count", "visual_group_recall",
        "excluded_selection_violation_rate", "failed_case_ids",
        "external_api_calls", "test_accessed")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
