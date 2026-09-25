"""Offline retrieval evaluation over reviewed question/evidence records.

The evaluator is deliberately independent of model providers. It consumes saved
retrieval reports, so running it never spends Embedding or Rerank quota.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_gold_cases(path: Path, *, require_approved: bool = True) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            case = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"评测集第 {line_number} 行不是有效 JSON: {exc.msg}") from exc
        _validate_case(case, line_number)
        if case["case_id"] in seen:
            raise ValueError(f"评测集存在重复 case_id: {case['case_id']}")
        seen.add(case["case_id"])
        if require_approved and not (case["review_status"] == "approved" and case["gold_eligible"] is True):
            continue
        cases.append(case)
    if require_approved and not cases:
        raise ValueError("没有审核通过且 gold_eligible=true 的评测样本；不能生成正式指标")
    return cases


def summarize_gold_dataset(path: Path) -> dict[str, Any]:
    """Return an auditable readiness summary without claiming model quality."""
    cases = load_gold_cases(path, require_approved=False)
    approved = [case for case in cases if case["review_status"] == "approved" and case["gold_eligible"] is True]
    return {
        "dataset": path.stem,
        "status": "approved" if len(approved) == len(cases) and cases else "pending_human_review",
        "sample_count": len(cases),
        "answerable_count": sum(bool(case["answerable"]) for case in cases),
        "unanswerable_count": sum(not case["answerable"] for case in cases),
        "approved_count": len(approved),
        "quality_metrics": None,
        "note": "评测集已准备，但仍有样本未完成人工审核；暂不计算正式 Recall、Precision 或 MRR。" if len(approved) != len(cases) else "评测集已审核，可执行离线检索评测。",
    }


def _validate_case(case: dict[str, Any], line_number: int) -> None:
    required = {"case_id", "question", "reference_answer", "answerable", "evidence_chunk_ids",
                "review_status", "gold_eligible"}
    missing = required - case.keys()
    if missing:
        raise ValueError(f"评测集第 {line_number} 行缺少字段: {sorted(missing)}")
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError(f"评测集第 {line_number} 行 case_id 无效")
    if not isinstance(case["question"], str) or not case["question"].strip():
        raise ValueError(f"评测集第 {line_number} 行 question 为空")
    if not isinstance(case["evidence_chunk_ids"], list) or any(not isinstance(item, str) for item in case["evidence_chunk_ids"]):
        raise ValueError(f"评测集第 {line_number} 行 evidence_chunk_ids 无效")
    if case["answerable"] and not case["evidence_chunk_ids"]:
        raise ValueError(f"评测集第 {line_number} 行可回答问题必须标注证据 Chunk")
    if not case["answerable"] and case["evidence_chunk_ids"]:
        raise ValueError(f"评测集第 {line_number} 行无答案问题不能标注证据 Chunk")


def score_retrieval_case(case: dict[str, Any], report: dict[str, Any], *, k: int = 5) -> dict[str, Any]:
    if k < 1:
        raise ValueError("k 必须大于 0")
    result_ids = [str(row.get("chunk_id")) for row in report.get("results", [])[:k]
                  if isinstance(row, dict) and row.get("chunk_id")]
    expected = set(case["evidence_chunk_ids"])
    if not case["answerable"]:
        return {"case_id": case["case_id"], "answerable": False, "retrieved_count": len(result_ids),
                "matched_count": 0, "reciprocal_rank": None}
    matched = expected.intersection(result_ids)
    first_rank = next((index + 1 for index, chunk_id in enumerate(result_ids) if chunk_id in expected), None)
    return {"case_id": case["case_id"], "answerable": True, "retrieved_count": len(result_ids),
            "relevant_count": len(expected), "matched_count": len(matched),
            "recall_at_k": len(matched) / len(expected),
            "precision_at_k": len(matched) / len(result_ids) if result_ids else 0.0,
            "reciprocal_rank": 1 / first_rank if first_rank else 0.0}


def score_retrieval(cases: list[dict[str, Any]], reports: list[dict[str, Any]], *, k: int = 5) -> dict[str, Any]:
    by_query = {str(report.get("query", "")): report for report in reports}
    details: list[dict[str, Any]] = []
    missing: list[str] = []
    for case in cases:
        report = by_query.get(case["question"])
        if report is None:
            missing.append(case["case_id"])
            continue
        details.append(score_retrieval_case(case, report, k=k))
    answerable = [item for item in details if item["answerable"]]
    if not answerable:
        raise ValueError("没有匹配到可回答评测样本，不能计算 Recall/Precision/MRR")
    return {
        "status": "approved",
        "note": "离线检索评测已完成，以下指标来自审核通过的问题与证据标注。",
        "k": k,
        "sample_count": len(details),
        "answerable_count": len(answerable),
        "unanswerable_count": sum(not item["answerable"] for item in details),
        "missing_report_case_ids": missing,
        "metrics": {
            "recall_at_k": sum(item["recall_at_k"] for item in answerable) / len(answerable),
            "precision_at_k": sum(item["precision_at_k"] for item in answerable) / len(answerable),
            "mrr": sum(item["reciprocal_rank"] for item in answerable) / len(answerable),
        },
        "cases": details,
    }
