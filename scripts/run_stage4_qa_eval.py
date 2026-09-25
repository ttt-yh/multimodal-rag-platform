"""阶段4A：对已审核的文本问答和无答案问题执行真实RAG问答评测。"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.application.qa_service import answer
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.settings import load_settings


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cited_numbers(text: str) -> set[int]:
    """Extract only explicit evidence references such as [1] from the answer."""
    return {int(value) for value in re.findall(r"\[(\d+)\]", text or "")}


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段4A：真实RAG问答评测")
    parser.add_argument("--dataset", default="evals/datasets/retrieval_stage3_text_gold_v1.jsonl")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--only-failed", action="store_true",
                        help="只重试上一次报告中失败的题，避免重复调用已成功样本")
    parser.add_argument("--case", action="append",
                        help="只重跑指定 case_id，可重复传入；汇总仍基于全部样本")
    parser.add_argument("--report-only", action="store_true",
                        help="不调用模型，只基于当前索引已有逐题结果重新汇总报告")
    args = parser.parse_args()
    if not args.confirm_live and not args.report_only:
        parser.error("问答评测会调用Chat/Embedding/Rerank，请提供 --confirm-live")
    if args.only_failed and args.report_only:
        parser.error("--only-failed 与 --report-only 不能同时使用")
    if args.case and (args.only_failed or args.report_only):
        parser.error("--case 不能与 --only-failed/--report-only 同时使用")

    settings = load_settings()
    root = settings.project_root
    source_cases = _rows(root / args.dataset)
    unreleased = [row.get("case_id") for row in source_cases
                  if row.get("gold_eligible") is not True or row.get("review_status") != "approved"]
    if unreleased:
        parser.error(f"数据集包含未审核发布的样本：{', '.join(str(item) for item in unreleased[:5])}")
    cases = [row for row in source_cases if row.get("type") != "visual"]
    if not cases:
        parser.error("评测集中没有可执行的文本样本")
    active_index = get_active_index(settings)
    index_version_id = active_index["index_version_id"]
    report_dir = root / "evals/results/stage4-qa" / index_version_id
    report_dir.mkdir(parents=True, exist_ok=True)
    execution_cases = list(cases)
    if args.only_failed:
        failed_ids = set()
        for path in report_dir.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError):
                continue
            if item.get("status") == "failed":
                failed_ids.add(item.get("case_id"))
        execution_cases = [case for case in cases if case.get("case_id") in failed_ids]
        if not execution_cases:
            parser.error("没有找到可重试的失败样本")
    if args.report_only:
        execution_cases = []
    if args.case:
        selected = set(args.case)
        unknown = selected - {case["case_id"] for case in cases}
        if unknown:
            parser.error(f"未找到 case_id：{', '.join(sorted(unknown))}")
        execution_cases = [case for case in cases if case["case_id"] in selected]
    for case in execution_cases:
        started = datetime.now(timezone.utc)
        try:
            result = answer(settings, case["question"], max_requests=3)
            expected = set(case.get("evidence_chunk_ids", []))
            citations = result.get("citations", [])
            cited_numbers = _cited_numbers(result.get("answer", ""))
            available_numbers = set(range(1, len(citations) + 1))
            valid_numbers = cited_numbers.intersection(available_numbers)
            actual = {citations[number - 1].get("chunk_id") for number in valid_numbers
                      if citations[number - 1].get("chunk_id")}
            matched = expected.intersection(actual)
            if case.get("answerable") is True:
                check = {"citation_recall": len(matched) / len(expected) if expected else 0.0,
                         "expected_citation_count": len(expected), "matched_citation_count": len(matched),
                         "cited_numbers": sorted(cited_numbers),
                         "invalid_cited_numbers": sorted(cited_numbers - available_numbers),
                         "citation_validity": bool(cited_numbers) and cited_numbers <= available_numbers,
                         "full_evidence_coverage": bool(expected) and expected <= actual,
                         "refusal_check": None}
            else:
                text = result.get("answer", "")
                check = {"citation_recall": None, "expected_citation_count": 0,
                         "matched_citation_count": 0,
                         "cited_numbers": sorted(cited_numbers),
                         "invalid_cited_numbers": sorted(cited_numbers - available_numbers),
                         "citation_validity": cited_numbers <= available_numbers,
                         "full_evidence_coverage": None,
                         "refusal_check": any(token in text for token in ("无法", "不能", "不足", "没有足够", "需要明确"))}
            item = {"case_id": case["case_id"], "question": case["question"],
                    "answerable": case.get("answerable"), "result": result, "checks": check,
                    "started_at": started.isoformat()}
        except (AppError, RuntimeError) as exc:
            item = {"case_id": case["case_id"], "question": case["question"],
                    "answerable": case.get("answerable"), "status": "failed",
                    "error_code": getattr(exc, "code", type(exc).__name__),
                    "error": str(exc), "started_at": started.isoformat()}
        (report_dir / f"{case['case_id']}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")

    details = []
    for case in cases:
        path = report_dir / f"{case['case_id']}.json"
        if not path.exists():
            details.append({"case_id": case["case_id"], "question": case["question"],
                            "answerable": case.get("answerable"), "status": "missing_report"})
            continue
        details.append(json.loads(path.read_text(encoding="utf-8")))

    answerable = [item for item in details if item.get("answerable") is True and "checks" in item]
    unanswerable = [item for item in details if item.get("answerable") is not True and "checks" in item]
    summary = {
        "status": "completed" if all("checks" in item for item in details) else "partial_failure",
        "sample_count": len(details), "success_count": sum("checks" in item for item in details),
        "failure_count": sum("checks" not in item for item in details),
        "answerable_count": len(answerable), "unanswerable_count": len(unanswerable),
        "citation_recall_average": round(sum(item["checks"]["citation_recall"] for item in answerable) / len(answerable), 4) if answerable else None,
        "citation_validity_rate": round(sum(bool(item["checks"]["citation_validity"]) for item in answerable) / len(answerable), 4) if answerable else None,
        "full_evidence_coverage_rate": round(sum(bool(item["checks"]["full_evidence_coverage"]) for item in answerable) / len(answerable), 4) if answerable else None,
        "unanswerable_refusal_rate": round(sum(bool(item["checks"]["refusal_check"]) for item in unanswerable) / len(unanswerable), 4) if unanswerable else None,
        "index_version_id": index_version_id,
        "external_api_calls": sum(item.get("result", {}).get("external_calls", 0) for item in details),
        "reports_dir": str(report_dir), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target = root / "evals/results/stage4/qa_eval_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
