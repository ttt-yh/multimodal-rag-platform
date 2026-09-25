"""阶段4E：对既有问答结果执行答案要点检查和LLM语义评审。"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.application.semantic_evaluator import (
    build_judge_prompt,
    check_required_terms,
    parse_judge_result,
)
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.model_adapters import ChatAdapter
from multimodal_rag.infrastructure.settings import load_settings


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _used_citations(result: dict) -> list[dict]:
    citations = result.get("citations", [])
    numbers = {int(value) for value in re.findall(r"\[(\d+)\]", result.get("answer", ""))}
    return [citations[number - 1] for number in sorted(numbers)
            if 1 <= number <= len(citations)]


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段4E：答案语义质量开发评测")
    parser.add_argument("--dataset", default="evals/datasets/retrieval_stage3_text_gold_v1.jsonl")
    parser.add_argument("--facts", default="evals/datasets/qa_stage4_required_facts_v1.jsonl")
    parser.add_argument("--case", action="append",
                        help="只重跑指定 case_id，可重复传入；汇总仍基于磁盘上全部逐题报告")
    parser.add_argument("--confirm-live", action="store_true")
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("语义评测会为每条可回答题调用一次Chat评审，请提供 --confirm-live")

    settings = load_settings()
    root = settings.project_root
    cases = {item["case_id"]: item for item in _jsonl(root / args.dataset)
             if item.get("answerable") is True}
    annotations = {item["case_id"]: item for item in _jsonl(root / args.facts)}
    if set(cases) != set(annotations):
        parser.error("答案要点标注与可回答评测题编号不一致")
    if any(item.get("gold_eligible") is not True or item.get("review_status") != "approved"
           for item in cases.values()):
        parser.error("基础评测集包含未审核发布样本")
    if args.case:
        unknown = set(args.case) - set(cases)
        if unknown:
            parser.error(f"未找到 case_id：{', '.join(sorted(unknown))}")

    index_version_id = get_active_index(settings)["index_version_id"]
    qa_dir = root / "evals" / "results" / "stage4-qa" / index_version_id
    output_dir = root / "evals" / "results" / "stage4-semantic" / index_version_id
    output_dir.mkdir(parents=True, exist_ok=True)
    gateway = HttpGateway(settings, "chat", budget=CallBudget(len(cases)))
    details: list[dict] = []
    run_case_ids = [case_id for case_id in cases
                    if not args.case or case_id in set(args.case)]
    try:
        for case_id in run_case_ids:
            case = cases[case_id]
            qa_path = qa_dir / f"{case_id}.json"
            if not qa_path.is_file():
                details.append({"case_id": case_id, "status": "failed", "error_code": "qa_report_missing"})
                continue
            qa_report = json.loads(qa_path.read_text(encoding="utf-8"))
            result = qa_report.get("result", {})
            annotation = annotations[case_id]
            term_check = check_required_terms(result.get("answer", ""),
                                               annotation["required_term_groups"])
            record_start = len(gateway.records)
            try:
                response = ChatAdapter(gateway).complete(build_judge_prompt(
                    question=case["question"], reference_answer=case["reference_answer"],
                    answer=result.get("answer", ""), cited_evidence=_used_citations(result)))
                judge = parse_judge_result(response["text"])
                requires_review = bool(annotation.get("known_issue") or not judge["semantic_pass"]
                                       or term_check["coverage"] < 0.8)
                item = {
                    "case_id": case_id, "question": case["question"], "status": "completed",
                    "index_version_id": index_version_id, "term_check": term_check,
                    "judge": judge, "judge_model": response["model"],
                    "known_issue": annotation.get("known_issue"),
                    "human_review_required": requires_review,
                    "records": gateway.records[record_start:],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            except AppError as exc:
                item = {"case_id": case_id, "question": case["question"], "status": "failed",
                        "error_code": exc.code, "error": str(exc), "term_check": term_check,
                        "known_issue": annotation.get("known_issue"),
                        "human_review_required": True, "records": gateway.records[record_start:],
                        "created_at": datetime.now(timezone.utc).isoformat()}
            (output_dir / f"{case_id}.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
            details.append(item)
    finally:
        gateway.close()

    if args.case:
        details = []
        for case_id in cases:
            report_path = output_dir / f"{case_id}.json"
            if not report_path.is_file():
                details.append({"case_id": case_id, "status": "failed",
                                "error_code": "report_missing"})
                continue
            details.append(json.loads(report_path.read_text(encoding="utf-8")))

    completed = [item for item in details if item.get("status") == "completed"]
    summary = {
        "status": "completed" if len(completed) == len(cases) else "partial_failure",
        "evaluation_level": "development_llm_judge_pending_human_review",
        "sample_count": len(cases), "success_count": len(completed),
        "failure_count": len(cases) - len(completed),
        "required_term_coverage_average": round(
            sum(item["term_check"]["coverage"] for item in completed) / len(completed), 4) if completed else None,
        "correctness_average_4": round(
            sum(item["judge"]["correctness"] for item in completed) / len(completed), 4) if completed else None,
        "completeness_average_4": round(
            sum(item["judge"]["completeness"] for item in completed) / len(completed), 4) if completed else None,
        "groundedness_average_4": round(
            sum(item["judge"]["groundedness"] for item in completed) / len(completed), 4) if completed else None,
        "semantic_pass_rate": round(
            sum(bool(item["judge"]["semantic_pass"]) for item in completed) / len(completed), 4) if completed else None,
        "human_review_queue_count": sum(bool(item.get("human_review_required")) for item in details),
        "known_dataset_issue_count": sum(bool(item.get("known_issue")) for item in details),
        "external_api_calls": len(gateway.records), "index_version_id": index_version_id,
        "reports_dir": str(output_dir), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target = root / "evals" / "results" / "stage4" / "semantic_eval_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
