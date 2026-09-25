"""阶段5：使用人工发布的视觉Gold执行正式、可复现的图文问答评测。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.application.multimodal_qa_service import answer_multimodal
from multimodal_rag.application.semantic_evaluator import check_required_terms
from multimodal_rag.application.visual_evaluator import score_visual_answer
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.settings import load_settings


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段5正式视觉问答评测")
    parser.add_argument("--dataset", default="evals/datasets/retrieval_stage5_visual_gold_v2_4.jsonl")
    parser.add_argument("--facts", default="evals/datasets/qa_stage5_visual_facts_v2_4.jsonl")
    parser.add_argument("--split", choices=("dev", "test"), required=True,
                        help="dev用于开发验证；test只允许方案冻结后的最终评测")
    parser.add_argument("--plan", action="store_true", help="只校验Gold并打印调用预算")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--max-external-requests", type=int, default=0,
                        help="本轮外部请求总预算，每题最多3次")
    parser.add_argument("--confirm-test-final", action="store_true",
                        help="明确确认test只运行本次最终评测且不用于调参")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="不调用模型，仅按当前Gold重新判定已有逐题报告")
    parser.add_argument("--reuse-from-version", choices=("v2_1", "v2_2", "v2_3"),
                        help="复用上一版本逐题模型结果，并按当前Gold重新判定")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    if not args.plan and not args.confirm_live and not args.reuse_existing:
        parser.error("视觉问答每题最多调用Embedding、Rerank、Vision各一次，请提供 --confirm-live")

    settings = load_settings()
    root = settings.project_root
    cases = {row["case_id"]: row for row in _jsonl(root / args.dataset)
             if row.get("type") == "visual" and row.get("split") == args.split}
    annotations = {row["case_id"]: row for row in _jsonl(root / args.facts)
                   if row.get("split") == args.split}
    if not cases:
        parser.error("正式视觉Gold中没有可评测样本")
    if set(cases) != set(annotations):
        parser.error("视觉题与必要答案要点编号不一致")
    invalid_cases = [case_id for case_id, row in cases.items() if not (
        row.get("review_status") == "approved"
        and row.get("gold_eligible") is True
        and row.get("evaluation_status") == "released_visual_gold_v2_4"
        and row.get("visual_review_status") == "human_released_gold_v2_4"
        and isinstance(row.get("visual_gold"), dict)
    )]
    invalid_annotations = [case_id for case_id, row in annotations.items() if not (
        row.get("review_status") == "approved" and row.get("gold_eligible") is True
    )]
    if invalid_cases or invalid_annotations:
        parser.error(f"正式Gold门禁失败：cases={invalid_cases}, facts={invalid_annotations}")
    selected = args.case or list(cases)
    if any(case_id not in cases for case_id in selected):
        parser.error("--case包含不存在的视觉题编号")
    if args.split == "test" and args.case:
        parser.error("test必须一次性运行全部样本，不能按case选择")
    if args.split == "test" and not args.plan and not args.reuse_existing and not args.confirm_test_final:
        parser.error("运行test前必须提供 --confirm-test-final")
    required_budget = len(selected) * 3
    if args.plan:
        print(json.dumps({"status": "ready", "split": args.split,
                          "sample_count": len(selected),
                          "max_external_requests_required": required_budget,
                          "test_final_confirmation_required": args.split == "test",
                          "external_api_calls": 0}, ensure_ascii=False, indent=2))
        return 0
    if not args.reuse_existing and args.max_external_requests < required_budget:
        parser.error(f"预算不足：{len(selected)}条样本最多需要{required_budget}次外部请求")
    index_version_id = get_active_index(settings)["index_version_id"]
    output_dir = root / "evals" / "results" / "stage5-visual-formal" / "v2_4" / args.split / index_version_id
    reuse_dir = (root / "evals" / "results" / "stage5-visual-formal" /
                 args.reuse_from_version / args.split / index_version_id
                 if args.reuse_from_version else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    details = []
    for case_id in selected:
        case, annotation = cases[case_id], annotations[case_id]
        started = datetime.now(timezone.utc)
        try:
            report_path = output_dir / f"{case_id}.json"
            if args.reuse_existing:
                source_report = (report_path if report_path.is_file() else
                                 reuse_dir / f"{case_id}.json" if reuse_dir else report_path)
                if not source_report.is_file():
                    raise RuntimeError("没有可复用的正式逐题报告")
                previous = json.loads(source_report.read_text(encoding="utf-8"))
                if previous.get("status") != "completed" or not isinstance(previous.get("result"), dict):
                    raise RuntimeError("已有逐题报告未成功完成，不能复用")
                result = previous["result"]
            else:
                result = answer_multimodal(settings, case["question"], max_images=3, max_requests=3)
            terms = check_required_terms(result.get("answer", ""), annotation["required_term_groups"])
            score = score_visual_answer(case, annotation, result, terms)
            item = {"case_id": case_id, "question": case["question"], "status": "completed",
                    "evaluation_level": "formal_human_reviewed_visual_gold_v2_4",
                    "result": result, "term_check": terms, "formal_score": score,
                    "known_issue": annotation.get("known_issue"),
                    "gold_release": {"review_status": case["review_status"],
                                     "evaluation_status": case["evaluation_status"],
                                     "visual_review_status": case["visual_review_status"]},
                    "started_at": started.isoformat()}
        except (AppError, RuntimeError) as exc:
            item = {"case_id": case_id, "question": case["question"], "status": "failed",
                    "error_code": getattr(exc, "code", type(exc).__name__), "error": str(exc),
                    "started_at": started.isoformat()}
        (output_dir / f"{case_id}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        details.append(item)

    details = []
    for case_id in cases:
        path = output_dir / f"{case_id}.json"
        if path.is_file():
            details.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            details.append({"case_id": case_id, "status": "missing_report"})
    completed = [item for item in details if item.get("status") == "completed"]
    scored = [item["formal_score"] for item in completed]
    summary = {
        "status": "completed" if len(completed) == len(cases) else "partial_failure",
        "evaluation_level": "formal_human_reviewed_visual_gold_v2_4",
        "split": args.split,
        "sample_count": len(cases), "success_count": len(completed),
        "failure_count": len(cases) - len(completed),
        "visual_answer_rate": round(sum(item["result"].get("answer_mode") == "vision" for item in completed) / len(completed), 4) if completed else None,
        "required_term_coverage_average": round(sum(item["term_check"]["coverage"] for item in completed) / len(completed), 4) if completed else None,
        "visual_evidence_group_recall_average": round(sum(item["visual_evidence_group_recall"] for item in scored) / len(scored), 4) if scored else None,
        "visual_evidence_group_citation_recall_average": round(sum(item["visual_evidence_group_citation_recall"] for item in scored) / len(scored), 4) if scored else None,
        "excluded_image_violation_rate": round(sum(item["excluded_image_selection_violation"] for item in scored) / len(scored), 4) if scored else None,
        "image_citation_validity_rate": round(sum(item["image_citation_valid"] for item in scored) / len(scored), 4) if scored else None,
        "image_citation_rate": round(sum(bool(item["image_citation_numbers"]) for item in scored) / len(scored), 4) if scored else None,
        "formal_pass_rate": round(sum(item["formal_case_pass"] for item in scored) / len(scored), 4) if scored else None,
        "known_dataset_issue_count": sum(bool(item.get("known_issue")) for item in details),
        "external_api_calls": sum(item.get("result", {}).get("external_calls", 0) for item in details),
        "index_version_id": index_version_id, "reports_dir": str(output_dir),
        "scoring_mode": "reused_existing_results" if args.reuse_existing else "live_model_calls",
        "sample_size_warning": ("dev共20条，用于开发验证与问题定位。" if args.split == "dev" else
                                "test共10条，仅用于方案冻结后的最终评测，样本量仍有限。"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target = root / "evals" / "results" / "stage5" / f"visual_qa_formal_v2_4_{args.split}_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
