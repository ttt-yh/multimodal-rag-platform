"""阶段5：对正式视觉问答结果执行独立语义正确性评审。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.application.semantic_evaluator import (
    build_visual_reference_judge_prompt,
    parse_judge_result,
)
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.model_adapters import ChatAdapter
from multimodal_rag.infrastructure.settings import load_settings


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段5视觉答案语义正确性评测")
    parser.add_argument("--dataset",
                        default="evals/datasets/retrieval_stage5_visual_gold_v2_4.jsonl")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--max-external-requests", type=int, default=0)
    parser.add_argument("--confirm-test-final", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="不调用模型，按已有Judge结果重新执行视觉语义联合判分")
    parser.add_argument("--reuse-from-version", choices=("v2_1", "v2_2", "v2_3"),
                        help="复用上一版本Judge结果，并结合当前视觉门禁重新汇总")
    args = parser.parse_args()

    settings = load_settings()
    root = settings.project_root
    cases = {row["case_id"]: row for row in _jsonl(root / args.dataset)
             if row.get("type") == "visual" and row.get("split") == args.split}
    if not cases:
        parser.error("正式视觉Gold中没有可评测样本")
    invalid = [case_id for case_id, row in cases.items() if not (
        row.get("review_status") == "approved"
        and row.get("gold_eligible") is True
        and row.get("evaluation_status") == "released_visual_gold_v2_4"
        and row.get("visual_review_status") == "human_released_gold_v2_4"
    )]
    if invalid:
        parser.error(f"正式视觉Gold门禁失败：{invalid}")
    selected = args.case or list(cases)
    if any(case_id not in cases for case_id in selected):
        parser.error("--case包含不存在的视觉题编号")
    if args.split == "test" and args.case:
        parser.error("test必须一次性评测全部样本")
    if args.split == "test" and not args.plan and not args.confirm_test_final:
        parser.error("test语义评测前必须提供 --confirm-test-final")
    required_budget = 0 if args.reuse_existing else len(selected)
    if args.plan:
        print(json.dumps({"status": "ready", "split": args.split,
                          "sample_count": len(selected),
                          "max_external_requests_required": required_budget,
                          "external_api_calls": 0}, ensure_ascii=False, indent=2))
        return 0
    if not args.reuse_existing and not args.confirm_live:
        parser.error("每条答案调用一次Chat评审，请提供 --confirm-live")
    if not args.reuse_existing and args.max_external_requests < required_budget:
        parser.error(f"预算不足：{len(selected)}条答案需要{required_budget}次外部请求")

    index_version_id = get_active_index(settings)["index_version_id"]
    qa_dir = (root / "evals/results/stage5-visual-formal/v2_4" /
              args.split / index_version_id)
    output_dir = (root / "evals/results/stage5-visual-semantic/v2_4" /
                  args.split / index_version_id)
    reuse_dir = (root / "evals/results/stage5-visual-semantic" /
                 args.reuse_from_version / args.split / index_version_id
                 if args.reuse_from_version else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    gateway = (None if args.reuse_existing else
               HttpGateway(settings, "chat", budget=CallBudget(required_budget)))
    try:
        for case_id in selected:
            case = cases[case_id]
            qa_path = qa_dir / f"{case_id}.json"
            if not qa_path.is_file():
                item = {"case_id": case_id, "status": "failed",
                        "error_code": "visual_qa_report_missing"}
            else:
                qa_report = json.loads(qa_path.read_text(encoding="utf-8"))
                result = qa_report.get("result") or {}
                record_start = len(gateway.records) if gateway is not None else 0
                try:
                    report_path = output_dir / f"{case_id}.json"
                    if args.reuse_existing:
                        source_report = (report_path if report_path.is_file() else
                                         reuse_dir / f"{case_id}.json" if reuse_dir else report_path)
                        if not source_report.is_file():
                            raise RuntimeError("没有可复用的视觉语义评审报告")
                        previous = json.loads(source_report.read_text(encoding="utf-8"))
                        judge = previous.get("judge")
                        if not isinstance(judge, dict):
                            raise RuntimeError("已有报告缺少Judge结果")
                        judge_model = previous.get("judge_model")
                        records = previous.get("records") or []
                    else:
                        response = ChatAdapter(gateway).complete(build_visual_reference_judge_prompt(
                            question=case["question"],
                            reference_answer=case["reference_answer"],
                            answer=result.get("answer", ""),
                        ))
                        judge = parse_judge_result(response["text"])
                        judge_model = response["model"]
                        records = gateway.records[record_start:]
                    deterministic_pass = bool(
                        (qa_report.get("formal_score") or {}).get("formal_case_pass"))
                    # The Judge cannot inspect the original image.  Correctness
                    # and completeness are therefore checked against the
                    # human-reviewed reference answer, while image grounding is
                    # enforced independently by the deterministic Gold-image
                    # selection and citation gate.
                    visual_semantic_pass = (
                        not judge["critical_error"]
                        and judge["correctness"] >= 3
                        and judge["completeness"] >= 3
                    )
                    combined_pass = deterministic_pass and visual_semantic_pass
                    item = {
                        "case_id": case_id, "question": case["question"],
                        "status": "completed", "index_version_id": index_version_id,
                        "deterministic_visual_gate_pass": deterministic_pass,
                        "judge": judge, "judge_model": judge_model,
                        "visual_semantic_pass": visual_semantic_pass,
                        "combined_case_pass": combined_pass,
                        "human_review_required": not combined_pass,
                        "records": records,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                except (AppError, RuntimeError) as exc:
                    item = {"case_id": case_id, "question": case["question"],
                            "status": "failed", "error_code": getattr(exc, "code", type(exc).__name__),
                            "error": str(exc), "human_review_required": True,
                            "records": (gateway.records[record_start:] if gateway is not None else []),
                            "created_at": datetime.now(timezone.utc).isoformat()}
            (output_dir / f"{case_id}.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if gateway is not None:
            gateway.close()

    details = []
    for case_id in cases:
        path = output_dir / f"{case_id}.json"
        details.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                       else {"case_id": case_id, "status": "missing_report",
                             "human_review_required": True})
    completed = [item for item in details if item.get("status") == "completed"]
    summary = {
        "status": "completed" if len(completed) == len(cases) else "partial_failure",
        "evaluation_level": "development_llm_judge_pending_human_review",
        "split": args.split, "sample_count": len(cases),
        "success_count": len(completed), "failure_count": len(cases) - len(completed),
        "visual_semantic_pass_rate": round(sum(item["visual_semantic_pass"] for item in completed)
                                           / len(completed), 4) if completed else None,
        "combined_pass_rate": round(sum(item["combined_case_pass"] for item in completed)
                                    / len(completed), 4) if completed else None,
        "correctness_average_4": round(sum(item["judge"]["correctness"] for item in completed)
                                       / len(completed), 4) if completed else None,
        "completeness_average_4": round(sum(item["judge"]["completeness"] for item in completed)
                                        / len(completed), 4) if completed else None,
        "groundedness_average_4": round(sum(item["judge"]["groundedness"] for item in completed)
                                        / len(completed), 4) if completed else None,
        "human_review_queue_count": sum(item.get("human_review_required", True)
                                        for item in details),
        "external_api_calls": sum(len(item.get("records") or []) for item in details),
        "index_version_id": index_version_id, "reports_dir": str(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scoring_mode": "reused_existing_judge" if args.reuse_existing else "live_llm_judge",
        "metric_notice": ("LLM Judge仅依据人工参考答案检查正确性与完整性，不能查看原图；"
                          "图片依据由Gold图片选择和引用门禁独立验证，正式正确率仍需人工复核。"),
    }
    target = root / "evals/results/stage5" / f"visual_semantic_v2_4_{args.split}_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
