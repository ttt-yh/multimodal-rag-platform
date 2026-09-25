"""Read and sanitize local retrieval/QA run reports for the workbench."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.settings import Settings


_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_KINDS = {"all", "retrieval", "qa"}


def _report_root(settings: Settings, kind: str) -> Path:
    return settings.project_root / "evals" / "results" / kind


def _offline_validation(settings: Settings) -> dict | None:
    root = settings.project_root / "evals" / "results" / "offline"
    # A completed quality report takes precedence over the dataset readiness report.
    for path in (root / "retrieval_quality.json", root / "retrieval_v1_validation.json"):
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _qa_validation(settings: Settings) -> dict | None:
    path = settings.project_root / "evals" / "results" / "stage4" / "qa_eval_summary.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    required = ("citation_recall_average", "citation_validity_rate",
                "full_evidence_coverage_rate", "unanswerable_refusal_rate")
    if not isinstance(value, dict) or not all(isinstance(value.get(key), (int, float)) for key in required):
        return None
    return {key: value[key] for key in (
        "status", "sample_count", "answerable_count", "unanswerable_count",
        "citation_recall_average", "citation_validity_rate",
        "full_evidence_coverage_rate", "unanswerable_refusal_rate",
        "index_version_id", "external_api_calls", "created_at") if key in value}


def _semantic_validation(settings: Settings) -> dict | None:
    path = settings.project_root / "evals" / "results" / "stage4" / "semantic_eval_summary.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    required = ("required_term_coverage_average", "correctness_average_4",
                "completeness_average_4", "groundedness_average_4", "semantic_pass_rate")
    if not isinstance(value, dict) or not all(isinstance(value.get(key), (int, float)) for key in required):
        return None
    return {key: value[key] for key in (
        "status", "evaluation_level", "sample_count", "required_term_coverage_average",
        "correctness_average_4", "completeness_average_4", "groundedness_average_4",
        "semantic_pass_rate", "human_review_queue_count", "known_dataset_issue_count",
        "index_version_id", "external_api_calls", "created_at") if key in value}


def _visual_validation(settings: Settings) -> dict | None:
    root = settings.project_root / "evals" / "results" / "stage5"
    # 优先展示冻结方案的一次性 test 结果，再回退到 dev 或早期兼容报告。
    path = next((candidate for candidate in (
        root / "visual_qa_formal_v2_4_test_summary.json",
        root / "visual_qa_formal_v2_4_dev_summary.json",
        root / "visual_qa_formal_summary.json",
        root / "visual_qa_summary.json",
    ) if candidate.is_file()), None)
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    required = ("visual_answer_rate", "required_term_coverage_average", "image_citation_rate")
    if not isinstance(value, dict) or not all(isinstance(value.get(key), (int, float)) for key in required):
        return None
    return {key: value[key] for key in (
        "status", "evaluation_level", "sample_count", "visual_answer_rate",
        "required_term_coverage_average", "visual_evidence_group_recall_average",
        "visual_evidence_group_citation_recall_average",
        "excluded_image_violation_rate", "image_citation_validity_rate",
        "image_citation_rate", "formal_pass_rate", "sample_size_warning",
        "known_dataset_issue_count",
        "index_version_id", "external_api_calls", "created_at") if key in value}


def _visual_semantic_validation(settings: Settings) -> dict | None:
    root = settings.project_root / "evals" / "results" / "stage5"
    path = next((candidate for candidate in (
        root / "visual_semantic_v2_4_test_summary.json",
        root / "visual_semantic_v2_4_dev_summary.json",
    ) if candidate.is_file()), None)
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    required = ("visual_semantic_pass_rate", "combined_pass_rate",
                "correctness_average_4", "completeness_average_4",
                "groundedness_average_4")
    if not isinstance(value, dict) or not all(
            isinstance(value.get(key), (int, float)) for key in required):
        return None
    return {key: value[key] for key in (
        "status", "evaluation_level", "split", "sample_count",
        "visual_semantic_pass_rate", "combined_pass_rate",
        "correctness_average_4", "completeness_average_4",
        "groundedness_average_4", "human_review_queue_count",
        "index_version_id", "external_api_calls", "scoring_mode",
        "metric_notice", "created_at") if key in value}


def _final_test_manifest(settings: Settings) -> dict | None:
    path = (settings.project_root / "evals" / "results" / "stage5" /
            "visual_v2_4_final_test_manifest.json")
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    metrics = value.get("metrics") if isinstance(value, dict) else None
    execution = value.get("execution") if isinstance(value, dict) else None
    if not isinstance(metrics, dict) or not isinstance(execution, dict):
        return None
    required = ("strict_deterministic_pass_rate", "llm_judge_semantic_pass_rate",
                "visual_evidence_group_recall", "image_citation_validity_rate")
    if not all(isinstance(metrics.get(key), (int, float)) for key in required):
        return None
    return {
        "status": value.get("status"),
        "dataset_version": value.get("dataset_version"),
        "completed_at": value.get("completed_at"),
        "active_index_version_id": value.get("active_index_version_id"),
        "sample_count": value.get("sample_count"),
        "metrics": {key: metrics[key] for key in (
            "visual_answer_rate", "visual_evidence_group_recall",
            "visual_evidence_group_citation_recall", "image_citation_validity_rate",
            "excluded_image_violation_rate", "required_term_coverage_average",
            "strict_deterministic_pass_rate", "llm_judge_semantic_pass_rate",
            "llm_judge_correctness_average_4", "llm_judge_completeness_average_4",
            "combined_strict_pass_rate") if key in metrics},
        "execution": {key: execution[key] for key in (
            "visual_qa_external_api_calls", "semantic_judge_external_api_calls",
            "total_external_api_calls", "rerun_permitted",
            "post_test_pipeline_changes_applied") if key in execution},
        "reporting_notice": value.get("reporting_notice"),
    }


def _safe_record(record: dict) -> dict:
    return {key: record[key] for key in
            ("service", "method", "status_code", "outcome", "duration_ms", "usage")
            if key in record}


def _read_report(path: Path, kind: str, *, detail: bool = False) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    if not isinstance(data, dict):
        return None
    records = data.get("records") if isinstance(data.get("records"), list) else []
    results = data.get("results") if isinstance(data.get("results"), list) else []
    citations = data.get("citations") if isinstance(data.get("citations"), list) else []
    created = data.get("created_at")
    if not isinstance(created, str):
        created = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    row = {
        "run_id": path.stem,
        "kind": kind,
        "query": data.get("query", ""),
        "index_version_id": data.get("index_version_id"),
        "status": "degraded" if data.get("degraded") else "validated",
        "degraded": bool(data.get("degraded")),
        "external_calls": data.get("external_calls", len(records)),
        "result_count": len(results),
        "citation_count": len(citations),
        "created_at": created,
    }
    if detail:
        row.update({
            "answer": data.get("answer"),
            "results": results,
            "citations": citations,
            "retrieval": data.get("retrieval", {}),
            "records": [_safe_record(item) for item in records if isinstance(item, dict)],
        })
    return row


def list_evaluations(settings: Settings, *, kind: str = "all", limit: int = 50) -> dict:
    if kind not in _KINDS or not 1 <= limit <= 100:
        raise AppError("invalid_evaluation_filter", "评测类型或数量参数不正确", 422)
    kinds = ("retrieval", "qa") if kind == "all" else (kind,)
    rows: list[dict] = []
    skipped = 0
    for current in kinds:
        root = _report_root(settings, current)
        if not root.is_dir():
            continue
        for path in root.glob("*.json"):
            row = _read_report(path, current)
            if row is None:
                skipped += 1
            else:
                rows.append(row)
    rows.sort(key=lambda row: row["created_at"], reverse=True)
    rows = rows[:limit]
    validation = _offline_validation(settings)
    qa_quality_metrics = _qa_validation(settings)
    semantic_quality_metrics = _semantic_validation(settings)
    visual_quality_metrics = _visual_validation(settings)
    visual_semantic_quality_metrics = _visual_semantic_validation(settings)
    final_test_summary = _final_test_manifest(settings)
    quality_metrics = None
    if validation:
        raw_metrics = validation.get("metrics")
        if isinstance(raw_metrics, dict) and all(isinstance(raw_metrics.get(key), (int, float))
                                                 for key in ("recall_at_k", "precision_at_k", "mrr")):
            quality_metrics = {"k": validation.get("k", 5), **raw_metrics}
        elif isinstance(validation.get("quality_metrics"), dict):
            quality_metrics = validation["quality_metrics"]
    metrics_available = isinstance(quality_metrics, dict)
    note = (validation.get("note") if validation and isinstance(validation.get("note"), str) else
            "当前目录包含联调报告，尚未导入带标准答案和证据标注的离线评测成绩。")
    if final_test_summary:
        note = ("Visual Gold v2.4 已完成冻结后的单次 test 评测；"
                "严格确定性门禁与开发级 LLM 语义评审必须同时披露。")
    return {
        "runs": rows,
        "summary": {
            "total_runs": len(rows),
            "retrieval_runs": sum(row["kind"] == "retrieval" for row in rows),
            "qa_runs": sum(row["kind"] == "qa" for row in rows),
            "validated_runs": sum(row["status"] == "validated" for row in rows),
            "degraded_runs": sum(row["degraded"] for row in rows),
            "external_calls": sum(row["external_calls"] for row in rows),
            "quality_metrics_available": metrics_available,
            "qa_metrics_available": qa_quality_metrics is not None,
            "semantic_metrics_available": semantic_quality_metrics is not None,
            "visual_metrics_available": visual_quality_metrics is not None,
            "visual_semantic_metrics_available": visual_semantic_quality_metrics is not None,
            "final_test_available": final_test_summary is not None,
        },
        "quality_metrics": quality_metrics if metrics_available else None,
        "qa_quality_metrics": qa_quality_metrics,
        "semantic_quality_metrics": semantic_quality_metrics,
        "visual_quality_metrics": visual_quality_metrics,
        "visual_semantic_quality_metrics": visual_semantic_quality_metrics,
        "final_test_summary": final_test_summary,
        "note": note,
        "skipped_reports": skipped,
    }


def get_evaluation(settings: Settings, run_id: str) -> dict:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise AppError("invalid_evaluation_id", "评测编号格式不正确", 422)
    for kind in ("retrieval", "qa"):
        path = _report_root(settings, kind) / f"{run_id}.json"
        if path.is_file():
            result = _read_report(path, kind, detail=True)
            if result is not None:
                return result
    raise AppError("evaluation_not_found", "评测报告不存在", 404)
