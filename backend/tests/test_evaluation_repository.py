import json

import pytest

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.evaluation_repository import get_evaluation, list_evaluations
from multimodal_rag.infrastructure.settings import Settings


def report(root, kind, name, payload):
    target = root / "evals" / "results" / kind
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_list_evaluations_reads_only_supported_reports(tmp_path):
    report(tmp_path, "retrieval", "run_retrieval", {
        "query": "查询", "index_version_id": "idx", "external_calls": 2,
        "results": [{"chunk_id": "a"}], "records": [{"service": "embedding", "status_code": 200}],
    })
    report(tmp_path, "qa", "run_qa", {
        "query": "问答", "index_version_id": "idx", "external_calls": 3,
        "answer": "答案", "citations": [{"citation": "[1]"}], "records": [],
    })
    settings = Settings(_env_file=None, project_root=tmp_path)
    result = list_evaluations(settings)
    assert result["summary"] == {
        "total_runs": 2, "retrieval_runs": 1, "qa_runs": 1,
        "validated_runs": 2, "degraded_runs": 0, "external_calls": 5,
        "quality_metrics_available": False, "qa_metrics_available": False,
        "semantic_metrics_available": False,
        "visual_metrics_available": False,
        "visual_semantic_metrics_available": False,
        "final_test_available": False,
    }
    assert result["quality_metrics"] is None


def test_get_evaluation_sanitizes_records_and_returns_detail(tmp_path):
    report(tmp_path, "qa", "run_detail", {
        "query": "问答", "answer": "答案", "citations": [], "results": [],
        "records": [{"service": "chat", "status_code": 200, "secret": "never"}],
    })
    detail = get_evaluation(Settings(_env_file=None, project_root=tmp_path), "run_detail")
    assert detail["answer"] == "答案"
    assert "secret" not in json.dumps(detail)


def test_list_evaluations_reports_pending_offline_dataset(tmp_path):
    (tmp_path / "evals" / "results" / "offline").mkdir(parents=True)
    (tmp_path / "evals" / "results" / "offline" / "retrieval_v1_validation.json").write_text(
        json.dumps({"status": "pending_human_review", "quality_metrics": None,
                    "note": "待审核"}), encoding="utf-8")
    result = list_evaluations(Settings(_env_file=None, project_root=tmp_path))
    assert result["quality_metrics"] is None
    assert result["summary"]["quality_metrics_available"] is False
    assert result["note"] == "待审核"


def test_list_evaluations_flattens_completed_quality_metrics(tmp_path):
    (tmp_path / "evals" / "results" / "offline").mkdir(parents=True)
    (tmp_path / "evals" / "results" / "offline" / "retrieval_quality.json").write_text(
        json.dumps({"status": "approved", "k": 5, "metrics": {
            "recall_at_k": 1.0, "precision_at_k": .2, "mrr": .9},
            "note": "已完成"}), encoding="utf-8")
    result = list_evaluations(Settings(_env_file=None, project_root=tmp_path))
    assert result["summary"]["quality_metrics_available"] is True
    assert result["quality_metrics"] == {"k": 5, "recall_at_k": 1.0, "precision_at_k": .2, "mrr": .9}


def test_list_evaluations_exposes_formal_qa_metrics(tmp_path):
    target = tmp_path / "evals" / "results" / "stage4"
    target.mkdir(parents=True)
    (target / "qa_eval_summary.json").write_text(json.dumps({
        "status": "completed", "sample_count": 11, "answerable_count": 8,
        "unanswerable_count": 3, "citation_recall_average": 1.0,
        "citation_validity_rate": 1.0, "full_evidence_coverage_rate": 1.0,
        "unanswerable_refusal_rate": 1.0, "index_version_id": "idx_formal",
        "external_api_calls": 33,
    }), encoding="utf-8")
    result = list_evaluations(Settings(_env_file=None, project_root=tmp_path))
    assert result["summary"]["qa_metrics_available"] is True
    assert result["qa_quality_metrics"]["sample_count"] == 11
    assert result["qa_quality_metrics"]["unanswerable_refusal_rate"] == 1.0


def test_list_evaluations_exposes_development_semantic_metrics(tmp_path):
    target = tmp_path / "evals" / "results" / "stage4"
    target.mkdir(parents=True)
    (target / "semantic_eval_summary.json").write_text(json.dumps({
        "status": "completed", "evaluation_level": "development_llm_judge_pending_human_review",
        "sample_count": 8, "required_term_coverage_average": .975,
        "correctness_average_4": 4.0, "completeness_average_4": 3.875,
        "groundedness_average_4": 4.0, "semantic_pass_rate": 1.0,
        "human_review_queue_count": 2, "known_dataset_issue_count": 2,
    }), encoding="utf-8")
    result = list_evaluations(Settings(_env_file=None, project_root=tmp_path))
    assert result["summary"]["semantic_metrics_available"] is True
    assert result["semantic_quality_metrics"]["required_term_coverage_average"] == .975
    assert result["semantic_quality_metrics"]["human_review_queue_count"] == 2


def test_list_evaluations_exposes_development_visual_metrics(tmp_path):
    target = tmp_path / "evals" / "results" / "stage5"
    target.mkdir(parents=True)
    (target / "visual_qa_summary.json").write_text(json.dumps({
        "status": "completed", "evaluation_level": "development_visual_candidate_pending_human_release",
        "sample_count": 2, "visual_answer_rate": 1.0,
        "required_term_coverage_average": 1.0, "image_citation_rate": 1.0,
        "known_dataset_issue_count": 1,
    }), encoding="utf-8")
    result = list_evaluations(Settings(_env_file=None, project_root=tmp_path))
    assert result["summary"]["visual_metrics_available"] is True
    assert result["visual_quality_metrics"]["image_citation_rate"] == 1.0


def test_list_evaluations_prioritizes_frozen_visual_test_and_exposes_disclosure(tmp_path):
    target = tmp_path / "evals" / "results" / "stage5"
    target.mkdir(parents=True)
    (target / "visual_qa_summary.json").write_text(json.dumps({
        "status": "completed", "sample_count": 2, "visual_answer_rate": 1.0,
        "required_term_coverage_average": 1.0, "image_citation_rate": 1.0,
    }), encoding="utf-8")
    (target / "visual_qa_formal_v2_4_test_summary.json").write_text(json.dumps({
        "status": "completed", "evaluation_level": "formal_human_reviewed_visual_gold_v2_4",
        "split": "test", "sample_count": 10, "visual_answer_rate": 1.0,
        "required_term_coverage_average": .9083, "image_citation_rate": 1.0,
        "visual_evidence_group_recall_average": 1.0, "formal_pass_rate": .7,
    }), encoding="utf-8")
    (target / "visual_semantic_v2_4_test_summary.json").write_text(json.dumps({
        "status": "completed", "evaluation_level": "development_llm_judge_pending_human_review",
        "split": "test", "sample_count": 10, "visual_semantic_pass_rate": 1.0,
        "combined_pass_rate": .7, "correctness_average_4": 4.0,
        "completeness_average_4": 4.0, "groundedness_average_4": 4.0,
        "metric_notice": "开发级语义评审",
    }), encoding="utf-8")
    (target / "visual_v2_4_final_test_manifest.json").write_text(json.dumps({
        "status": "final_test_completed", "dataset_version": "visual_gold_v2_4",
        "sample_count": 10, "metrics": {
            "strict_deterministic_pass_rate": .7,
            "llm_judge_semantic_pass_rate": 1.0,
            "visual_evidence_group_recall": 1.0,
            "image_citation_validity_rate": 1.0,
        }, "execution": {"total_external_api_calls": 40, "rerun_permitted": False},
        "reporting_notice": "两套指标必须同时披露",
    }), encoding="utf-8")

    result = list_evaluations(Settings(_env_file=None, project_root=tmp_path))
    assert result["summary"]["visual_semantic_metrics_available"] is True
    assert result["summary"]["final_test_available"] is True
    assert result["visual_quality_metrics"]["sample_count"] == 10
    assert result["visual_quality_metrics"]["formal_pass_rate"] == .7
    assert result["visual_semantic_quality_metrics"]["visual_semantic_pass_rate"] == 1.0
    assert result["final_test_summary"]["metrics"]["strict_deterministic_pass_rate"] == .7
    assert "同时披露" in result["final_test_summary"]["reporting_notice"]


@pytest.mark.parametrize("run_id", ["../secret", "x/y", "", "x" * 129])
def test_get_evaluation_rejects_unsafe_id(tmp_path, run_id):
    with pytest.raises(AppError) as error:
        get_evaluation(Settings(_env_file=None, project_root=tmp_path), run_id)
    assert error.value.code == "invalid_evaluation_id"
