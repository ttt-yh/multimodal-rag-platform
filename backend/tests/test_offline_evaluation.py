import json

import pytest

from multimodal_rag.application.offline_evaluation import load_gold_cases, score_retrieval, score_retrieval_case, summarize_gold_dataset


def case(answerable=True):
    return {"case_id": "c1", "question": "问题", "reference_answer": "答案",
            "answerable": answerable, "evidence_chunk_ids": ["chunk-a"] if answerable else [],
            "review_status": "approved", "gold_eligible": True}


def test_score_retrieval_calculates_recall_precision_and_mrr():
    result = score_retrieval([case()], [{"query": "问题", "results": [{"chunk_id": "chunk-a"}, {"chunk_id": "other"}]}], k=2)
    assert result["metrics"] == {"recall_at_k": 1.0, "precision_at_k": .5, "mrr": 1.0}


def test_unanswerable_case_is_not_in_quality_metrics():
    result = score_retrieval_case(case(False), {"results": [{"chunk_id": "other"}]})
    assert result["answerable"] is False
    assert result["reciprocal_rank"] is None


def test_pending_cases_are_not_loaded_as_formal_gold(tmp_path):
    path = tmp_path / "gold.jsonl"
    value = case()
    value["review_status"] = "pending_human_review"
    value["gold_eligible"] = False
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="没有审核通过"):
        load_gold_cases(path)


def test_summary_marks_dataset_pending_without_human_review(tmp_path):
    path = tmp_path / "gold.jsonl"
    value = case()
    value["review_status"] = "pending_human_review"
    value["gold_eligible"] = False
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = summarize_gold_dataset(path)
    assert summary["status"] == "pending_human_review"
    assert summary["sample_count"] == 1
    assert summary["quality_metrics"] is None
