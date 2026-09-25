import pytest

from multimodal_rag.application.semantic_evaluator import (
    build_visual_reference_judge_prompt,
    check_required_terms,
    parse_judge_result,
)
from multimodal_rag.core.errors import AppError


def test_required_terms_accept_equivalent_arrow_and_reports_missing():
    result = check_required_terms("Client → TiDB，另外包含 RocksDB Compaction。", [
        ["Client -> TiDB", "Client → TiDB"], ["RocksDB Compaction"], ["TiKV -> Rocksdb"]])
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["missing_groups"] == [2]


def test_required_terms_ignore_ui_quotes_and_preserve_composite_atoms():
    result = check_required_terms(
        "左上方 PD 版本是 v6.5.0，按钮位于配置界面的“Session”区域。",
        [["PD v6.5.0"], ["Session 区域"]])
    assert result["coverage"] == 1.0


def test_parse_judge_result_validates_contract_and_calculates_pass():
    value = parse_judge_result('```json\n{"correctness":4,"completeness":3,"groundedness":4,'
                               '"critical_error":false,"missing_points":[],"unsupported_claims":[],'
                               '"reason":"证据支持"}\n```')
    assert value["semantic_pass"] is True


def test_parse_judge_result_rejects_out_of_range_score():
    with pytest.raises(AppError) as error:
        parse_judge_result('{"correctness":5,"completeness":3,"groundedness":4,'
                           '"critical_error":false,"missing_points":[],"unsupported_claims":[],'
                           '"reason":"错误"}')
    assert error.value.code == "invalid_judge_output"


def test_visual_reference_judge_prompt_separates_image_grounding():
    prompt = build_visual_reference_judge_prompt(
        question="哪个数值最大？", reference_answer="3.5 s 最大。",
        answer="列出 3.5 s，但结论选择 3.4 s。")
    assert "不能因为缺少图片" in prompt
    assert "groundedness固定返回4" in prompt
    assert "相反的最大值或最小值" in prompt
