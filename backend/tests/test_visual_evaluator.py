from multimodal_rag.application.visual_evaluator import score_visual_answer


def _case():
    return {"visual_gold": {
        "image_elements": [{"element_id": "gold"}],
        "alternative_image_groups": [[{"element_id": "alt_a"}, {"element_id": "alt_b"}]],
        "excluded_image_elements": [{"element_id": "wrong"}],
    }}


def test_score_visual_answer_accepts_one_image_from_each_required_group():
    score = score_visual_answer(
        _case(), {"review_status": "approved"},
        {"answer": "结论[图1][图2]", "answer_mode": "vision",
         "visual_evidence": [{"element_id": "gold"}, {"element_id": "alt_b"}]},
        {"coverage": 1.0},
    )
    assert score["visual_evidence_group_recall"] == 1.0
    assert score["visual_evidence_group_citation_recall"] == 1.0
    assert score["image_citation_valid"] is True
    assert score["formal_case_pass"] is True


def test_score_visual_answer_rejects_excluded_image_selection():
    score = score_visual_answer(
        _case(), {"review_status": "approved"},
        {"answer": "结论[图1]", "answer_mode": "vision",
         "visual_evidence": [{"element_id": "gold"}, {"element_id": "wrong"}]},
        {"coverage": 1.0},
    )
    assert score["excluded_image_selection_violation"] is True
    assert score["formal_case_pass"] is False


def test_score_visual_answer_rejects_out_of_range_image_citation():
    score = score_visual_answer(
        {"visual_gold": {"image_elements": [{"element_id": "gold"}]}},
        {"review_status": "approved"},
        {"answer": "结论[图2]", "answer_mode": "vision",
         "visual_evidence": [{"element_id": "gold"}]},
        {"coverage": 1.0},
    )
    assert score["image_citation_valid"] is False
    assert score["formal_case_pass"] is False


def test_score_visual_answer_rejects_citation_to_unrelated_selected_image():
    score = score_visual_answer(
        {"visual_gold": {"image_elements": [{"element_id": "gold"}]}},
        {"review_status": "approved"},
        {"answer": "结论[图2]", "answer_mode": "vision",
         "visual_evidence": [{"element_id": "gold"}, {"element_id": "other"}]},
        {"coverage": 1.0},
    )
    assert score["image_citation_valid"] is True
    assert score["visual_evidence_group_citation_recall"] == 0.0
    assert score["formal_case_pass"] is False
