"""Deterministic scoring for human-reviewed visual QA Gold cases."""
from __future__ import annotations

import re


_IMAGE_CITATION = re.compile(r"\[图(\d+)\]")


def score_visual_answer(case: dict, annotation: dict, result: dict,
                        term_check: dict) -> dict:
    """Score answer content and selected images without asking another model.

    ``image_elements`` are mandatory singleton evidence groups, while every
    ``alternative_image_groups`` entry is satisfied by any one image in that
    group.  Images explicitly excluded by the human reviewer are treated as a
    selection violation even if the final answer happens to be correct.
    """
    visual_gold = case.get("visual_gold") or {}
    required_groups = [
        {item["element_id"]}
        for item in visual_gold.get("image_elements", [])
        if item.get("element_id")
    ]
    required_groups.extend({
        item["element_id"] for item in group if item.get("element_id")
    } for group in visual_gold.get("alternative_image_groups", []))
    required_groups = [group for group in required_groups if group]

    selected = result.get("visual_evidence") or []
    selected_ids = {item.get("element_id") for item in selected if item.get("element_id")}
    group_matches = [sorted(group & selected_ids) for group in required_groups]
    group_recall = (sum(bool(match) for match in group_matches) / len(required_groups)
                    if required_groups else 1.0)

    excluded_ids = {
        item["element_id"] for item in visual_gold.get("excluded_image_elements", [])
        if item.get("element_id")
    }
    excluded_selected = sorted(excluded_ids & selected_ids)

    cited_numbers = [int(value) for value in _IMAGE_CITATION.findall(result.get("answer", ""))]
    citations_valid = bool(cited_numbers) and all(1 <= number <= len(selected) for number in cited_numbers)
    cited_ids = {
        selected[number - 1].get("element_id")
        for number in cited_numbers if 1 <= number <= len(selected)
    }
    cited_ids.discard(None)
    cited_group_matches = [sorted(group & cited_ids) for group in required_groups]
    cited_group_recall = (sum(bool(match) for match in cited_group_matches) / len(required_groups)
                          if required_groups else 1.0)
    term_coverage = float(term_check.get("coverage", 0.0))
    formal_pass = (
        result.get("answer_mode") == "vision"
        and term_coverage == 1.0
        and group_recall == 1.0
        and cited_group_recall == 1.0
        and not excluded_selected
        and citations_valid
    )
    return {
        "required_term_coverage": term_coverage,
        "required_visual_group_count": len(required_groups),
        "matched_visual_groups": group_matches,
        "visual_evidence_group_recall": round(group_recall, 4),
        "selected_image_element_ids": sorted(selected_ids),
        "cited_image_element_ids": sorted(cited_ids),
        "cited_visual_group_matches": cited_group_matches,
        "visual_evidence_group_citation_recall": round(cited_group_recall, 4),
        "excluded_image_element_ids_selected": excluded_selected,
        "excluded_image_selection_violation": bool(excluded_selected),
        "image_citation_numbers": cited_numbers,
        "image_citation_valid": citations_valid,
        "formal_case_pass": formal_pass,
        "annotation_review_status": annotation.get("review_status"),
    }
