"""Validate human decisions and release the 30-case visual Gold v2 dataset."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

from multimodal_rag.application.visual_evidence import read_active_visual_element
from multimodal_rag.infrastructure.visual_repository import load_active_image_element
from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.settings import load_settings


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evals/datasets/retrieval_stage5_visual_mapped_v2.jsonl"
DECISIONS = ROOT / "evals/results/stage5/visual_gold_v2_review_decisions.json"
REVISIONS = ROOT / "evals/datasets/qa_stage5_visual_question_revisions_v2.jsonl"
V1 = ROOT / "evals/datasets/retrieval_stage5_visual_gold_v1.jsonl"
TARGET = ROOT / "evals/datasets/retrieval_stage5_visual_gold_v2_4.jsonl"
FACTS = ROOT / "evals/datasets/qa_stage5_visual_facts_v2_4.jsonl"
SUMMARY = ROOT / "evals/results/stage5/visual_gold_v2_4_release.json"

TERM_GROUP_OVERRIDES = {
    "zh_visual_24c7329ac125cb41": [
        ["query OK 的最大值为 21.13 K", "query OK Max: 21.13 K"],
        ["query Error 的最大值为 0", "query Error Max: 0"],
    ],
    "zh_visual_4bc5cb82004a2d46": [
        ["3 个分区", "三个虚线部署分区", "共有3个虚线部署分区", "有3个虚线部署分区"],
        ["每区 1 TiDB", "每个分区各包含一个 TiDB", "TiDB各1个"],
        ["每区 1 PD", "每个分区各包含一个 PD", "PD各1个"],
        ["每区 2 TiKV", "每个分区各包含两个 TiKV", "TiKV各2个"],
    ],
    "zh_visual_59c6725c29f77f6c": [
        ["Recent 30 min"], ["Select Database"], ["Select SQL kind"],
        ["Dashboard version 4.0.0", "Dashboard 版本 4.0.0"],
    ],
    "zh_visual_a52c7655802e8443": [
        ["tpcc.bmsql_stock", "bmsql_stock"], ["3.5 s"],
    ],
    "zh_visual_ac4744f453150db3": [
        ["左下向右上", "左下到右上"], ["阶梯状", "斜带状"],
        ["不能读出精确时间及键值范围", "不能确定精确时间或键值范围",
         "无法读出确切时间和键值范围", "不能仅凭该图读出确切的时间和键值范围"],
    ],
    "zh_visual_bafe7a39391d026c": [
        ["View search histroy", "search histroy"],
        ["拼写错误", "正确应为 history"],
    ],
}

REFERENCE_ANSWER_OVERRIDES = {
    "zh_visual_bafe7a39391d026c": (
        "截图中红色箭头指向的完整入口文本为 View search histroy；其中 histroy 是截图中的"
        "原始拼写错误，规范拼写为 history。"
    ),
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visual_gold_images(visual_gold: dict) -> list[dict]:
    images = list(visual_gold.get("image_elements") or [])
    images.extend(visual_gold.get("excluded_image_elements") or [])
    for group in visual_gold.get("alternative_image_groups") or []:
        images.extend(group)
    return images


def _validate_carried_visual_gold(settings, visual_gold: dict) -> None:
    images = _visual_gold_images(visual_gold)
    if not images:
        raise ValueError("沿用的视觉Gold没有图片证据")
    for image in images:
        content = read_active_visual_element(settings, image["element_id"])
        path = ROOT / image["asset_path"]
        if (content["sha256"] != image["sha256"] or not path.is_file()
                or _sha(path) != image["sha256"] or len(content["content"]) != image["byte_size"]):
            raise ValueError(f"沿用的视觉Gold图片已变化：{image['element_id']}")


def main() -> int:
    settings = load_settings()
    active = get_active_index(settings)["index_version_id"]
    candidates = _jsonl(SOURCE)
    v1 = {row["case_id"]: row for row in _jsonl(V1)}
    revisions = {row["case_id"]: row for row in _jsonl(REVISIONS)}
    for revision in revisions.values():
        if not (
            revision.get("status") == "approved"
            and revision.get("review_status") == "approved"
            and str(revision.get("reviewer", "")).strip()
            and str(revision.get("reviewed_at", "")).strip()
        ):
            raise ValueError(f"问题修订尚未通过人工审核：{revision.get('case_id')}")
    decisions_list = json.loads(DECISIONS.read_text(encoding="utf-8"))
    if not isinstance(decisions_list, list):
        raise ValueError("审核决定必须是JSON数组")
    decision_ids = [row.get("case_id") for row in decisions_list]
    candidate_ids = [row["case_id"] for row in candidates]
    if len(decisions_list) != 30 or len(set(decision_ids)) != 30 or set(decision_ids) != set(candidate_ids):
        raise ValueError("审核决定必须与30条候选一一对应且编号唯一")
    for decision in decisions_list:
        if decision.get("status") not in {"approved", "needs_changes", "rejected"}:
            raise ValueError(f"审核状态不合法：{decision.get('case_id')}")
        if not str(decision.get("reviewer", "")).strip() or not str(decision.get("notes", "")).strip():
            raise ValueError(f"审核人或说明为空：{decision.get('case_id')}")
        try:
            datetime.fromisoformat(str(decision.get("reviewed_at", "")).replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"审核时间格式不合法：{decision.get('case_id')}") from None
    if any(row["status"] != "approved" for row in decisions_list):
        raise ValueError("仍有未通过样本，不能发布完整视觉Gold v2")
    decisions = {row["case_id"]: row for row in decisions_list}

    released, facts = [], []
    for source in candidates:
        row = dict(source)
        mapping = row["active_visual_mapping"]
        element_id = mapping["element_ids"][0]
        chunk_id = mapping["chunk_ids"][0]
        db_image = load_active_image_element(settings, element_id)
        content = read_active_visual_element(settings, element_id)
        if db_image is None or content["sha256"] != mapping["sha256"]:
            raise ValueError(f"活动图片证据已变化：{row['case_id']}")
        asset = ROOT / mapping["asset_path"]
        if not asset.is_file() or _sha(asset) != mapping["sha256"]:
            raise ValueError(f"原图路径或哈希已变化：{row['case_id']}")
        image = {
            "element_id": element_id, "image_ref": db_image["image_ref"],
            "chunk_id": chunk_id, "asset_path": mapping["asset_path"],
            "byte_size": len(content["content"]), "sha256": content["sha256"],
        }
        # Preserve the v1-reviewed disambiguation/exclusion and alternative
        # evidence groups for the two previously released cases.
        if row["case_id"] in v1:
            previous = v1[row["case_id"]]
            row["question"] = previous["question"]
            row["reference_answer"] = previous["reference_answer"]
            visual_gold = previous["visual_gold"]
            _validate_carried_visual_gold(settings, visual_gold)
            release_lineage = "carried_forward_from_visual_gold_v1"
        else:
            visual_gold = {"image_elements": [image],
                           "matching_rule": "该人工核验原图必须被选择并由回答实际引用"}
            release_lineage = "new_human_reviewed_visual_gold_v2"
        if row["case_id"] in revisions:
            revision = revisions[row["case_id"]]
            if row["question"] != revision["original_question"]:
                raise ValueError(f"问题修订基线不匹配：{row['case_id']}")
            row["question"] = revision["proposed_question"]
            row["question_revision"] = revision
            release_lineage += "+human_approved_question_revision"
        if row["case_id"] in REFERENCE_ANSWER_OVERRIDES:
            row["reference_answer"] = REFERENCE_ANSWER_OVERRIDES[row["case_id"]]
            row["reference_answer_revision"] = {
                "reason": "与人工看图审核记录对齐，保留截图中的原始拼写",
                "reviewer": "human-review",
                "reviewed_at": "2026-09-25",
            }
            release_lineage += "+human_review_aligned_reference_answer"
        decision = decisions[row["case_id"]]
        row.update({
            "review_status": "approved", "gold_eligible": True,
            "evaluation_status": "released_visual_gold_v2_4",
            "visual_review_status": "human_released_gold_v2_4",
            "index_version_id": active, "visual_gold": visual_gold,
            "human_review": decision, "release_lineage": release_lineage,
        })
        released.append(row)
        facts.append({
            "case_id": row["case_id"],
            "required_term_groups": TERM_GROUP_OVERRIDES.get(
                row["case_id"], [[fact] for fact in row.get("required_facts", [])]),
            "review_status": "approved", "gold_eligible": True,
            "split": row["split"], "source": "human_released_visual_gold_v2_4",
        })
    if len(released) != 30 or sum(row["split"] == "dev" for row in released) != 20:
        raise ValueError("发布集规模或划分不符合20 dev / 10 test")
    TARGET.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in released) + "\n",
                      encoding="utf-8")
    FACTS.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in facts) + "\n",
                     encoding="utf-8")
    summary = {
        "status": "released", "version": "visual_gold_v2_4",
        "sample_count": 30, "dev_count": 20, "test_count": 10,
        "approved_count": 30, "needs_changes_count": 0, "rejected_count": 0,
        "active_index_version_id": active,
        "candidate_sha256": _sha(SOURCE), "decision_sha256": _sha(DECISIONS),
        "question_revisions_sha256": _sha(REVISIONS),
        "gold_sha256": _sha(TARGET), "facts_sha256": _sha(FACTS),
        "external_api_calls": 0,
        "test_policy": "test只允许方案冻结后一次性最终评测，不参与调参",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
