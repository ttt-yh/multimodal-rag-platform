"""Release the two human-reviewed visual cases as a versioned gold set.

Independent human review (2026-09-25):
- QPS case now explicitly targets the DEFAULT-config screenshot; the modified-config
  screenshot (query OK Max 20.37 K) is recorded as excluded.
- TiCDC case records architecture-1.jpg and architecture-6.jpg as an interchangeable
  evidence group; either image alone supports the reference answer.

Image hashes and byte sizes are recomputed at release time via locate_visual_evidence;
nothing is hard-coded. Source rows come from retrieval_stage3_eval_ready.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

from multimodal_rag.application.visual_evidence import locate_visual_evidence
from multimodal_rag.infrastructure.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evals/datasets/retrieval_stage3_eval_ready.jsonl"
TARGET = ROOT / "evals/datasets/retrieval_stage5_visual_gold_v1.jsonl"

QPS_CASE = "zh_visual_24c7329ac125cb41"
TICDC_CASE = "zh_visual_93b780f4379dc7b9"

REVISED_QPS_QUESTION = (
    "依据《三节点混合部署最佳实践》中默认配置（QPS with default config）的 QPS 原始截图，"
    "该截图图例中 query OK 的最大值与 query Error 的最大值分别是多少？"
)

# element_id -> role, resolved against chunks listed per case.
QPS_GOLD_CHUNKS = ["chk_c7f1113d61d1ce51e8a87aff5f27ae99"]
QPS_GOLD_IDS = {"el_72043ee543ba516b8af2a457": "gold_default_config"}
QPS_EXCLUDED_IDS = {
    "el_87d28b59fbbc4f64f618639c": "调整后配置截图（QPS with modified config），query OK Max 为 20.37 K，非本题所问的默认配置",
}
TICDC_GOLD_CHUNKS = ["chk_b8827c23a359038ebe2673a8b84f1595",
                     "chk_0df9ddafebdec023133bfdd4b3cae826"]
TICDC_ALTERNATIVE_IDS = ["el_ed0336d26886f5b806c1fa93",
                         "el_98ae917df4265fd28c3406b0"]


def _image_index(settings, chunks: list[str]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for chunk_id in chunks:
        result = locate_visual_evidence(settings, [chunk_id], max_images=5)
        for cand in result["candidates"]:
            if cand.get("vision_eligible"):
                index[cand["element_id"]] = {
                    "element_id": cand["element_id"], "image_ref": cand["image_ref"],
                    "chunk_id": cand["chunk_id"], "asset_path": cand["asset_path"],
                    "byte_size": cand["byte_size"], "sha256": cand["sha256"],
                }
    return index


def main() -> int:
    settings = load_settings()
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {row["case_id"]: row for row in rows if row.get("type") == "visual"}
    if set(cases) != {QPS_CASE, TICDC_CASE}:
        raise ValueError("视觉候选数量或编号不符合预期（应为指定的两条）")
    for row in cases.values():
        if row.get("mapping_status") != "mapped":
            raise ValueError(f"{row['case_id']} 尚未完成证据映射")

    images = _image_index(settings, QPS_GOLD_CHUNKS + TICDC_GOLD_CHUNKS)

    def require(element_id: str) -> dict:
        if element_id not in images:
            raise ValueError(f"图片元素未通过安全定位：{element_id}")
        return dict(images[element_id])

    released: list[dict] = []

    qps = dict(cases[QPS_CASE])
    qps.update({
        "question": REVISED_QPS_QUESTION,
        "review_status": "approved", "gold_eligible": True,
        "evaluation_status": "released_visual_gold_v1",
        "visual_review_status": "human_released_gold_v1",
        "question_revision": {
            "version": "v1",
            "source_file": "evals/datasets/qa_stage5_visual_facts_v1.jsonl",
            "reason": "原问题未区分默认配置与调整后两张QPS截图，易把21.13K与20.37K混淆",
            "reviewed_by": "human-review", "reviewed_at": "2026-09-25",
        },
        "visual_gold": {
            "image_elements": [dict(require(eid), role=role)
                              for eid, role in QPS_GOLD_IDS.items()],
            "excluded_image_elements": [dict(require(eid), reason=reason)
                                        for eid, reason in QPS_EXCLUDED_IDS.items()],
        },
        "review_note": "2026-09-25 人工实际查看默认配置截图：query OK Max 21.13 K、query Error Max 0，与参考答案一致；调整后截图(20.37K)已排除",
    })
    released.append(qps)

    ticdc = dict(cases[TICDC_CASE])
    ticdc.update({
        "review_status": "approved", "gold_eligible": True,
        "evaluation_status": "released_visual_gold_v1",
        "visual_review_status": "human_released_gold_v1",
        "visual_gold": {
            "alternative_image_groups": [
                [require(eid) for eid in TICDC_ALTERNATIVE_IDS],
            ],
            "matching_rule": "组内任一图片即可独立支撑参考答案（Owner 位于 node1 Capture；下游 MySQL/TiDB/Kafka）",
        },
        "review_note": "2026-09-25 人工实际查看 ticdc-architecture-1.jpg 与 -6.jpg：两图均显示 Owner 在 node1 Capture 内、下游为 MySQL/TiDB/Kafka，登记为可替代证据组",
    })
    released.append(ticdc)

    if len(released) != 2:
        raise ValueError("视觉Gold应恰好2条")
    TARGET.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in released) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": "released", "path": str(TARGET), "sample_count": 2,
                      "answerable_count": 2}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
