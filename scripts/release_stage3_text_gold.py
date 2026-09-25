"""Release the reviewed non-visual stage-3 cases as a versioned gold set."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evals/datasets/retrieval_stage3_eval_ready.jsonl"
TARGET = ROOT / "evals/datasets/retrieval_stage3_text_gold_v1.jsonl"


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    released = []
    for row in rows:
        if row.get("type") == "visual":
            continue
        if row.get("mapping_status") not in {"mapped", "no_answer_candidate"}:
            raise ValueError(f"{row.get('case_id')} 尚未完成证据映射")
        item = dict(row)
        item.update(review_status="approved", gold_eligible=True,
                    evaluation_status="released_text_gold_v1",
                    review_note="文本证据或无答案/澄清行为已完成独立审核并发布为阶段3文本评测集")
        released.append(item)
    if len(released) != 11 or sum(row.get("answerable") is True for row in released) != 8:
        raise ValueError("阶段3文本金标准数量不符合预期（应为11条，其中8条可回答）")
    TARGET.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in released) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": "released", "path": str(TARGET), "sample_count": len(released),
                      "answerable_count": 8, "unanswerable_or_clarify_count": 3},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
