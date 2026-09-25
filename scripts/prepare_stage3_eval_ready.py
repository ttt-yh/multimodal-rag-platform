"""根据阶段3人工核对结果生成本轮可评测子集，不覆盖原始候选集。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "evals/datasets/retrieval_stage3_mapped.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    ready = []
    deferred = []
    for row in rows:
        if row.get("type") == "multi_document":
            deferred.append({**row, "evaluation_status": "deferred_missing_second_source",
                             "gold_eligible": False,
                             "review_note": "第二来源文档未进入本批索引，不能用于双来源召回指标"})
            continue
        item = dict(row)
        item["evaluation_status"] = "ready_pending_gold_release"
        item["gold_eligible"] = False
        # 离线评测器使用 evidence_chunk_ids；无答案/澄清题必须清空证据集合，
        # 而视觉题保留图片所在 Chunk 作为“定位证据”，不代表文本包含图中答案。
        item["evidence_chunk_ids"] = list(row.get("chunk_ids", [])) if row.get("answerable") is True else []
        if row.get("type") == "visual":
            item["visual_review_status"] = "assistant_checked_pending_human_release"
            item["review_note"] = "已核对原始图片；当前文本Chunk只代表图片所在位置，答案需走多模态链路"
        else:
            item["review_note"] = "文本证据或无答案/澄清行为已完成核对"
        ready.append(item)

    ready_path = root / "evals/datasets/retrieval_stage3_eval_ready.jsonl"
    ready_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in ready) + "\n", encoding="utf-8")
    deferred_path = root / "evals/datasets/retrieval_stage3_deferred.jsonl"
    deferred_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in deferred) + "\n", encoding="utf-8")
    summary = {"status": "prepared", "ready_count": len(ready), "deferred_count": len(deferred),
               "ready_types": dict(Counter(row["type"] for row in ready)),
               "deferred_types": dict(Counter(row["type"] for row in deferred)),
               "gold_eligible": False, "external_api_calls": 0,
               "ready_path": str(ready_path), "deferred_path": str(deferred_path)}
    summary_path = root / "evals/results/stage3/eval_ready_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
