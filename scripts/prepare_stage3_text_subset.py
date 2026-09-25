"""Prepare a deterministic, diverse Markdown subset for the next RAG baseline.

This script only reads the existing source manifest and writes a new ingestion
allowlist. It never calls a model and never changes the active index.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


KEYWORDS = {
    "deployment": 8, "deploy": 8, "architecture": 8, "cluster": 6,
    "configuration": 7, "config": 7, "troubleshoot": 8, "error": 5,
    "performance": 7, "monitor": 6, "dashboard": 6, "storage": 7,
    "transaction": 5, "sql": 4, "tikv": 5, "tidb": 3, "pd": 5,
    "backup": 5, "restore": 5, "security": 4, "connect": 5,
}


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _bucket(source_path: str) -> str:
    parts = source_path.split("/")
    return parts[0] if len(parts) > 1 else "root"


def _score(row: dict) -> int:
    value = row["source_path"].lower()
    return sum(weight for keyword, weight in KEYWORDS.items() if keyword in value)


def select(rows: list[dict], count: int) -> list[dict]:
    # 评测/生产入库只使用开发集；test 文档保留在原始数据中，避免与开发白名单混用。
    candidates = [row for row in rows if row.get("format") == "md" and row.get("split") == "dev"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        groups[_bucket(row["source_path"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (-_score(row), row["source_path"], row["document_id"]))

    selected: list[dict] = []
    # Round-robin by source directory first, then continue by topic score. This
    # prevents a single large docs subtree from dominating the subset.
    buckets = sorted(groups)
    cursor = 0
    while len(selected) < min(count, len(candidates)) and buckets:
        bucket = buckets[cursor % len(buckets)]
        if groups[bucket]:
            selected.append(groups[bucket].pop(0))
        buckets = [name for name in buckets if groups[name]]
        cursor += 1
    selected.sort(key=lambda row: (-_score(row), row["source_path"], row["document_id"]))
    return selected


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "data/manifests/ingestion_chinese_md.jsonl"
    target = root / "data/manifests/ingestion_chinese_md_stage3.jsonl"
    summary_path = root / "evals/results/stage3/text_subset_summary.json"
    candidate_path = root / "evals/datasets/retrieval_stage3_candidates.jsonl"
    rows = _load(source)
    selected = select(rows, 40)
    if len(selected) != 40:
        raise SystemExit(f"可选 Markdown 文档不足 40 份，实际得到 {len(selected)} 份")
    output = []
    for row in selected:
        output.append({**row, "stage3_subset": True, "selection_score": _score(row),
                       "selection_bucket": _bucket(row["source_path"])})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n", encoding="utf-8")
    summary = {
        "source_manifest": "data/manifests/ingestion_chinese_md.jsonl",
        "target_manifest": "data/manifests/ingestion_chinese_md_stage3.jsonl",
        "source_count": len(rows), "selected_count": len(output),
        "document_ids_sha256": hashlib.sha256("\n".join(row["document_id"] for row in output).encode()).hexdigest(),
        "buckets": dict(Counter(row["selection_bucket"] for row in output)),
        "formats": dict(Counter(row["format"] for row in output)),
        "external_calls": 0,
        "active_index_changed": False,
    }
    selected_ids = {row["document_id"] for row in output}
    qa_source = root / "data/annotations/tidb_zh/qa_candidates.jsonl"
    qa_rows = _load(qa_source)
    candidates = []
    for row in qa_rows:
        evidence = row.get("evidence") or []
        matching = [item for item in evidence if item.get("document_id") in selected_ids]
        if not matching:
            continue
        candidates.append({
            "case_id": row["case_id"], "split": row["split"], "type": row["type"],
            "question": row["question"], "reference_answer": row.get("reference_answer"),
            "answerable": row.get("answerable"), "source_evidence_ids": [item["evidence_id"] for item in matching],
            "reference_document_ids": sorted({item["document_id"] for item in matching}),
            "source_paths": sorted({item["source_path"] for item in matching}),
            "review_status": "pending_index_mapping", "gold_eligible": False,
            "source": "tidb_zh_qa_candidate_subset",
        })
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in candidates) + "\n", encoding="utf-8")
    summary.update({"qa_candidates_total": len(qa_rows), "stage3_candidate_count": len(candidates),
                    "stage3_candidate_answerable": sum(row["answerable"] is True for row in candidates),
                    "stage3_candidate_unanswerable": sum(row["answerable"] is False for row in candidates),
                    "candidate_path": "evals/datasets/retrieval_stage3_candidates.jsonl"})
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
