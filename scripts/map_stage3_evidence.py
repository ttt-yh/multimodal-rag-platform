"""将阶段3候选问题的原文 Section 映射到新索引中的真实 Chunk。

映射只依据 document_id、source_path 和行号交集，不调用模型，也不把确定性映射
直接标记为人工审核通过；输出仍需人工确认后才能作为正式评测金标准。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import load_settings


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_key(value: str) -> str:
    prefix = "data/raw/tidb_zh/source/"
    return value.replace("\\", "/").removeprefix(prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description="映射阶段3评测候选的原文证据到真实Chunk")
    parser.add_argument("--candidates", default="evals/datasets/retrieval_stage3_candidates.jsonl")
    parser.add_argument("--report", default="evals/results/stage3/stage3b_ingestion_report.json")
    parser.add_argument("--output", default="evals/datasets/retrieval_stage3_mapped.jsonl")
    args = parser.parse_args()
    settings = load_settings()
    root = settings.project_root
    candidates = _rows(root / args.candidates)
    sections = {row["evidence_id"]: row for row in _rows(root / "data/annotations/tidb_zh/sections.jsonl")}
    report = json.loads((root / args.report).read_text(encoding="utf-8"))
    processing_ids = sorted(report.get("processing_versions", {}).keys())
    if not processing_ids:
        raise SystemExit("阶段3B报告没有处理版本")

    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            """SELECT chunk_id,document_id,source_locations
               FROM mrag.chunks WHERE processing_version_id = ANY(%s)""",
            (processing_ids,),
        ).fetchall()

    chunks_by_document: dict[str, list[dict]] = {}
    for chunk_id, document_id, source_locations in rows:
        chunks_by_document.setdefault(document_id, []).append({
            "chunk_id": chunk_id, "source_locations": source_locations or []
        })

    mapped = []
    for candidate in candidates:
        chunk_ids: list[str] = []
        evidence_mappings = []
        for evidence_id in candidate.get("source_evidence_ids", []):
            section = sections.get(evidence_id)
            if not section:
                evidence_mappings.append({"evidence_id": evidence_id, "chunk_ids": [],
                                          "status": "section_missing"})
                continue
            matched = []
            for chunk in chunks_by_document.get(section["document_id"], []):
                overlaps = any(
                    _source_key(location.get("source_path", "")) == _source_key(section["source_path"])
                    and location.get("line_start") is not None
                    and location.get("line_end") is not None
                    and location["line_end"] >= section["line_start"]
                    and location["line_start"] <= section["line_end"]
                    for location in chunk["source_locations"]
                )
                if overlaps:
                    matched.append(chunk["chunk_id"])
            matched = sorted(set(matched))
            chunk_ids.extend(matched)
            evidence_mappings.append({"evidence_id": evidence_id, "chunk_ids": matched,
                                      "status": "mapped" if matched else "no_chunk_overlap"})
        chunk_ids = sorted(set(chunk_ids))
        answerable = candidate.get("answerable") is True
        if answerable:
            status = "mapped" if chunk_ids and all(item["status"] == "mapped" for item in evidence_mappings) else "mapping_review"
        else:
            status = "no_answer_candidate"
        mapped.append({**candidate, "chunk_ids": chunk_ids,
                       "evidence_mappings": evidence_mappings,
                       "mapping_status": status, "gold_eligible": False,
                       "review_status": "pending_human_review"})

    target = root / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in mapped) + "\n", encoding="utf-8")
    summary = {"status": "prepared", "candidate_count": len(mapped),
               "mapping_status": dict(Counter(row["mapping_status"] for row in mapped)),
               "answerable_with_chunks": sum(row["answerable"] is True and bool(row["chunk_ids"]) for row in mapped),
               "output": str(target), "external_api_calls": 0,
               "gold_eligible": False}
    summary_path = root / "evals/results/stage3/evidence_mapping_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
