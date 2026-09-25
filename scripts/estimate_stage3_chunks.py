"""Estimate structural chunks for the Stage 3 text subset offline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from multimodal_rag.core.chunking import split_elements
from multimodal_rag.core.models import Document, DocumentVersion
from multimodal_rag.infrastructure.text_parser import parse_text, verify_sources


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "data/manifests/ingestion_chinese_md_stage3.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    documents = []
    failures = []
    total_chars = 0
    total_elements = 0
    total_chunks = 0
    chunk_lengths: list[int] = []
    for row in rows:
        path = root / row["path"]
        try:
            text = path.read_text(encoding="utf-8")
            version_id = "ver_stage3_" + row["document_id"].replace("tidb85_", "")
            document = Document(document_id=row["document_id"], title=row["title"],
                                knowledge_base=row["knowledge_base"], source_path=row["source_path"],
                                source_family=row["source_family"], format="md", license=row["license"])
            version = DocumentVersion(version_id=version_id, document_id=row["document_id"],
                                      content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                      source_revision=row["source_revision"])
            elements, _ = parse_text(document, version, text)
            verify_sources(document, version, text, elements)
            chunks = split_elements(elements, "proc_stage3_estimate", max_characters=1200,
                                    overlap_characters=160, min_characters=100)
            total_chars += len(text)
            total_elements += len(elements)
            total_chunks += len(chunks)
            chunk_lengths.extend(item.estimated_length for item in chunks)
            documents.append({"document_id": row["document_id"], "source_path": row["source_path"],
                              "characters": len(text), "elements": len(elements), "chunks": len(chunks)})
        except Exception as exc:  # report every source failure instead of hiding it
            failures.append({"document_id": row.get("document_id"), "path": row.get("path"), "error": str(exc)})
    summary = {
        "manifest": "data/manifests/ingestion_chinese_md_stage3.jsonl",
        "document_count": len(rows), "parsed_count": len(documents), "failure_count": len(failures),
        "total_characters": total_chars, "total_elements": total_elements, "estimated_chunks": total_chunks,
        "chunk_characters": {"max": max(chunk_lengths) if chunk_lengths else 0,
                              "min": min(chunk_lengths) if chunk_lengths else 0,
                              "average": round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0,
                              "under_100_count": sum(length < 100 for length in chunk_lengths)},
        "parameters": {"max_characters": 1200, "overlap_characters": 160, "min_characters": 100},
        "external_calls": 0, "active_index_changed": False, "documents": documents, "failures": failures,
    }
    target = root / "evals/results/stage3/chunk_estimate.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("document_count", "parsed_count", "failure_count",
        "total_characters", "total_elements", "estimated_chunks", "chunk_characters", "external_calls")},
        ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
