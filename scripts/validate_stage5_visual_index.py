"""Read-only release preflight for the Stage 5B draft index."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段5B草稿索引发布前核验")
    parser.add_argument("index_version_id")
    parser.add_argument("--report", default="evals/results/stage5/stage5b_index_preflight.json")
    args = parser.parse_args()
    settings = load_settings()
    active = get_active_index(settings)
    with connection(settings, read_only=True) as conn:
        row = conn.execute(
            """SELECT status,chunk_count,vector_count,keyword_count,artifact_path,manifest
                 FROM mrag.index_versions WHERE index_version_id=%s""",
            (args.index_version_id,),).fetchone()
        if not row:
            raise ValueError("草稿索引不存在")
        manifest = row[5]
        scope = manifest.get("processing_version_ids") or []
        approved = conn.execute(
            """SELECT count(*)::int FROM mrag.processing_versions
                WHERE processing_version_id=ANY(%s) AND quality_status='approved'""",
            (scope,),).fetchone()[0]
        chunk_ids = [item[0] for item in conn.execute(
            """SELECT chunk_id FROM mrag.chunks WHERE processing_version_id=ANY(%s)
                ORDER BY processing_version_id,ordinal""", (scope,)).fetchall()]
    artifact_root = (settings.project_root / row[4]).resolve()
    disk_manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    bm25_meta = json.loads((artifact_root / "bm25" / args.index_version_id / "metadata.json")
                           .read_text(encoding="utf-8"))
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    logging.getLogger("posthog").disabled = True
    import chromadb
    client = chromadb.PersistentClient(path=str(artifact_root / "chroma"),
        settings=chromadb.config.Settings(anonymized_telemetry=False))
    collection = client.get_collection(name=args.index_version_id)
    chroma_count = collection.count()
    checks = {
        "target_is_draft": row[0] == "draft",
        "current_active_unchanged": active["index_version_id"] != args.index_version_id,
        "scope_is_strict_superset_of_active": set(
            (active.get("manifest") or {}).get("processing_version_ids") or
            [active["processing_version_id"]]) < set(scope),
        "all_processing_versions_approved": approved == len(scope),
        "database_counts_consistent": row[1] == row[2] == row[3] == len(chunk_ids) == 1078,
        "chunk_ids_unique": len(chunk_ids) == len(set(chunk_ids)),
        "chunk_hash_matches": hashlib.sha256("\n".join(chunk_ids).encode()).hexdigest()
                              == manifest["chunk_ids_sha256"],
        "disk_and_database_manifest_match": disk_manifest == manifest,
        "chroma_count_matches": chroma_count == len(chunk_ids),
        "bm25_count_matches": bm25_meta.get("count") == len(chunk_ids),
        "bm25_chunk_scope_matches": bm25_meta.get("chunk_ids") == chunk_ids,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "index_version_id": args.index_version_id,
        "current_active_index_id": active["index_version_id"],
        "processing_version_count": len(scope), "approved_processing_version_count": approved,
        "chunk_count": len(chunk_ids), "chroma_count": chroma_count,
        "bm25_count": bm25_meta.get("count"), "checks": checks,
        "external_api_calls": 0, "active_index_changed": False,
    }
    target = settings.project_root / args.report
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
