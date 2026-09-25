"""Plan or build the Stage 5B index as an incremental extension.

Dry-run is the default and makes no external calls.  A live build requires an
explicit flag and request budget; it reuses every vector from the active index,
embeds only newly added Chunks, and leaves the result in draft status.
"""
from __future__ import annotations

import argparse
import json
import math

from multimodal_rag.application.index_builder import build_incremental_index_from_active
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段5B增量索引构建")
    parser.add_argument("--report", default="evals/results/stage5/stage5b_visual_ingestion_report.json")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--confirm-live", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    ingestion = json.loads((settings.project_root / args.report).read_text(encoding="utf-8"))
    if ingestion.get("status") != "succeeded":
        parser.error("阶段5B入库尚未全部成功")
    active = get_active_index(settings)
    active_scope = set((active.get("manifest") or {}).get("processing_version_ids") or
                       [active["processing_version_id"]])
    report_scope = set(ingestion.get("processing_versions") or {})
    added = sorted(report_scope - active_scope)
    if not added:
        parser.error("没有需要加入活动索引的新处理版本")
    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            """SELECT pv.processing_version_id,pv.quality_status,count(c.chunk_id)::int
                 FROM mrag.processing_versions pv
                 LEFT JOIN mrag.chunks c ON c.processing_version_id=pv.processing_version_id
                WHERE pv.processing_version_id=ANY(%s)
                GROUP BY pv.processing_version_id,pv.quality_status""",
            (added,),).fetchall()
    states = {row[0]: {"quality_status": row[1], "chunk_count": row[2]} for row in rows}
    if set(states) != set(added) or any(row["quality_status"] != "approved" for row in states.values()):
        parser.error("新增处理版本未全部通过质量审核")
    new_chunks = sum(row["chunk_count"] for row in states.values())
    expected_requests = math.ceil(new_chunks / 10)
    plan = {
        "status": "ready_for_confirmed_build",
        "build_mode": "incremental_reuse",
        "base_index_version_id": active["index_version_id"],
        "base_chunk_count": active["chunk_count"],
        "reused_vector_count_expected": active["vector_count"],
        "added_processing_version_count": len(added),
        "new_chunk_count": new_chunks,
        "target_chunk_count_expected": active["chunk_count"] + new_chunks,
        "embedding_batch_size": 10,
        "embedding_requests_required": expected_requests,
        "external_api_calls": 0,
        "active_index_changed": False,
    }
    if not args.confirm_live:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.max_requests < expected_requests:
        parser.error(f"Embedding预算不足：至少需要{expected_requests}次请求")
    result = build_incremental_index_from_active(
        settings, added, max_requests=args.max_requests)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
