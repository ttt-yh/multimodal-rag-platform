"""记录阶段3B多文档审核并激活指定索引。

该命令只允许在明确确认后执行；审核范围来自索引manifest中的全部处理版本，
避免多文档索引只批准一个锚点版本而误发布未审核内容。
"""
from __future__ import annotations

import argparse
import hashlib
import json

from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.index_lifecycle import activate_index
from multimodal_rag.infrastructure.settings import load_settings
from psycopg.types.json import Jsonb


def main() -> int:
    parser = argparse.ArgumentParser(description="审核并激活阶段3B多文档索引")
    parser.add_argument("--index-version-id", required=True)
    parser.add_argument("--reviewer", default="human-review")
    parser.add_argument("--notes", default="已核对阶段3B入库范围、Chunk来源映射和索引产物；确认发布该多文档draft索引。")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("该命令会批准处理版本并激活索引，请提供 --confirm")
    if not args.reviewer or len(args.reviewer) > 128 or not args.notes or len(args.notes) > 4000:
        parser.error("reviewer或notes长度不合法")

    settings = load_settings()
    with connection(settings) as conn:
        row = conn.execute(
            "SELECT manifest,status FROM mrag.index_versions WHERE index_version_id=%s FOR UPDATE",
            (args.index_version_id,),
        ).fetchone()
        if not row:
            parser.error("索引版本不存在")
        manifest, status = row
        processing_ids = sorted(set((manifest or {}).get("processing_version_ids", [])))
        if not processing_ids:
            parser.error("索引manifest没有处理版本范围")
        found = conn.execute(
            "SELECT processing_version_id FROM mrag.processing_versions WHERE processing_version_id = ANY(%s)",
            (processing_ids,),
        ).fetchall()
        if len(found) != len(processing_ids):
            parser.error("索引manifest引用了不存在的处理版本")
        for processing_id in processing_ids:
            review_id = "review_" + hashlib.sha256(
                f"{args.index_version_id}:{processing_id}".encode()).hexdigest()[:48]
            conn.execute(
                "INSERT INTO mrag.quality_reviews(review_id,processing_version_id,reviewer,decision,notes) "
                "VALUES (%s,%s,%s,'approved',%s) ON CONFLICT (review_id) DO NOTHING",
                (review_id, processing_id, args.reviewer, args.notes),
            )
            conn.execute(
                "UPDATE mrag.processing_versions SET quality_status='approved',updated_at=now() "
                "WHERE processing_version_id=%s", (processing_id,),
            )

    result = activate_index(settings, args.index_version_id)
    result.update({"reviewer": args.reviewer, "reviewed_processing_versions": len(processing_ids),
                   "previous_status": status})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
