"""在明确确认额度后，为阶段3B文本集合构建 draft Dense/BM25 索引。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from multimodal_rag.application.index_builder import build_real_indexes_for_processing_versions
from multimodal_rag.infrastructure.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段3B：构建多文档 draft 索引")
    parser.add_argument("--report", default="evals/results/stage3/stage3b_ingestion_report.json")
    parser.add_argument("--max-requests", type=int, required=True,
                        help="本批Embedding HTTP调用预算；665个Chunk按10条一批预计为67")
    parser.add_argument("--confirm-live", action="store_true",
                        help="明确确认真实Embedding调用；不提供则不会发起网络请求")
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("构建索引会调用真实Embedding，请提供 --confirm-live；未发起网络请求")

    settings = load_settings()
    report_path = settings.project_root / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("stage") != "3B" or report.get("status") != "succeeded":
        parser.error("阶段3B入库报告不存在或未成功，不能构建索引")
    processing_ids = report.get("processing_versions", {}).keys()
    processing_ids = sorted(processing_ids)
    chunks = int(report.get("chunk_count", 0))
    expected = math.ceil(chunks / 10)
    if not processing_ids or chunks <= 0:
        parser.error("阶段3B报告没有可索引的处理版本或Chunk")
    if args.max_requests < expected:
        parser.error(f"预算不足：当前约需{expected}次Embedding请求，实际只提供{args.max_requests}次")
    result = build_real_indexes_for_processing_versions(settings, processing_ids,
                                                        max_requests=args.max_requests)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
