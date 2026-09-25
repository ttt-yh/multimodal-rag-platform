"""Measure local retrieval latency without calling any external model.

The optional embedding JSON should contain one vector with the active index
dimension. Without it, Dense receives a zero vector and the report explicitly
marks the result synthetic; BM25/RRF/hydration timings remain representative
of local artifact access only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.retrieval import bm25_search, dense_search, hydrate_chunks, rrf_fuse
from multimodal_rag.infrastructure.settings import load_settings


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def measure(fn, iterations: int) -> tuple[object, list[float]]:
    result = None
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000)
    return result, samples


def main() -> int:
    parser = argparse.ArgumentParser(description="本地 Chroma/BM25/RRF 检索延迟基线，不调用外部模型")
    parser.add_argument("--query", default="数据库连接超时时应该检查哪些配置？")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--embedding-json", type=Path,
                        help="可选：包含当前索引查询向量的 JSON 文件")
    parser.add_argument("--output", type=Path,
                        help="可选：将报告写入指定 JSON 文件")
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100:
        parser.error("--iterations 必须在1到100之间")

    settings = load_settings()
    active = get_active_index(settings)
    root = (settings.project_root / active["artifact_path"]).resolve()
    vector = [0.0] * active["embedding_dimensions"]
    vector_source = "synthetic_zero_vector"
    if args.embedding_json:
        payload = json.loads(args.embedding_json.read_text(encoding="utf-8"))
        vector = payload.get("vector", payload) if isinstance(payload, (dict, list)) else []
        if not isinstance(vector, list) or len(vector) != active["embedding_dimensions"]:
            parser.error("embedding JSON 必须是当前索引维度的向量，或包含 vector 数组")
        vector_source = "provided_embedding"

    dense_rows, dense_ms = measure(
        lambda: dense_search(root, active["index_version_id"], vector, top_k=10), args.iterations)
    keyword_rows, keyword_ms = measure(
        lambda: bm25_search(root, active["index_version_id"], args.query, top_k=10), args.iterations)
    fused, fusion_ms = measure(
        lambda: rrf_fuse(dense_rows, keyword_rows, top_k=5), args.iterations)
    fused_rows = fused
    _, hydrate_ms = measure(
        lambda: hydrate_chunks(root, active["index_version_id"],
                               [row["chunk_id"] for row in fused_rows]), args.iterations)

    report = {
        "status": "measured",
        "scope": "local_retrieval_only",
        "index_version_id": active["index_version_id"],
        "chunk_count": active["chunk_count"],
        "query": args.query,
        "iterations": args.iterations,
        "vector_source": vector_source,
        "external_model_calls": 0,
        "metrics_ms": {
            name: {"p50": round(statistics.median(samples), 3),
                   "p95": round(percentile(samples, 0.95), 3),
                   "samples": len(samples)}
            for name, samples in (("dense", dense_ms), ("bm25", keyword_ms),
                                  ("rrf", fusion_ms), ("hydrate", hydrate_ms))
        },
        "result_counts": {"dense": len(dense_rows), "bm25": len(keyword_rows), "fused": len(fused_rows)},
        "notice": "未提供查询向量时，Dense指标使用零向量，只能作为本地访问基线；不代表语义召回质量。",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
