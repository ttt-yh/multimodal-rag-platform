"""Application service for the read side of the published RAG index.

The service deliberately keeps retrieval separate from answer generation:
one query spends at most one Embedding and one Rerank request, while Dense,
BM25 and RRF execute locally against the immutable active index.
"""
from __future__ import annotations

from pathlib import Path

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.index_repository import get_active_index
from multimodal_rag.infrastructure.model_adapters import EmbeddingAdapter, RerankAdapter
from multimodal_rag.infrastructure.retrieval import (
    bm25_search,
    dense_search,
    hydrate_chunks,
    rrf_fuse,
)
from multimodal_rag.infrastructure.settings import Settings


def _validate_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise AppError("empty_query", "查询文本不能为空", 422)
    query = query.strip()
    if len(query) > 2000:
        raise AppError("query_too_long", "查询文本超过长度限制", 422)
    return query


def _artifact_root(settings: Settings, artifact_path: str) -> Path:
    root = (settings.project_root / artifact_path).resolve()
    if not root.is_relative_to(settings.project_root) or not root.is_dir():
        raise AppError("index_artifact_missing", "已激活索引的产物目录不存在", 409)
    return root


def retrieve(settings: Settings, query: str, *, dense_top_k: int = 10,
             keyword_top_k: int = 10, fusion_top_k: int = 10,
             final_top_k: int = 5, max_requests: int = 2) -> dict:
    """Run query embedding -> hybrid retrieval -> optional remote rerank.

    Rerank failures intentionally degrade to the deterministic RRF order.  A
    client can therefore inspect ``degraded`` and decide whether to generate
    an answer or ask the user for a narrower query.
    """
    query = _validate_query(query)
    if not all(isinstance(value, int) and 1 <= value <= 20
               for value in (dense_top_k, keyword_top_k, fusion_top_k, final_top_k)):
        raise AppError("invalid_retrieval_parameters", "检索数量必须在1到20之间", 422)
    if final_top_k > fusion_top_k:
        raise AppError("invalid_retrieval_parameters", "最终结果数不能大于融合候选数", 422)

    index = get_active_index(settings)
    if not (index["chunk_count"] == index["vector_count"] == index["keyword_count"]):
        raise AppError("index_incomplete", "已激活索引的数量校验未通过", 409)
    artifact_root = _artifact_root(settings, index["artifact_path"])
    budget = CallBudget(max_requests)
    records: list[dict] = []

    embedding_gateway = HttpGateway(settings, "embedding", budget=budget)
    try:
        embedding = EmbeddingAdapter(embedding_gateway).embed([query])
    finally:
        records.extend(embedding_gateway.records)
        embedding_gateway.close()

    dense = dense_search(artifact_root, index["index_version_id"], embedding["vectors"][0],
                         top_k=dense_top_k)
    keyword = bm25_search(artifact_root, index["index_version_id"], query, top_k=keyword_top_k)
    fused = rrf_fuse(dense, keyword, top_k=fusion_top_k)
    details = hydrate_chunks(artifact_root, index["index_version_id"],
                             [row["chunk_id"] for row in fused])

    candidates = []
    for row in fused:
        detail = details.get(row["chunk_id"])
        if not detail or not isinstance(detail.get("text"), str) or not detail["text"].strip():
            continue
        candidates.append({**row, "text": detail["text"], "metadata": detail["metadata"]})
    if not candidates:
        return {"query": query, "index_version_id": index["index_version_id"],
                "retrieval": {"dense_count": len(dense), "keyword_count": len(keyword),
                               "fused_count": 0, "rerank": "skipped"},
                "results": [], "degraded": False, "external_calls": len(records),
                "records": records}

    rerank_status = "validated"
    selected = candidates[:final_top_k]
    rerank_gateway = HttpGateway(settings, "rerank", budget=budget)
    try:
        result = RerankAdapter(rerank_gateway).rerank(
            query, [item["text"] for item in candidates], final_top_k)
        selected = []
        for item in result["results"]:
            candidate = candidates[item["index"]]
            selected.append({**candidate, "rerank_score": item["score"]})
    except AppError as exc:
        # Retrieval remains useful when the optional remote reranker is down
        # or this request has exhausted its explicit API budget.
        rerank_status = f"fallback:{exc.code}"
        selected = [{**item, "rerank_score": None} for item in selected]
    finally:
        records.extend(rerank_gateway.records)
        rerank_gateway.close()

    public_results = []
    for item in selected:
        public_results.append({
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "metadata": item["metadata"],
            "rrf_score": item["rrf_score"],
            "rerank_score": item.get("rerank_score"),
        })
    return {
        "query": query,
        "index_version_id": index["index_version_id"],
        "retrieval": {"dense_count": len(dense), "keyword_count": len(keyword),
                       "fused_count": len(candidates), "rerank": rerank_status},
        "results": public_results,
        "degraded": rerank_status != "validated",
        "external_calls": len(records),
        "records": records,
    }
