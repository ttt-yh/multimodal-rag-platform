"""Evidence-grounded RAG answer generation."""
from __future__ import annotations

from multimodal_rag.application.context_builder import build_context, expand_with_adjacent_chunks
from multimodal_rag.application.retrieval_service import retrieve
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.model_adapters import ChatAdapter
from multimodal_rag.infrastructure.settings import Settings


def answer(settings: Settings, query: str, *, max_context_chars: int = 12000,
           max_requests: int = 3) -> dict:
    """Retrieve evidence and generate one answer with a bounded API budget."""
    if not isinstance(max_requests, int) or not 3 <= max_requests <= 3:
        raise AppError("invalid_call_budget", "问答请求预算必须为3次：Embedding、Rerank和Chat", 422)

    # Retrieval uses two calls at most; Chat is deliberately a separate call
    # so retrieval diagnostics remain available even when generation fails.
    retrieval = retrieve(settings, query, max_requests=2)
    expansion = expand_with_adjacent_chunks(settings, retrieval["results"], radius=1, max_primary=3)
    context_results = expansion["results"]
    context = build_context(query, context_results, max_chars=max_context_chars)
    if not context["citations"]:
        return {"query": query, "answer": "当前知识库没有足够证据回答该问题。",
                "citations": [], "retrieval": retrieval["retrieval"],
                "index_version_id": retrieval["index_version_id"],
                "degraded": retrieval["degraded"],
                "external_calls": retrieval["external_calls"],
                "records": retrieval["records"], "context_expansion": {
                    "anchor_count": expansion["anchor_count"],
                    "adjacent_count": expansion["adjacent_count"],
                }}

    gateway = HttpGateway(settings, "chat", budget=CallBudget(1))
    try:
        result = ChatAdapter(gateway).complete(context["prompt"])
    finally:
        records = list(gateway.records)
        gateway.close()
    by_chunk = {item.get("chunk_id"): item for item in context_results}
    citations = []
    for citation in context["citations"]:
        source = by_chunk.get(citation.get("chunk_id"), {})
        citations.append({
            **citation,
            "text": source.get("text", ""),
            "metadata": source.get("metadata", {}),
            "rerank_score": source.get("rerank_score"),
            "rrf_score": source.get("rrf_score"),
        })
    return {
        "query": query, "answer": result["text"], "citations": citations,
        "retrieval": retrieval["retrieval"], "index_version_id": retrieval["index_version_id"],
        "degraded": retrieval["degraded"],
        "context_expansion": {"anchor_count": expansion["anchor_count"],
                              "adjacent_count": expansion["adjacent_count"]},
        "external_calls": retrieval["external_calls"] + len(records),
        "records": retrieval["records"] + records,
    }
