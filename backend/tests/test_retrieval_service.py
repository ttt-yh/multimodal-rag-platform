from pathlib import Path

import pytest

from multimodal_rag.core.errors import AppError
from multimodal_rag.application import retrieval_service as service
from multimodal_rag.infrastructure.settings import Settings


class FakeGateway:
    def __init__(self, settings, name, *, budget):
        self.settings = settings
        self.service = name
        self.records = [{"service": name, "outcome": "validated"}]

    def close(self):
        pass


class FakeEmbedding:
    def __init__(self, gateway):
        self.gateway = gateway

    def embed(self, texts):
        assert texts == ["数据库连接超时"]
        return {"vectors": [[0.1, 0.2]], "usage": None}


class FakeRerank:
    def __init__(self, gateway):
        self.gateway = gateway

    def rerank(self, query, documents, top_n):
        assert query == "数据库连接超时" and len(documents) == 2 and top_n == 2
        return {"results": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.4}]}


def configured_settings(tmp_path):
    artifact = tmp_path / "indexes" / "idx_active"
    artifact.mkdir(parents=True)
    return Settings(_env_file=None, project_root=tmp_path, mode="api",
                    api_enabled=True, embedding_api_key="e", rerank_api_key="r",
                    embedding_base_url="https://embedding.example/v1",
                    rerank_base_url="https://rerank.example/v1")


def active_index():
    return {"index_version_id": "idx_active", "processing_version_id": "proc",
            "embedding_model": "text-embedding-v4", "embedding_dimensions": 2,
            "bm25_version": "bm25s-v1", "chunk_count": 2, "vector_count": 2,
            "keyword_count": 2, "artifact_path": "indexes/idx_active",
            "status": "active", "manifest": {}}


def patch_common(monkeypatch):
    monkeypatch.setattr(service, "get_active_index", lambda settings: active_index())
    monkeypatch.setattr(service, "HttpGateway", FakeGateway)
    monkeypatch.setattr(service, "EmbeddingAdapter", FakeEmbedding)
    monkeypatch.setattr(service, "RerankAdapter", FakeRerank)
    monkeypatch.setattr(service, "dense_search", lambda *args, **kwargs: [
        {"chunk_id": "a", "rank": 1, "score": .8},
        {"chunk_id": "b", "rank": 2, "score": .7},
    ])
    monkeypatch.setattr(service, "bm25_search", lambda *args, **kwargs: [
        {"chunk_id": "b", "rank": 1, "score": 2.0},
        {"chunk_id": "a", "rank": 2, "score": 1.0},
    ])
    monkeypatch.setattr(service, "hydrate_chunks", lambda *args, **kwargs: {
        "a": {"text": "检查数据库地址和端口", "metadata": {"source_path": "a.md"}},
        "b": {"text": "检查连接超时配置", "metadata": {"source_path": "b.md"}},
    })


def test_retrieve_uses_hybrid_and_rerank(monkeypatch, tmp_path):
    patch_common(monkeypatch)
    result = service.retrieve(configured_settings(tmp_path), "数据库连接超时", fusion_top_k=2,
                              final_top_k=2, max_requests=2)
    assert result["index_version_id"] == "idx_active"
    assert result["results"][0]["chunk_id"] == "b"
    assert result["retrieval"]["rerank"] == "validated"
    assert result["external_calls"] == 2
    assert not result["degraded"]


def test_rerank_failure_returns_explicit_rrf_fallback(monkeypatch, tmp_path):
    patch_common(monkeypatch)

    class BrokenRerank(FakeRerank):
        def rerank(self, query, documents, top_n):
            raise AppError("provider_transport_error", "down", 502)

    monkeypatch.setattr(service, "RerankAdapter", BrokenRerank)
    result = service.retrieve(configured_settings(tmp_path), "数据库连接超时", fusion_top_k=2,
                              final_top_k=2, max_requests=2)
    assert result["degraded"]
    assert result["retrieval"]["rerank"] == "fallback:provider_transport_error"
    assert all(item["rerank_score"] is None for item in result["results"])


@pytest.mark.parametrize("query", ["", "   ", "x" * 2001])
def test_retrieve_rejects_invalid_query(query, tmp_path):
    with pytest.raises(AppError) as error:
        service.retrieve(configured_settings(tmp_path), query)
    assert error.value.status == 422
