"""Optional Chroma and BM25s builders.

Both builders consume the same ordered Chunk list. They are intentionally
separate from the Embedding HTTP adapter so a failed keyword build cannot be
mistaken for a successful dense index.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from multimodal_rag.core.errors import AppError
from multimodal_rag.core.models import Chunk


def _metadata(chunk: Chunk) -> dict[str, Any]:
    source = chunk.source_locations[0]
    return {"document_id": chunk.document_id, "version_id": chunk.version_id,
            "processing_version_id": chunk.processing_version_id,
            "heading_path": " / ".join(chunk.heading_path),
            "source_path": source.source_path,
            "line_start": source.line_start or 0, "line_end": source.line_end or 0}


def build_chroma(root: Path, index_version_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict:
    if len(chunks) != len(vectors):
        raise ValueError("Chunk与向量数量不一致")
    if not chunks:
        raise ValueError("不能为零个Chunk构建索引")
    try:
        import chromadb
    except ImportError:
        raise AppError("dependency_missing", "未安装chromadb，请安装index依赖后再构建向量索引", 503) from None
    dimensions = len(vectors[0])
    if dimensions <= 0 or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("向量维度不一致")
    # Chroma 0.6 may still log PostHog compatibility errors even when
    # anonymized telemetry is disabled; index construction must stay offline.
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    logging.getLogger("posthog").disabled = True
    client = chromadb.PersistentClient(
        path=str(root / "chroma"),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=index_version_id, metadata={"hnsw:space": "cosine"})
    collection.upsert(ids=[c.chunk_id for c in chunks], documents=[c.text for c in chunks],
                      embeddings=vectors, metadatas=[_metadata(c) for c in chunks])
    return {"backend": "chroma", "index_version_id": index_version_id,
            "count": len(chunks), "dimensions": dimensions}


def load_chroma_embeddings(root: Path, index_version_id: str,
                           chunk_ids: list[str]) -> dict[str, list[float]]:
    """Read reusable vectors from an immutable Chroma index by Chunk ID."""
    if not chunk_ids:
        return {}
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("待读取的Chunk编号不能重复")
    try:
        import chromadb
    except ImportError:
        raise AppError("dependency_missing", "未安装chromadb，请安装index依赖后再读取向量索引", 503) from None
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    logging.getLogger("posthog").disabled = True
    try:
        client = chromadb.PersistentClient(
            path=str(root / "chroma"),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=index_version_id)
        result = collection.get(ids=chunk_ids, include=["embeddings"])
        ids = result.get("ids") or []
        embeddings = result.get("embeddings")
        embeddings = [] if embeddings is None else embeddings
    except (OSError, ValueError, KeyError, TypeError):
        raise AppError("index_artifact_invalid", "旧向量索引不可读取，不能执行增量构建", 409) from None
    if len(ids) != len(embeddings) or len(ids) != len(chunk_ids):
        raise AppError("index_artifact_invalid", "旧向量索引缺少需要复用的Chunk向量", 409)
    vectors = {str(chunk_id): [float(value) for value in vector]
               for chunk_id, vector in zip(ids, embeddings)}
    if set(vectors) != set(chunk_ids):
        raise AppError("index_artifact_invalid", "旧向量索引返回的Chunk范围不完整", 409)
    return vectors


def build_bm25(root: Path, index_version_id: str, chunks: list[Chunk]) -> dict:
    if not chunks:
        raise ValueError("不能为零个Chunk构建索引")
    try:
        import bm25s
    except ImportError:
        raise AppError("dependency_missing", "未安装bm25s，请安装index依赖后再构建关键词索引", 503) from None
    target = root / "bm25" / index_version_id
    target.mkdir(parents=True, exist_ok=True)
    # BM25s expects whitespace-delimited terms. Keep error codes/config keys
    # intact while making Chinese text searchable at character granularity.
    corpus = [" ".join(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", chunk.text.lower()))
              for chunk in chunks]
    corpus_tokens = bm25s.tokenize(corpus, token_pattern=r"(?u)\S+", stopwords=None,
                                   show_progress=False)
    retriever = bm25s.BM25(method="lucene", corpus=corpus)
    retriever.index(corpus_tokens, show_progress=False)
    retriever.save(str(target), show_progress=False)
    (target / "metadata.json").write_text(json.dumps({
        "index_version_id": index_version_id, "chunk_ids": [c.chunk_id for c in chunks],
        "tokenizer": "bm25s.tokenize", "count": len(chunks)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"backend": "bm25s", "index_version_id": index_version_id, "count": len(chunks)}
