"""Dense, BM25 and RRF read-side retrieval over one immutable index version."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from multimodal_rag.core.errors import AppError


def _load_chroma(root: Path, index_version_id: str):
    try:
        import chromadb
    except ImportError:
        raise AppError("dependency_missing", "未安装chromadb", 503) from None
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    logging.getLogger("posthog").disabled = True
    return chromadb.PersistentClient(path=str(root / "chroma"),
                                     settings=chromadb.config.Settings(anonymized_telemetry=False)) \
        .get_collection(index_version_id)


def dense_search(root: Path, index_version_id: str, vector: list[float], *, top_k: int = 5) -> list[dict]:
    collection = _load_chroma(root, index_version_id)
    result = collection.query(query_embeddings=[vector], n_results=top_k,
                              include=["documents", "metadatas", "distances"])
    return [{"chunk_id": chunk_id, "text": text, "metadata": metadata,
             "score": 1 - distance, "rank": rank + 1}
            for rank, (chunk_id, text, metadata, distance) in enumerate(zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]))]


def hydrate_chunks(root: Path, index_version_id: str, chunk_ids: list[str]) -> dict[str, dict]:
    """Read text and source metadata for fused IDs from the immutable Chroma index."""
    if not chunk_ids:
        return {}
    collection = _load_chroma(root, index_version_id)
    result = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    ids = result.get("ids", [])
    texts = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    if not (len(ids) == len(texts) == len(metadatas)):
        raise AppError("index_artifact_invalid", "向量索引返回的Chunk数据不完整", 409)
    return {str(chunk_id): {"text": text, "metadata": metadata or {}}
            for chunk_id, text, metadata in zip(ids, texts, metadatas)}


def bm25_search(root: Path, index_version_id: str, query: str, *, top_k: int = 5) -> list[dict]:
    try:
        import bm25s
    except ImportError:
        raise AppError("dependency_missing", "未安装bm25s", 503) from None
    target = root / "bm25" / index_version_id
    try:
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        retriever = bm25s.BM25.load(str(target), load_corpus=True)
        tokens = bm25s.tokenize([" ".join(__import__("re").findall(
            r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", query.lower()))],
            token_pattern=r"(?u)\S+", stopwords=None, show_progress=False)
        values, scores = retriever.retrieve(tokens, corpus=metadata["chunk_ids"], k=top_k,
                                            show_progress=False)
    except (OSError, ValueError, KeyError, TypeError):
        raise AppError("keyword_index_invalid", "关键词索引产物不可读取", 409) from None
    rows = []
    for rank, (value, score) in enumerate(zip(values[0], scores[0])):
        rows.append({"chunk_id": str(value), "score": float(score), "rank": rank + 1})
    return rows


def rrf_fuse(*ranked_lists: list[dict], k: int = 60, top_k: int = 5) -> list[dict]:
    fused: dict[str, dict] = {}
    for rows in ranked_lists:
        for row in rows:
            item = fused.setdefault(row["chunk_id"], {"chunk_id": row["chunk_id"], "rrf_score": 0.0})
            item["rrf_score"] += 1.0 / (k + row["rank"])
    return sorted(fused.values(), key=lambda row: (-row["rrf_score"], row["chunk_id"]))[:top_k]
