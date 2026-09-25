"""Index-version metadata and immutable artifact manifest.

This module does not call Embedding, Chroma or BM25. It freezes the inputs
needed by those builders so a later failed build cannot alter an active index.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from multimodal_rag.core.models import Chunk


def index_version_id(processing_version_id: str, embedding_model: str, dimensions: int,
                     bm25_version: str) -> str:
    value = f"{processing_version_id}:{embedding_model}:{dimensions}:{bm25_version}".encode()
    return "idx_" + hashlib.sha256(value).hexdigest()[:48]


def build_manifest(processing_version_id: str, chunks: list[Chunk], *, embedding_model: str,
                   dimensions: int, bm25_version: str,
                   processing_version_ids: list[str] | None = None) -> dict[str, Any]:
    if dimensions <= 0 or not embedding_model or not bm25_version:
        raise ValueError("索引配置不完整")
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk 编号必须唯一")
    scope_ids = sorted(set(processing_version_ids or [processing_version_id]))
    if not scope_ids or processing_version_id not in scope_ids:
        raise ValueError("索引范围必须包含数据库兼容锚点处理版本")
    scope_key = "|".join(scope_ids)
    return {
        # 单处理版本保持原编号规则；多文档索引按完整输入集合生成不可变编号。
        "index_version_id": index_version_id(scope_key, embedding_model, dimensions, bm25_version),
        "processing_version_id": processing_version_id,
        "processing_version_ids": scope_ids,
        "embedding": {"model": embedding_model, "dimensions": dimensions},
        "keyword": {"engine": "bm25s", "version": bm25_version},
        "chunk_count": len(chunks), "vector_count": 0, "keyword_count": 0,
        "status": "draft", "length_unit": chunks[0].length_unit if chunks else "characters",
        "chunk_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }


def write_manifest(root: Path, manifest: dict[str, Any]) -> Path:
    target = root / "indexes" / manifest["index_version_id"] / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target
