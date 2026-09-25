import pytest

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.index_backends import build_chroma, build_bm25, load_chroma_embeddings


def test_index_builders_reject_empty_input(tmp_path):
    with pytest.raises(ValueError):
        build_chroma(tmp_path, "idx_fixture", [], [])
    with pytest.raises(ValueError):
        build_bm25(tmp_path, "idx_fixture", [])


def test_embedding_loader_rejects_duplicate_chunk_ids(tmp_path):
    with pytest.raises(ValueError):
        load_chroma_embeddings(tmp_path, "idx_fixture", ["chk_a", "chk_a"])
