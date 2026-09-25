from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.chunking import split_elements
from multimodal_rag.infrastructure.index_manifest import build_manifest, index_version_id


def test_index_manifest_freezes_chunk_set(settings):
    preview = preview_document("doc_demo", settings)
    chunks = split_elements(preview.elements, "proc_fixture", max_characters=160, overlap_characters=20)
    manifest = build_manifest("proc_fixture", chunks, embedding_model="text-embedding-v4",
                              dimensions=1024, bm25_version="bm25s-v1")
    assert manifest["index_version_id"] == index_version_id("proc_fixture", "text-embedding-v4", 1024, "bm25s-v1")
    assert manifest["chunk_count"] == len(chunks)
    assert manifest["vector_count"] == 0
    assert manifest["status"] == "draft"


def test_multi_document_manifest_keeps_full_processing_scope(settings):
    preview = preview_document("doc_demo", settings)
    chunks = split_elements(preview.elements, "proc_a", max_characters=160, overlap_characters=20)
    manifest = build_manifest("proc_a", chunks, embedding_model="text-embedding-v4",
                              dimensions=1024, bm25_version="bm25s-v1",
                              processing_version_ids=["proc_b", "proc_a"])
    assert manifest["processing_version_id"] == "proc_a"
    assert manifest["processing_version_ids"] == ["proc_a", "proc_b"]
    assert manifest["index_version_id"] != index_version_id("proc_a", "text-embedding-v4", 1024, "bm25s-v1")
