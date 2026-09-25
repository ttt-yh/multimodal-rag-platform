from multimodal_rag.infrastructure.retrieval import rrf_fuse


def test_rrf_fuses_two_rankings_deterministically():
    dense = [{"chunk_id": "a", "rank": 1}, {"chunk_id": "b", "rank": 2}]
    keyword = [{"chunk_id": "b", "rank": 1}, {"chunk_id": "c", "rank": 2}]
    first = rrf_fuse(dense, keyword, top_k=3)
    second = rrf_fuse(dense, keyword, top_k=3)
    assert first == second
    assert [row["chunk_id"] for row in first] == ["b", "a", "c"]
