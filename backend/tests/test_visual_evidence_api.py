from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app


def test_visual_evidence_api_uses_retrieved_chunk_scope(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.locate_visual_evidence", lambda *args, **kwargs: {
        "chunk_ids": ["chk_a"], "candidates": [], "ready_count": 0,
        "candidate_count": 0, "max_images": 2,
    })
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/visual-evidence/locate", json={
            "chunk_ids": ["chk_a"], "max_images": 2})
    assert response.status_code == 200
    assert response.json()["max_images"] == 2


def test_visual_evidence_api_rejects_duplicate_chunk_ids(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/visual-evidence/locate", json={
            "chunk_ids": ["chk_a", "chk_a"], "max_images": 2})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_chunk_scope"
