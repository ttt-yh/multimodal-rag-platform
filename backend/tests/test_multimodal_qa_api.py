from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app


def test_multimodal_qa_api(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.answer_multimodal", lambda *args, **kwargs: {
        "query": "看图", "answer": "答案", "answer_mode": "vision", "visual_evidence": []})
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/qa/multimodal", json={
            "query": "看图", "max_context_chars": 12000, "max_images": 2, "max_requests": 3})
    assert response.status_code == 200
    assert response.json()["answer_mode"] == "vision"
