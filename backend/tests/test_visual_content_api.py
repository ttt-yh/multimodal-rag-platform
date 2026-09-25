from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app


def test_visual_content_api_serves_only_validated_bytes(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.read_active_visual_element", lambda *args: {
        "content": b"\x89PNG\r\n\x1a\nimage", "mime_type": "image/png",
        "sha256": "a" * 64, "element_id": "el_demo"})
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/visual-elements/el_demo/content")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
