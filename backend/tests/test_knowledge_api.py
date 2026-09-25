from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app


def test_knowledge_overview_api(monkeypatch, settings):
    def fake_list(current_settings, *, status, limit, offset):
        assert current_settings is settings
        return {"documents": [{"document_id": "doc_demo", "title": "示例文档",
                                "chunk_count": 20, "index": {"status": "active"}}],
                "summary": {"total_documents": 1, "approved_processing_versions": 1,
                            "active_indexes": 1},
                "pagination": {"limit": limit, "offset": offset, "returned": 1, "total": 1},
                "filter": status}

    monkeypatch.setattr("multimodal_rag.api.app.list_knowledge", fake_list)
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/knowledge?status=active&limit=20")
    assert response.status_code == 200
    assert response.json()["summary"]["active_indexes"] == 1
    assert response.json()["filter"] == "active"


def test_knowledge_filter_is_validated(settings):
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/knowledge?status=unknown")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
