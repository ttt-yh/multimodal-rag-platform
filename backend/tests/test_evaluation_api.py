from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app


def test_evaluations_api(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.list_evaluations", lambda *args, **kwargs: {
        "runs": [], "summary": {"total_runs": 0}, "quality_metrics": None,
        "note": "no metrics", "skipped_reports": 0,
    })
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/evaluations?kind=qa&limit=10")
    assert response.status_code == 200
    assert response.json()["quality_metrics"] is None


def test_evaluation_kind_is_validated(settings):
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/evaluations?kind=benchmark")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
