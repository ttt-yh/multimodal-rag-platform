from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app


def test_review_queue_and_index_list(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.list_review_queue",
                        lambda current_settings, *, state, limit, offset: {
                            "items": [{"processing_version_id": "proc_demo", "quality_status": "candidate"}],
                            "filter": state, "pagination": {"limit": limit, "offset": offset, "returned": 1, "total": 1},
                        })
    monkeypatch.setattr("multimodal_rag.api.app.list_indexes",
                        lambda current_settings, *, status, limit: {
                            "items": [{"index_version_id": "idx_demo", "status": status}],
                            "filter": status,
                        })
    with TestClient(create_app(settings)) as client:
        queue = client.get("/api/v1/review-queue?limit=10")
        indexes = client.get("/api/v1/indexes?status=draft")
    assert queue.status_code == 200
    assert queue.json()["items"][0]["quality_status"] == "candidate"
    assert indexes.status_code == 200
    assert indexes.json()["items"][0]["status"] == "draft"


def test_build_index_requires_explicit_live_confirmation(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/indexes/build", json={
            "processing_version_ids": ["proc_demo"], "max_requests": 1,
        })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "live_confirmation_required"


def test_build_index_passes_processing_scope_and_budget(monkeypatch, settings):
    def fake_build(current_settings, processing_ids, *, max_requests):
        assert current_settings is settings
        assert processing_ids == ["proc_demo"]
        assert max_requests == 2
        return {"status": "built", "index_version_id": "idx_demo"}

    monkeypatch.setattr("multimodal_rag.application.index_builder.build_real_indexes_for_processing_versions", fake_build)
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/indexes/build", json={
            "processing_version_ids": ["proc_demo"], "max_requests": 2, "confirm_live": True,
        })
    assert response.status_code == 200
    assert response.json()["index_version_id"] == "idx_demo"


def test_incremental_build_uses_active_index_reuse(monkeypatch, settings):
    def fake_incremental(current_settings, processing_ids, *, max_requests):
        assert current_settings is settings
        assert processing_ids == ["proc_pdf"]
        assert max_requests == 1
        return {"status": "built", "build_mode": "incremental_reuse",
                "index_version_id": "idx_incremental"}

    monkeypatch.setattr(
        "multimodal_rag.application.index_builder.build_incremental_index_from_active",
        fake_incremental,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/indexes/build", json={
            "processing_version_ids": ["proc_pdf"], "max_requests": 1,
            "confirm_live": True, "build_mode": "incremental_from_active",
        })
    assert response.status_code == 200
    assert response.json()["build_mode"] == "incremental_reuse"


def test_activate_index_requires_explicit_confirmation(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/indexes/idx_demo/activate", json={
            "reviewer": "human-review", "notes": "已核对产物",
        })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "activation_confirmation_required"


def test_document_retirement_requires_confirmation(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/documents/doc_demo/retire", json={
            "reviewer": "human-review", "notes": "资料已过期",
        })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "retirement_confirmation_required"


def test_document_retirement_returns_audit_result(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.retire_document",
                        lambda current_settings, document_id, reviewer, notes: {
                            "status": "retired", "document_id": document_id,
                            "reviewer": reviewer, "notes": notes,
                            "requires_index_rebuild": True,
                        })
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/documents/doc_demo/retire", json={
            "reviewer": "human-review", "notes": "资料已过期", "confirm": True,
        })
    assert response.status_code == 200
    assert response.json()["requires_index_rebuild"] is True


def test_rebuild_without_retired_uses_explicit_mode(monkeypatch, settings):
    def fake_incremental(current_settings, processing_ids, *, max_requests, exclude_retired):
        assert current_settings is settings
        assert processing_ids == []
        assert max_requests == 1
        assert exclude_retired is True
        return {"status": "built", "build_mode": "retire_rebuild"}

    monkeypatch.setattr(
        "multimodal_rag.application.index_builder.build_incremental_index_from_active",
        fake_incremental,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/indexes/build", json={
            "max_requests": 1,
            "confirm_live": True, "build_mode": "rebuild_without_retired",
        })
    assert response.status_code == 200
    assert response.json()["build_mode"] == "retire_rebuild"
