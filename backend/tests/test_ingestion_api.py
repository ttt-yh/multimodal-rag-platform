from fastapi.testclient import TestClient

from multimodal_rag.api.app import create_app


def test_ingestion_catalog_exposes_only_whitelisted_dev_documents(settings):
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/ingestion/catalog?limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total"] == 1
    assert body["documents"][0]["document_id"] == "doc_demo"
    assert body["documents"][0]["format"] == "md"


def test_create_ingestion_job_uses_document_id_not_file_path(monkeypatch, settings):
    def fake_enqueue(current_settings, document_id, *, profile=None):
        assert current_settings is settings
        assert document_id == "doc_demo"
        return {"job_id": "job_demo", "document_id": document_id,
                "version_id": "ver_demo", "processing_version_id": "proc_demo",
                "idempotency_key": "ingest:proc_demo", "profile": {}}

    monkeypatch.setattr("multimodal_rag.api.app.enqueue_document", fake_enqueue)
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/ingestion/jobs", json={"document_id": "doc_demo"})
    assert response.status_code == 200
    assert response.json()["job_id"] == "job_demo"


def test_run_ingestion_job_returns_current_state(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.run_once",
                        lambda current_settings, worker_id, *, job_id: {
                            "status": "succeeded", "job": {"job_id": job_id}
                        })
    monkeypatch.setattr("multimodal_rag.api.app.get_job",
                        lambda current_settings, job_id: {
                            "job_id": job_id, "status": "succeeded", "stage": "quality_checked"
                        })
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/ingestion/jobs/job_demo/run")
    assert response.status_code == 200
    assert response.json()["job"]["stage"] == "quality_checked"


def test_run_ingestion_job_is_idempotent_after_completion(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.run_once", lambda *args, **kwargs: None)
    monkeypatch.setattr("multimodal_rag.api.app.get_job",
                        lambda current_settings, job_id: {
                            "job_id": job_id, "status": "succeeded", "stage": "quality_checked"
                        })
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/ingestion/jobs/job_demo/run")
    assert response.status_code == 200
    assert response.json()["result"] is None
    assert response.json()["job"]["status"] == "succeeded"
