from types import SimpleNamespace

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


def test_browser_upload_registers_file_and_creates_job(monkeypatch, settings):
    captured = {}

    def fake_register(current_settings, filename, payload, *, knowledge_base, title):
        captured.update(filename=filename, payload=payload, knowledge_base=knowledge_base, title=title)
        return SimpleNamespace(document_id="upload_demo", title="上传示例", format="md",
                               path="data/raw/uploads/upload_demo/a.md",
                               knowledge_base=knowledge_base, sha256="a" * 64)

    monkeypatch.setattr("multimodal_rag.api.app.register_upload", fake_register)
    monkeypatch.setattr("multimodal_rag.api.app.enqueue_document",
                        lambda current_settings, document_id: {"job_id": "job_upload",
                            "document_id": document_id, "format": "md"})
    monkeypatch.setattr("multimodal_rag.api.app.run_once",
                        lambda current_settings, worker_id, *, job_id: captured.update(
                            background_job_id=job_id))
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/ingestion/uploads?filename=sample.md&knowledge_base=team&title=上传示例",
            content=b"# sample", headers={"Content-Type": "application/octet-stream"},
        )
    assert response.status_code == 201
    assert response.json()["job"]["job_id"] == "job_upload"
    assert response.json()["processing"] == "background"
    assert response.json()["upload"]["document_id"] == "upload_demo"
    assert captured == {"filename": "sample.md", "payload": b"# sample",
                        "knowledge_base": "team", "title": "上传示例",
                        "background_job_id": "job_upload"}


def test_pdf_browser_upload_does_not_start_unconfirmed_background_call(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.register_upload", lambda *args, **kwargs:
                        SimpleNamespace(document_id="upload_pdf", title="PDF", format="pdf",
                            path="data/raw/uploads/upload_pdf/a.pdf", knowledge_base="team",
                            sha256="b" * 64))
    monkeypatch.setattr("multimodal_rag.api.app.enqueue_document", lambda *args, **kwargs:
                        {"job_id": "job_pdf", "document_id": "upload_pdf", "format": "pdf"})
    monkeypatch.setattr("multimodal_rag.api.app.run_once", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("PDF must await confirmation")))
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/ingestion/uploads?filename=sample.pdf",
                               content=b"%PDF-sample")
    assert response.status_code == 201
    assert response.json()["processing"] == "awaiting_external_confirmation"


def test_browser_upload_rejects_oversized_declared_body(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/ingestion/uploads?filename=sample.pdf",
            content=b"small", headers={"Content-Length": str(settings.max_pdf_bytes + 1)},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "document_too_large"


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


def test_pdf_run_requires_explicit_live_confirmation(monkeypatch, settings):
    monkeypatch.setattr("multimodal_rag.api.app.get_job",
                        lambda current_settings, job_id: {
                            "job_id": job_id, "document_id": "pdf_demo",
                            "status": "pending", "stage": "queued",
                        })
    monkeypatch.setattr(
        "multimodal_rag.api.app.WhitelistDocumentReader.read_source",
        lambda self, document_id: (SimpleNamespace(format="pdf"), b"%PDF-fixture"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/ingestion/jobs/job_pdf/run", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "live_confirmation_required"
