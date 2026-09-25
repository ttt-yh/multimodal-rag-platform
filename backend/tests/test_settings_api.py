import io
import json
import logging

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from multimodal_rag.api.app import create_app
from multimodal_rag.infrastructure.logging import LOGGER
from multimodal_rag.infrastructure.settings import Settings


def test_env_file_and_environment_precedence(tmp_path, monkeypatch):
    config = tmp_path / ".env"
    config.write_text("MRAG_PORT=8123\nMRAG_CHAT_API_KEY=private-key\n", encoding="utf-8")
    monkeypatch.setenv("MRAG_PORT", "8124")
    settings = Settings(_env_file=config)
    assert settings.port == 8124
    assert "private-key" not in str(settings)


@pytest.mark.parametrize("field,value", [("port", 0), ("port", 99999), ("mode", "auto"),
    ("host", "0.0.0.0"), ("log_level", "DEBUG"), ("max_document_bytes", -1)])
def test_invalid_settings(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize("proxy", ["socks5://127.0.0.1:7890", "http://user:pass@127.0.0.1:7890",
    "http://127.0.0.1:7890/?secret=1", "http://127.0.0.1"])
def test_invalid_proxy_is_rejected(proxy):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, https_proxy=proxy)


def test_unknown_dotenv_field_has_no_value_leak(tmp_path):
    config = tmp_path / ".env"
    config.write_text("MRAG_UNKNOWN=SECRET-VALUE", encoding="utf-8")
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=config)
    assert "SECRET-VALUE" not in str(error.value)


@pytest.mark.parametrize("mode", ["offline", "mock", "api"])
def test_alive_is_not_ready(settings, mode):
    settings = settings.model_copy(update={"mode": mode})
    with TestClient(create_app(settings)) as client:
        alive = client.get("/health/live")
        assert alive.status_code == 200 and alive.json()["mode"] == mode
        assert alive.json()["phase"] == "v1.1-pdf-ingestion"
        ready = client.get("/health/ready")
        assert ready.status_code == 503 and ready.json()["ready"] is False
        assert ready.json()["external_calls"] == 0
        assert all(s["status"] == "not_configured" for s in ready.json()["services"].values())
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200


def test_configured_does_not_mean_connected(source_project):
    settings = Settings(_env_file=None, project_root=source_project[0],
                        chat_api_key="private-key", postgres_dsn="password=private-pw")
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/dependencies")
        assert response.json()["services"]["chat"]["status"] == "configuration_incomplete"
        assert response.json()["services"]["postgres"]["connectivity_checked"] is False
        assert "private-key" not in response.text and "private-pw" not in response.text


def test_preview_response_and_request_id(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/preview", json={"document_id": "doc_demo"})
        assert response.status_code == 200
        result = response.json()
        assert result["source_verified"] and result["persisted"] is False
        assert result["external_calls"] == 0
        assert result["version"]["status"] == "draft"
        assert result["job"]["operation"] == "preview"
        assert len(response.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize("body", [{"document_id": "../../.env"}, {"path": ".env"},
    {"document_id": "doc_demo", "path": "SECRET"}, {"document_id": "x" * 129}])
def test_invalid_requests_are_sanitized(settings, body):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/preview", json=body)
        assert response.status_code == 422
        assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
        assert "SECRET" not in response.text


def test_unknown_document(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/preview", json={"document_id": "unknown"})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "document_not_allowed"


def test_logs_exclude_headers_queries_and_payloads(settings):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    LOGGER.addHandler(handler)
    try:
        with TestClient(create_app(settings)) as client:
            client.get("/health/live?api_key=QUERY-SECRET", headers={
                "Authorization": "Bearer AUTH-SECRET", "X-Request-ID": "HEADER-SECRET"})
            client.get("/SECRET-PATH")
            client.post("/api/v1/preview", json={"bad": "BODY-SECRET"})
        records = stream.getvalue()
        assert all(s not in records for s in ["QUERY-SECRET", "AUTH-SECRET", "HEADER-SECRET", "BODY-SECRET", "SECRET-PATH"])
        assert len([json.loads(line) for line in records.splitlines()]) == 3
    finally:
        LOGGER.removeHandler(handler)


def test_unexpected_exception_does_not_leak(settings, monkeypatch):
    def broken(*args):
        raise RuntimeError("password=DO-NOT-EXPOSE")
    monkeypatch.setattr("multimodal_rag.api.app.preview_document", broken)
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/v1/preview", json={"document_id": "doc_demo"})
        assert response.status_code == 500
        assert "DO-NOT-EXPOSE" not in response.text
        assert response.headers["X-Request-ID"]
