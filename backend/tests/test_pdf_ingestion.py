from io import BytesIO
import hashlib
import json
from zipfile import ZipFile

import pytest
import httpx2 as httpx
from pypdf import PdfWriter

from multimodal_rag.application.ingestion_worker import _pdf_page_count, _run_pdf
from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.documents import WhitelistDocumentReader
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.parser_normalization import (
    materialize_archive_assets, normalize_archive)
from multimodal_rag.infrastructure.settings import Settings


def _pdf_bytes(pages=2):
    output = BytesIO()
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    writer.write(output)
    return output.getvalue()


def _pdf_project(tmp_path, pages=2):
    source = tmp_path / "data/raw/manuals/guide.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(_pdf_bytes(pages))
    entry = {
        "document_id": "pdf_demo", "path": source.relative_to(tmp_path).as_posix(),
        "format": "pdf", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split": "dev", "title": "PDF 手册", "knowledge_base": "fixture",
        "source_family": "pdf-demo", "source_revision": "test", "license": "fixture",
    }
    manifest = tmp_path / "data/manifests/pdf.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(entry), encoding="utf-8")
    settings = Settings(_env_file=None, project_root=tmp_path,
                        ingestion_manifest="data/manifests/pdf.jsonl")
    return settings, source, entry


def _archive():
    output = BytesIO()
    rows = [
        {"type": "title", "text": "部署说明", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "先检查服务状态。", "page_idx": 0},
        {"type": "image", "image_caption": ["部署架构"],
         "img_path": "images/architecture.png", "page_idx": 1},
    ]
    with ZipFile(output, "w") as archive:
        archive.writestr("full.md", "# 部署说明")
        archive.writestr("result_content_list.json", json.dumps(rows, ensure_ascii=False))
        archive.writestr("images/architecture.png", b"\x89PNG\r\n\x1a\nfixture")
    return output.getvalue()


def test_pdf_is_discoverable_and_hash_verified(tmp_path):
    settings, source, _ = _pdf_project(tmp_path)
    reader = WhitelistDocumentReader(settings.project_root, settings.max_document_bytes,
                                     settings.ingestion_manifest, settings.max_pdf_bytes)
    assert [row.document_id for row in reader.list_allowed()] == ["pdf_demo"]
    entry, payload = reader.read_source("pdf_demo")
    assert entry.format == "pdf" and payload == source.read_bytes()
    assert _pdf_page_count(payload, settings) == 2
    with pytest.raises(AppError) as exc:
        preview_document("pdf_demo", settings)
    assert exc.value.code == "preview_format_unsupported"


def test_pdf_page_limit_is_enforced(tmp_path):
    settings, source, _ = _pdf_project(tmp_path, pages=2)
    limited = settings.model_copy(update={"max_pdf_pages": 1})
    with pytest.raises(AppError) as exc:
        _pdf_page_count(source.read_bytes(), limited)
    assert exc.value.code == "pdf_page_limit_exceeded"


def test_normalized_pdf_keeps_version_heading_and_materializes_images(tmp_path):
    settings, _, entry = _pdf_project(tmp_path)
    version_id = "ver_" + "b" * 64
    sample = {"sample_id": "job_pdf", "page_count": 2, "pages": [{
        "document_id": entry["document_id"], "version_id": version_id,
        "source_path": entry["path"], "source_sha256": entry["sha256"],
        "source_page": page, "split": "dev", "source_kind": "pdf",
    } for page in (1, 2)]}
    content = _archive()
    normalized = normalize_archive(content, sample)
    assert all(row["element"]["version_id"] == version_id for row in normalized["elements"])
    assert normalized["elements"][1]["element"]["heading_path"] == ["部署说明"]
    materialize_archive_assets(content, normalized, tmp_path, "proc_pdf")
    image_ref = normalized["elements"][2]["element"]["image_ref"]
    assert image_ref.startswith("data/derived/mineru_assets/proc_pdf/")
    assert (tmp_path / image_ref).read_bytes().startswith(b"\x89PNG")


def test_pdf_worker_submits_and_uploads_once_before_deferring(monkeypatch, tmp_path):
    settings, source, _ = _pdf_project(tmp_path)
    entry, payload = WhitelistDocumentReader(
        settings.project_root, settings.max_document_bytes,
        settings.ingestion_manifest, settings.max_pdf_bytes).read_source("pdf_demo")
    calls = []

    def handler(request):
        calls.append(request.method)
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["files"][0]["page_ranges"] == "1-2"
            return httpx.Response(200, json={"code": 0, "data": {
                "batch_id": "batch-pdf", "file_urls": [
                    "https://mineru.oss-cn-shanghai.aliyuncs.com/pdf?signature=secret"]}})
        assert request.content == source.read_bytes()
        return httpx.Response(200, content=b"")

    deferred = {}
    monkeypatch.setattr("multimodal_rag.application.ingestion_worker.defer_job",
                        lambda *args, **kwargs: deferred.update(kwargs))
    mock_settings = Settings(_env_file=None, project_root=settings.project_root,
                             ingestion_manifest=settings.ingestion_manifest,
                             mode="mock", parser_api_key="key")
    gateway = HttpGateway(mock_settings, "parser", budget=CallBudget(2),
                          transport=httpx.MockTransport(handler))
    try:
        result = _run_pdf(mock_settings,
            {"job_id": "job_pdf", "version_id": "ver_" + "b" * 64}, "worker", {
                "page_count": 2, "max_characters": 1200,
                "overlap_characters": 160, "min_characters": 100,
            }, entry, payload, None, gateway)
    finally:
        gateway.close()
    assert result["status"] == "pending" and result["stage"] == "parser_submitted"
    assert calls == ["POST", "PUT"]
    assert deferred["external_batch_id"] == "batch-pdf"


def test_pdf_worker_poll_does_not_resubmit_pending_batch(monkeypatch, tmp_path):
    settings, _, _ = _pdf_project(tmp_path)
    entry, payload = WhitelistDocumentReader(
        settings.project_root, settings.max_document_bytes,
        settings.ingestion_manifest, settings.max_pdf_bytes).read_source("pdf_demo")
    methods = []

    def handler(request):
        methods.append(request.method)
        return httpx.Response(200, json={"code": 0, "data": {"extract_result": [{
            "data_id": "job_pdf", "state": "running",
        }]}})

    deferred = {}
    monkeypatch.setattr("multimodal_rag.application.ingestion_worker.defer_job",
                        lambda *args, **kwargs: deferred.update(kwargs))
    mock_settings = Settings(_env_file=None, project_root=settings.project_root,
                             ingestion_manifest=settings.ingestion_manifest,
                             mode="mock", parser_api_key="key")
    gateway = HttpGateway(mock_settings, "parser", budget=CallBudget(1),
                          transport=httpx.MockTransport(handler))
    try:
        result = _run_pdf(mock_settings,
            {"job_id": "job_pdf", "version_id": "ver_" + "b" * 64}, "worker",
            {"page_count": 2}, entry, payload, "batch-pdf", gateway)
    finally:
        gateway.close()
    assert result == {"status": "pending", "stage": "parser_running", "progress": None}
    assert methods == ["GET"]
    assert deferred["external_batch_id"] == "batch-pdf"
