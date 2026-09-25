import json

import pytest

from multimodal_rag.core.errors import AppError
from multimodal_rag.application.preview import preview_document
from multimodal_rag.infrastructure.documents import WhitelistDocumentReader
from multimodal_rag.infrastructure.upload_repository import register_upload


def test_runtime_upload_is_hash_verified_and_discoverable(settings):
    payload = "# 上传文档\n\n用于验证网页入库。\n".encode("utf-8")
    entry = register_upload(settings, "研发说明.md", payload, knowledge_base="研发知识库")

    assert entry.document_id.startswith("upload_")
    assert entry.path.startswith("data/raw/uploads/")
    reader = WhitelistDocumentReader(
        settings.project_root, settings.max_document_bytes, settings.ingestion_manifest,
        settings.max_pdf_bytes, settings.runtime_upload_manifest,
    )
    loaded, content = reader.read_source(entry.document_id)
    assert loaded == entry
    assert content == payload
    assert {item.document_id for item in reader.list_allowed()} == {"doc_demo", entry.document_id}
    preview = preview_document(entry.document_id, settings)
    assert preview.source_verified and preview.document.source_family == "web_upload"


def test_same_logical_filename_replaces_runtime_revision_but_keeps_old_blob(settings):
    first = register_upload(settings, "guide.txt", b"first", knowledge_base="kb")
    second = register_upload(settings, "guide.txt", b"second", knowledge_base="kb")

    assert first.document_id == second.document_id
    assert first.sha256 != second.sha256
    manifest = settings.project_root / settings.runtime_upload_manifest
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["sha256"] == second.sha256
    assert (settings.project_root / first.path).exists()
    assert (settings.project_root / second.path).exists()


@pytest.mark.parametrize("filename", ["../secret.md", "C:/secret.md", "a.exe", "folder/a.md"])
def test_unsafe_or_unsupported_upload_names_are_rejected(settings, filename):
    with pytest.raises(AppError):
        register_upload(settings, filename, b"content")


def test_non_utf8_text_upload_is_rejected(settings):
    with pytest.raises(AppError) as error:
        register_upload(settings, "bad.txt", b"\xff\xfe")
    assert error.value.code == "invalid_encoding"
