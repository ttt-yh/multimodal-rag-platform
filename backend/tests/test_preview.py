import hashlib
import json

import pytest
from pydantic import ValidationError

from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.errors import AppError
from multimodal_rag.core.models import PreviewResult, SourceLocation
from multimodal_rag.infrastructure.settings import PROJECT_ROOT, Settings
from multimodal_rag.infrastructure.text_parser import verify_sources


def rewrite_entry(project, **changes):
    entry = dict(project[3], **changes)
    project[2].write_text(json.dumps(entry), encoding="utf-8")


def test_markdown_elements_and_round_trip(settings, source_project):
    result = preview_document("doc_demo", settings)
    assert {e.kind for e in result.elements} == {"heading", "text", "table", "image", "code"}
    assert PreviewResult.model_validate_json(result.model_dump_json()) == result
    assert result.elements[0].source.line_start == 4  # front matter 未改变原行号
    assert all(e.source.page_number is None and e.source.bbox is None for e in result.elements)
    assert len([e for e in result.elements if e.kind == "heading"]) == 2
    assert "<unknown>" in next(e.raw_text for e in result.elements if e.kind == "code")
    table = next(e for e in result.elements if e.kind == "table")
    assert "timeout" in table.raw_text and table.heading_path == ["部署说明", "参数"]
    assert len([e for e in result.elements if e.kind == "image"]) == 2
    verify_sources(result.document, result.version, source_project[1].read_text(encoding="utf-8"), result.elements)


def test_repeated_preview_has_stable_elements(settings):
    a, b = preview_document("doc_demo", settings), preview_document("doc_demo", settings)
    assert a.elements == b.elements and a.version == b.version
    assert a.job.job_id != b.job.job_id


def test_preview_can_use_explicit_stage_manifest(settings, source_project):
    stage_manifest = source_project[0] / "data/manifests/ingestion_chinese_md_stage3.jsonl"
    stage_manifest.write_text(source_project[2].read_text(encoding="utf-8"), encoding="utf-8")
    stage_settings = settings.model_copy(update={
        "ingestion_manifest": "data/manifests/ingestion_chinese_md_stage3.jsonl"
    })
    result = preview_document("doc_demo", stage_settings)
    assert result.source_verified is True


def test_manifest_path_cannot_escape_manifests(settings):
    with pytest.raises(AppError) as error:
        preview_document("doc_demo", settings.model_copy(update={
            "ingestion_manifest": "data/manifests/../raw/tidb_zh/source/sample.md"
        }))
    assert error.value.code == "unsafe_manifest"


def test_changed_source_is_rejected(settings, source_project):
    source_project[1].write_text("changed", encoding="utf-8")
    with pytest.raises(AppError, match="哈希不一致"):
        preview_document("doc_demo", settings)


@pytest.mark.parametrize("path", ["../../.env", "C:/Users/file.txt", "//server/share/a.md",
    "data/raw/tidb_zh/source/../secret.md", "data/annotations/answers.jsonl",
    "data/raw/tidb_zh/source/sample.md:stream", "data\\raw\\sample.md"])
def test_unsafe_paths_rejected(settings, source_project, path):
    rewrite_entry(source_project, path=path)
    with pytest.raises(AppError) as error:
        preview_document("doc_demo", settings)
    assert error.value.code == "unsafe_path"


def test_test_split_rejected(settings, source_project):
    rewrite_entry(source_project, split="test")
    with pytest.raises(AppError) as error:
        preview_document("doc_demo", settings)
    assert error.value.code == "split_not_allowed"


def test_size_limit(settings):
    with pytest.raises(AppError) as error:
        preview_document("doc_demo", settings.model_copy(update={"max_document_bytes": 5}))
    assert error.value.code == "document_too_large"


def test_duplicate_manifest_id(settings, source_project):
    entry = json.dumps(source_project[3])
    source_project[2].write_text(entry + "\n" + entry, encoding="utf-8")
    with pytest.raises(AppError) as error:
        preview_document("doc_demo", settings)
    assert error.value.code == "manifest_conflict"


def test_missing_manifest(settings, source_project):
    source_project[2].unlink()
    with pytest.raises(AppError) as error:
        preview_document("doc_demo", settings)
    assert error.value.code == "manifest_unavailable"


def test_empty_text(settings, source_project):
    source_project[1].write_bytes(b"\n ")
    rewrite_entry(source_project, sha256=hashlib.sha256(b"\n ").hexdigest())
    with pytest.raises(AppError) as error:
        preview_document("doc_demo", settings)
    assert error.value.code == "empty_document"


def test_txt_uses_plain_paragraphs(settings, source_project):
    path = source_project[1].with_suffix(".txt")
    payload = "第一段\r\n继续\r\n\r\n# 普通 TXT 不推断标题\r\n".encode()
    path.write_bytes(payload)
    rewrite_entry(source_project, path=path.relative_to(source_project[0]).as_posix(), format="txt",
                  sha256=hashlib.sha256(payload).hexdigest())
    result = preview_document("doc_demo", settings)
    assert [e.kind for e in result.elements] == ["text", "text"]
    assert result.elements[0].raw_text == "第一段\r\n继续\r\n"
    assert result.elements[1].source.line_start == 4


def test_corrupt_element_source_is_detected(settings, source_project):
    result = preview_document("doc_demo", settings)
    bad = result.elements[0].model_copy(update={"raw_text": "invented"})
    with pytest.raises(AppError) as error:
        verify_sources(result.document, result.version, source_project[1].read_text(encoding="utf-8"), [bad])
    assert error.value.code == "invalid_source_mapping"


def test_pdf_page_only_does_not_invent_coordinates():
    location = SourceLocation(source_path="sample.pdf", kind="pdf", precision="page", page_number=3)
    assert location.bbox is None


@pytest.mark.parametrize("values", [
    {"kind": "text", "precision": "line", "line_start": 5, "line_end": 1},
    {"kind": "pdf", "precision": "page", "page_number": 0},
    {"kind": "pdf", "precision": "region", "page_number": 1},
    {"kind": "pdf", "precision": "region", "page_number": 1, "bbox": (0, 0, 2, 1)},
    {"kind": "image", "precision": "image", "image_ref": "a.png"},
])
def test_invalid_locations(values):
    with pytest.raises(ValidationError):
        SourceLocation(source_path="sample", **values)


def test_image_and_region_contracts():
    location = SourceLocation(source_path="a.png", kind="image", precision="image",
                              image_ref="a.png", image_width=600, image_height=400)
    assert location.bbox is None
    region = SourceLocation(source_path="a.pdf", kind="pdf", precision="region",
                            page_number=2, bbox=(0.1, 0.2, 0.8, 0.9))
    assert region.page_number == 2


@pytest.mark.real_data
def test_real_development_document_offline():
    # 集成检查要求已备好的真实开发侧文档；没有数据时失败，不以 skip 冒充验收。
    settings = Settings(_env_file=None, project_root=PROJECT_ROOT)
    result = preview_document("tidb85_b5ecd98ba8b7dd54", settings)
    assert result.source_verified and result.external_calls == 0
    assert {"heading", "text", "table", "image"}.issubset({e.kind for e in result.elements})
    assert result.document.knowledge_base == "tidb_zh_85"
