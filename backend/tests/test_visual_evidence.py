from pathlib import Path

from multimodal_rag.application import visual_evidence
from multimodal_rag.infrastructure.settings import Settings


def _row(image_ref: str):
    return {"chunk_id": "chk_a", "element_id": "el_a", "document_id": "doc_a",
            "ordinal": 2, "image_ref": image_ref, "raw_text": "![图](...)\n",
            "heading_path": ["章节"], "source": {"kind": "text", "precision": "line",
            "source_path": "data/raw/demo/source/docs/a.md", "line_start": 2, "line_end": 2},
            "document_title": "示例", "document_source_path": "data/raw/demo/source/docs/a.md"}


def test_locate_visual_evidence_resolves_root_relative_asset(monkeypatch, tmp_path: Path):
    image = tmp_path / "data/raw/demo/source/media/chart.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    monkeypatch.setattr(visual_evidence, "load_image_elements_for_chunks",
                        lambda *args, **kwargs: [_row("/media/chart.png")])
    result = visual_evidence.locate_visual_evidence(
        Settings(_env_file=None, project_root=tmp_path, api_max_image_bytes=100), ["chk_a"])
    assert result["ready_count"] == 1
    assert result["candidates"][0]["asset_path"] == "data/raw/demo/source/media/chart.png"
    assert result["candidates"][0]["mime_type"] == "image/png"


def test_locate_visual_evidence_rejects_traversal(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(visual_evidence, "load_image_elements_for_chunks",
                        lambda *args, **kwargs: [_row("../../../../secret.png")])
    result = visual_evidence.locate_visual_evidence(
        Settings(_env_file=None, project_root=tmp_path), ["chk_a"])
    assert result["ready_count"] == 0
    assert result["candidates"][0]["status"] == "unsafe_reference"


def test_locate_visual_evidence_deduplicates_same_asset(monkeypatch, tmp_path: Path):
    image = tmp_path / "data/raw/demo/source/media/chart.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
    rows = [_row("/media/chart.png"), {**_row("/media/chart.png"), "element_id": "el_b"}]
    monkeypatch.setattr(visual_evidence, "load_image_elements_for_chunks",
                        lambda *args, **kwargs: rows)
    result = visual_evidence.locate_visual_evidence(
        Settings(_env_file=None, project_root=tmp_path, api_max_image_bytes=100), ["chk_a"])
    assert result["ready_count"] == 1
    assert result["candidate_count"] == 1


def test_locate_visual_evidence_diversifies_across_chunks(monkeypatch, tmp_path: Path):
    for name in ("a.png", "b.png", "c.png"):
        image = tmp_path / "data/raw/demo/source/media" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode())
    first = _row("/media/a.png")
    rows = [first, {**first, "element_id": "el_b", "image_ref": "/media/b.png"},
            {**first, "chunk_id": "chk_b", "element_id": "el_c", "image_ref": "/media/c.png"}]
    monkeypatch.setattr(visual_evidence, "load_image_elements_for_chunks",
                        lambda *args, **kwargs: rows)
    result = visual_evidence.locate_visual_evidence(
        Settings(_env_file=None, project_root=tmp_path, api_max_image_bytes=100),
        ["chk_a", "chk_b"], max_images=2)
    assert [item["image_ref"] for item in result["candidates"]] == ["/media/a.png", "/media/c.png"]


def test_title_locator_ranks_query_specific_image(monkeypatch, tmp_path: Path):
    for name in ("access.png", "view-progress.png", "start.png"):
        image = tmp_path / "data/raw/demo/source/media" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode())
    base = _row("/media/access.png")
    rows = [
        {**base, "raw_text": "访问页面", "chunk_text": "打开实例性能分析"},
        {**base, "element_id": "el_progress", "image_ref": "/media/view-progress.png",
         "raw_text": "CPU progress Heap Goroutine", "chunk_text": "查看性能采集进度"},
        {**base, "element_id": "el_start", "image_ref": "/media/start.png",
         "raw_text": "start profiling", "chunk_text": "开始分析"},
    ]
    monkeypatch.setattr(visual_evidence, "load_active_image_elements_for_title",
                        lambda *args, **kwargs: rows)
    result = visual_evidence.locate_visual_evidence_for_title(
        Settings(_env_file=None, project_root=tmp_path, api_max_image_bytes=100),
        "示例", "依据《示例》，CPU 采集进度以及 Heap 和 Goroutine 状态是什么？",
        max_images=2)
    assert result["selection_strategy"] == "exact_title_then_local_visual_relevance"
    assert result["candidates"][0]["element_id"] == "el_progress"


def test_title_locator_prioritises_explicit_quoted_image_label(monkeypatch, tmp_path: Path):
    for name in ("plan.png", "time.png"):
        image = tmp_path / "data/raw/demo/source/media" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode())
    base = _row("/media/plan.png")
    rows = [
        {**base, "raw_text": "![执行计划](/media/plan.png)", "chunk_text": "执行计划详情"},
        {**base, "element_id": "el_time", "image_ref": "/media/time.png",
         "raw_text": "![执行时间](/media/time.png)", "chunk_text": "执行时间"},
    ]
    monkeypatch.setattr(visual_evidence, "load_active_image_elements_for_title",
                        lambda *args, **kwargs: rows)
    result = visual_evidence.locate_visual_evidence_for_title(
        Settings(_env_file=None, project_root=tmp_path, api_max_image_bytes=100),
        "示例", "依据《示例》中标注为“执行时间”的原始截图回答问题", max_images=2)
    assert [item["element_id"] for item in result["candidates"]] == ["el_time"]
