from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.chunking import split_elements


def test_chunking_is_stable_and_keeps_heading_and_sources(settings):
    preview = preview_document("doc_demo", settings)
    first = split_elements(preview.elements, "proc_fixture", max_characters=160, overlap_characters=20)
    second = split_elements(preview.elements, "proc_fixture", max_characters=160, overlap_characters=20)
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert first
    assert all(item.length_unit == "characters" for item in first)
    assert all(item.element_ids and item.source_locations for item in first)
    assert any(item.heading_path == ["部署说明", "参数"] for item in first)


def test_long_element_is_recursively_split(settings):
    preview = preview_document("doc_demo", settings)
    chunks = split_elements(preview.elements, "proc_fixture", max_characters=80, overlap_characters=10)
    assert max(item.estimated_length for item in chunks) <= 80


def test_short_tail_keeps_all_element_sources(settings):
    preview = preview_document("doc_demo", settings)
    chunks = split_elements(preview.elements, "proc_fixture", max_characters=240,
                            overlap_characters=20, min_characters=100)
    # 合并章节短尾时，前一个 Chunk 与短尾元素的来源都必须保留。
    assert any(len(chunk.element_ids) >= 2 for chunk in chunks)
    assert all(len(chunk.element_ids) == len(chunk.source_locations) for chunk in chunks)
