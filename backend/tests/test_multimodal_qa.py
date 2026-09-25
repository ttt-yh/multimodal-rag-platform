from multimodal_rag.application import multimodal_qa_service
from multimodal_rag.infrastructure.settings import Settings


def _settings(tmp_path):
    return Settings(_env_file=None, project_root=tmp_path, mode="api", api_enabled=True,
                    bailian_api_key="key", bailian_workspace_id="ws")


def _retrieval():
    return {"results": [{"chunk_id": "chk_a", "text": "图表位于本章节。",
                         "metadata": {"source_path": "doc.md", "heading_path": "图表",
                                      "line_start": 2, "line_end": 3}}],
            "retrieval": {"rerank": "validated"}, "index_version_id": "idx",
            "degraded": False, "external_calls": 2,
            "records": [{"service": "embedding"}, {"service": "rerank"}]}


class FakeGateway:
    def __init__(self, settings, service, *, budget):
        self.records = [{"service": service, "status_code": 200}]
    def close(self):
        pass


class FakeVision:
    def __init__(self, gateway):
        pass
    def describe_many(self, prompt, images):
        assert "[图1]" in prompt and len(images) == 1
        return {"text": "图中最大值为21.13 K[图1]"}


def test_visual_question_uses_retrieved_image_and_vlm(monkeypatch, tmp_path):
    monkeypatch.setattr(multimodal_qa_service, "retrieve", lambda *args, **kwargs: _retrieval())
    monkeypatch.setattr(multimodal_qa_service, "expand_with_adjacent_chunks", lambda *args, **kwargs: {
        "results": _retrieval()["results"], "anchor_count": 1, "adjacent_count": 0})
    monkeypatch.setattr(multimodal_qa_service, "locate_visual_evidence", lambda *args, **kwargs: {
        "candidates": [{"chunk_id": "chk_a", "vision_eligible": True,
                        "document_title": "文档", "image_ref": "/media/a.png",
                        "source": {"source_path": "doc.md", "line_start": 2}}]})
    monkeypatch.setattr(multimodal_qa_service, "read_visual_candidate",
                        lambda *args, **kwargs: (b"\x89PNG\r\n\x1a\n", "image/png"))
    monkeypatch.setattr(multimodal_qa_service, "HttpGateway", FakeGateway)
    monkeypatch.setattr(multimodal_qa_service, "VisionAdapter", FakeVision)
    result = multimodal_qa_service.answer_multimodal(_settings(tmp_path), "截图图例最大值是什么？")
    assert result["answer_mode"] == "vision"
    assert result["external_calls"] == 3
    assert result["visual_evidence"][0]["image_citation"] == "[图1]"


def test_visual_question_refuses_when_retrieved_chunk_has_no_safe_image(monkeypatch, tmp_path):
    monkeypatch.setattr(multimodal_qa_service, "retrieve", lambda *args, **kwargs: _retrieval())
    monkeypatch.setattr(multimodal_qa_service, "expand_with_adjacent_chunks", lambda *args, **kwargs: {
        "results": _retrieval()["results"], "anchor_count": 1, "adjacent_count": 0})
    monkeypatch.setattr(multimodal_qa_service, "locate_visual_evidence", lambda *args, **kwargs: {
        "candidates": [], "ready_count": 0})
    result = multimodal_qa_service.answer_multimodal(_settings(tmp_path), "图中节点是什么？")
    assert result["answer_mode"] == "visual_refusal"
    assert result["external_calls"] == 2


def test_visual_router_recognises_domain_specific_flow_and_heatmaps():
    assert multimodal_qa_service.requires_visual("请观察原始流量图说明热点方向")
    assert multimodal_qa_service.requires_visual("根据热力图判断亮色区域")


def test_visual_task_checks_cover_comparison_and_shape_questions():
    comparison = multimodal_qa_service._task_check_text("哪个延迟最大？")
    distribution = multimodal_qa_service._task_check_text("亮色区域呈什么方向分布？")
    assert "换算为同一单位" in comparison
    assert "核对方向与形态" in distribution
