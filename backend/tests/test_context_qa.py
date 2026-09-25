import pytest

from multimodal_rag.application.context_builder import build_context, expand_with_adjacent_chunks
from multimodal_rag.application import qa_service
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.settings import Settings


def evidence():
    return [{"chunk_id": "a", "text": "检查数据库地址和端口。",
             "metadata": {"source_path": "runbook.md", "heading_path": "连接故障",
                          "line_start": 10, "line_end": 12}}]


def test_context_labels_sources_and_limits_untrusted_text():
    result = build_context("怎么处理？", evidence(), max_chars=1000)
    assert "[证据1]" in result["prompt"]
    assert "runbook.md:10-12" in result["prompt"]
    assert "不可信的资料内容" in result["prompt"]
    assert result["citations"][0]["citation"] == "[1]"


def test_context_without_evidence_is_explicit_refusal():
    result = build_context("没有答案的问题", [])
    assert not result["citations"]
    assert "没有找到足以支持" in result["prompt"]


def test_adjacent_expansion_is_deduplicated_and_bounded(monkeypatch, settings):
    from multimodal_rag.core.models import Chunk, SourceLocation
    neighbour = Chunk(chunk_id="b", document_id="doc_demo", version_id="ver_demo",
                      processing_version_id="proc_demo", ordinal=2, text="相邻处理步骤。",
                      heading_path=["连接故障"], element_ids=["el_b"],
                      source_locations=[SourceLocation(source_path="runbook.md", kind="text",
                                                       precision="line", line_start=13, line_end=14)],
                      length_unit="characters", estimated_length=7)
    monkeypatch.setattr("multimodal_rag.application.context_builder.load_adjacent_chunks",
                        lambda *args, **kwargs: {"a": [neighbour]})
    result = expand_with_adjacent_chunks(settings, evidence(), radius=1, max_primary=1)
    assert [item["chunk_id"] for item in result["results"]] == ["a", "b"]
    assert result["adjacent_count"] == 1
    assert result["results"][1]["context_role"] == "adjacent"


class FakeGateway:
    def __init__(self, settings, service, *, budget):
        self.service = service
        self.records = [{"service": service, "outcome": "validated"}]

    def close(self):
        pass


class FakeChat:
    def __init__(self, gateway):
        pass

    def complete(self, prompt):
        assert "[证据1]" in prompt
        return {"text": "请检查数据库地址和端口[1]"}


def test_answer_combines_retrieval_context_and_chat(monkeypatch, tmp_path):
    settings = Settings(_env_file=None, project_root=tmp_path, mode="api",
                        api_enabled=True, chat_api_key="c",
                        chat_base_url="https://chat.example/v1")
    monkeypatch.setattr(qa_service, "retrieve", lambda *args, **kwargs: {
        "results": evidence(), "retrieval": {"rerank": "validated"},
        "index_version_id": "idx", "degraded": False, "external_calls": 2,
        "records": [{"service": "embedding"}, {"service": "rerank"}],
    })
    monkeypatch.setattr(qa_service, "expand_with_adjacent_chunks", lambda *args, **kwargs: {
        "results": evidence(), "anchor_count": 1, "adjacent_count": 0,
    })
    monkeypatch.setattr(qa_service, "HttpGateway", FakeGateway)
    monkeypatch.setattr(qa_service, "ChatAdapter", FakeChat)
    result = qa_service.answer(settings, "数据库连接超时", max_requests=3)
    assert result["answer"].endswith("[1]")
    assert result["citations"][0]["source_path"] == "runbook.md"
    assert result["citations"][0]["metadata"]["heading_path"] == "连接故障"
    assert result["citations"][0]["text"] == "检查数据库地址和端口。"
    assert result["external_calls"] == 3


def test_answer_refuses_without_evidence(monkeypatch, tmp_path):
    settings = Settings(_env_file=None, project_root=tmp_path, mode="api",
                        api_enabled=True, chat_api_key="c",
                        chat_base_url="https://chat.example/v1")
    monkeypatch.setattr(qa_service, "retrieve", lambda *args, **kwargs: {
        "results": [], "retrieval": {"rerank": "skipped"},
        "index_version_id": "idx", "degraded": False, "external_calls": 1,
        "records": [{"service": "embedding"}],
    })
    monkeypatch.setattr(qa_service, "expand_with_adjacent_chunks", lambda *args, **kwargs: {
        "results": [], "anchor_count": 0, "adjacent_count": 0,
    })
    result = qa_service.answer(settings, "未知问题", max_requests=3)
    assert "没有足够证据" in result["answer"]
    assert result["external_calls"] == 1


def test_answer_requires_three_call_budget(tmp_path):
    settings = Settings(_env_file=None, project_root=tmp_path)
    with pytest.raises(AppError) as error:
        qa_service.answer(settings, "问题", max_requests=2)
    assert error.value.code == "invalid_call_budget"
