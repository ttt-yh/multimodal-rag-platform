"""真实API契约的人工构造夹具；通过不表示已验证供应商账号或模型能力。"""
import base64
from io import BytesIO
import json
from zipfile import ZipFile

import httpx2 as httpx
import pytest

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.model_adapters import ChatAdapter, VisionAdapter, EmbeddingAdapter, RerankAdapter
from multimodal_rag.infrastructure.parser_adapter import ParserAdapter, inspect_result_archive
from multimodal_rag.infrastructure.settings import Settings
from multimodal_rag.infrastructure.database import checked_dsn


@pytest.fixture
def gateway_factory():
    opened = []
    def factory(service, handler, limit=10, **overrides):
        values = {"mode": "mock", "bailian_api_key": "SECRET-KEY", "parser_api_key": "SECRET-PARSER",
                  "bailian_workspace_id": "ws-test", "embedding_dimensions": 64}
        values.update(overrides)
        gw = HttpGateway(Settings(_env_file=None, **values), service, budget=CallBudget(limit),
                         transport=httpx.MockTransport(handler), sleep=lambda _: None)
        opened.append(gw)
        return gw
    yield factory
    for gw in opened:
        gw.close()


def reply(data):
    return lambda _: httpx.Response(200, json=data)


def test_shared_key_and_distinct_paths():
    settings = Settings(_env_file=None, bailian_api_key="shared", chat_api_key="override", bailian_workspace_id="ws-test")
    assert settings.service_key("chat").get_secret_value() == "override"
    assert settings.service_key("vision").get_secret_value() == "shared"
    assert settings.service_key("parser").get_secret_value() == ""
    assert settings.service_base_url("embedding").endswith("compatible-mode/v1")
    assert settings.service_base_url("rerank").endswith("compatible-api/v1")


@pytest.mark.parametrize("mode,enabled", [("offline", False), ("mock", True), ("api", False)])
def test_real_call_gate(mode, enabled):
    with pytest.raises(AppError) as error:
        HttpGateway(Settings(_env_file=None, mode=mode, api_enabled=enabled), "chat", budget=CallBudget(1))
    assert error.value.code == "live_api_disabled"


def test_chat_contract(gateway_factory):
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path.endswith("/chat/completions")
        assert body["enable_thinking"] is False and body["max_tokens"] == 256
        assert request.headers["authorization"] == "Bearer SECRET-KEY"
        return httpx.Response(200, json={"choices": [{"message": {"content": "连接测试成功"}, "finish_reason": "stop"}],
                                       "usage": {"total_tokens": 18}})
    gw = gateway_factory("chat", handler)
    result = ChatAdapter(gw).complete("测试")
    assert result["usage"]["total_tokens"] == 18
    assert gw.records[-1]["outcome"] == "validated"
    assert "SECRET" not in json.dumps(gw.records)


@pytest.mark.parametrize("data", [{}, {"choices": []},
    {"choices": [{"message": {"content": "截断"}, "finish_reason": "length"}]},
    {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}])
def test_chat_invalid_responses(gateway_factory, data):
    gw = gateway_factory("chat", reply(data))
    with pytest.raises(AppError) as error:
        ChatAdapter(gw).complete("测试")
    assert error.value.code == "invalid_response_contract"


def test_vision_sends_actual_image_bytes(gateway_factory):
    image = b"\x89PNG\r\n\x1a\nmock-bytes"
    def handler(request):
        body = json.loads(request.content)
        uri = body["messages"][0]["content"][1]["image_url"]["url"]
        assert base64.b64decode(uri.split(",", 1)[1]) == image
        return httpx.Response(200, json={"choices": [{"message": {"content": "描述"}, "finish_reason": "stop"}]})
    gw = gateway_factory("vision", handler)
    assert VisionAdapter(gw).describe("图中是什么", image, "image/png")["usage"] is None


def test_vision_supports_bounded_multiple_images(gateway_factory):
    images = [(b"\x89PNG\r\n\x1a\none", "image/png"), (b"\xff\xd8\xfftwo", "image/jpeg")]
    def handler(request):
        content = json.loads(request.content)["messages"][0]["content"]
        assert len(content) == 3
        assert [item["type"] for item in content] == ["text", "image_url", "image_url"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "联合描述"},
                                                       "finish_reason": "stop"}]})
    result = VisionAdapter(gateway_factory("vision", handler)).describe_many("比较图片", images)
    assert result["text"] == "联合描述"


def test_vision_invalid_input_no_request(gateway_factory):
    gw = gateway_factory("vision", reply({}))
    with pytest.raises(AppError):
        VisionAdapter(gw).describe("图", b"text", "image/png")
    assert gw.budget.used == 0


def test_embedding_maps_original_indices(gateway_factory):
    gw = gateway_factory("embedding", reply({"data": [
        {"index": 1, "embedding": [2.0]*64}, {"index": 0, "embedding": [1.0]*64}]}))
    assert EmbeddingAdapter(gw).embed(["甲", "乙"])["vectors"][0] == [1.0]*64


@pytest.mark.parametrize("vector,index", [([1.0]*63, 0), ([0.0]*64, 0), ([float("inf")]*64, 0), ([1.0]*64, 4)])
def test_embedding_rejects_invalid_vectors(gateway_factory, vector, index):
    # 手工JSON字节允许构造非有限数，验证适配器不会将其写入索引。
    gw = gateway_factory("embedding", lambda _: httpx.Response(200, content=json.dumps({"data": [{"index": index, "embedding": vector}]}).encode()))
    with pytest.raises(AppError):
        EmbeddingAdapter(gw).embed(["测试"])


def test_embedding_empty_input_no_cost(gateway_factory):
    gw = gateway_factory("embedding", reply({}))
    with pytest.raises(AppError):
        EmbeddingAdapter(gw).embed([" "])
    assert gw.budget.used == 0


def test_rerank_endpoint_and_mapping(gateway_factory):
    def handler(request):
        assert request.url.path == "/compatible-api/v1/reranks"
        assert "parameters" not in json.loads(request.content)
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": .9}, {"index": 0, "relevance_score": .1}]})
    result = RerankAdapter(gateway_factory("rerank", handler)).rerank("连接", ["无关", "数据库连接"], 2)
    assert result["results"][0]["index"] == 1


@pytest.mark.parametrize("rows", [[], [{"index": 2, "relevance_score": .8}],
    [{"index": 0, "relevance_score": 1.5}], [{"index": True, "relevance_score": .5}]])
def test_rerank_invalid_results(gateway_factory, rows):
    with pytest.raises(AppError):
        RerankAdapter(gateway_factory("rerank", reply({"results": rows}))).rerank("q", ["a"], 1)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_post_errors_never_retry_or_leak(gateway_factory, status):
    gw = gateway_factory("chat", lambda _: httpx.Response(status, text="password=SECRET provider-detail"))
    with pytest.raises(AppError) as error:
        gw.request("POST", "/chat/completions", payload={})
    assert gw.budget.used == 1
    assert "SECRET" not in str(error.value) + json.dumps(gw.records)


def test_post_timeout_is_not_retried(gateway_factory):
    def handler(request):
        raise httpx.ReadTimeout("url?token=SECRET", request=request)
    gw = gateway_factory("chat", handler)
    with pytest.raises(AppError) as error:
        gw.request("POST", "/chat/completions", payload={})
    assert gw.budget.used == 1 and "SECRET" not in str(error.value)


def test_get_has_bounded_retry_and_budget(gateway_factory):
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(503 if len(calls) < 3 else 200, json={"code": 0})
    gw = gateway_factory("parser", handler)
    assert gw.request("GET", "/extract/task/task-1")["code"] == 0
    assert len(calls) == 3
    limited = gateway_factory("parser", lambda _: httpx.Response(503), limit=1)
    with pytest.raises(AppError) as error:
        limited.request("GET", "/extract/task/task-1")
    assert error.value.code == "call_budget_exhausted" and limited.budget.used == 1


@pytest.mark.parametrize("response,code", [
    (httpx.Response(200, text="not-json"), "invalid_provider_json"),
    (httpx.Response(200, json={"code": 1001, "msg": "secret"}), "provider_business_error"),
    (httpx.Response(302, headers={"Location": "https://evil.invalid"}), "provider_http_error")])
def test_bad_http_success_and_redirect(gateway_factory, response, code):
    gw = gateway_factory("chat", lambda _: response)
    with pytest.raises(AppError) as error:
        gw.request("POST", "/chat/completions", payload={})
    assert error.value.code == code


def test_response_size_limit(gateway_factory):
    gw = gateway_factory("chat", lambda _: httpx.Response(200, content=b"x"*1025), api_max_response_bytes=1024)
    with pytest.raises(AppError) as error:
        gw.request("POST", "/chat/completions")
    assert error.value.code == "response_too_large"


def test_parser_submit_and_poll(gateway_factory):
    def handler(request):
        if request.method == "POST":
            assert json.loads(request.content)["page_ranges"] == "1-2"
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "task-1"}})
        return httpx.Response(200, json={"code": 0, "data": {"task_id": "task-1", "state": "done",
            "full_zip_url": "https://cdn-mineru.openxlab.org.cn/a.zip?token=SECRET"}})
    gw = gateway_factory("parser", handler)
    adapter = ParserAdapter(gw)
    assert adapter.submit_url("https://cdn-mineru.openxlab.org.cn/demo/example.pdf")["task_id"] == "task-1"
    assert adapter.poll("task-1")["state"] == "done"
    assert "SECRET" not in json.dumps(gw.records)


def test_parser_signed_upload_no_bearer(gateway_factory):
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"code": 0, "data": {"batch_id": "batch-1", "file_urls": [
                "https://mineru.oss-cn-shanghai.aliyuncs.com/a.pdf?signature=SECRET"]}})
        assert "authorization" not in request.headers
        return httpx.Response(200, content=b"")
    gw = gateway_factory("parser", handler)
    adapter = ParserAdapter(gw)
    ticket = adapter.request_upload("sample.pdf", "sample-1")
    assert adapter.upload(ticket["signed_url"], b"%PDF-mock")["uploaded"]
    assert "SECRET" not in json.dumps(gw.records)


@pytest.mark.parametrize("url", ["https://127.0.0.1/a", "http://cdn-mineru.openxlab.org.cn/a",
    "https://cdn-mineru.openxlab.org.cn.evil.invalid/a", "https://u:p@cdn-mineru.openxlab.org.cn/a"])
def test_transfer_allowlist(gateway_factory, url):
    gw = gateway_factory("parser", reply({}))
    with pytest.raises(AppError):
        gw.request("GET", "", transfer_url=url)
    assert gw.budget.used == 0


def archive(extra=None):
    buffer = BytesIO()
    files = {"full.md": "# 测试", "result_content_list.json": json.dumps([
        {"type": "text", "text": "测试", "page_idx": 0},
        {"type": "image", "img_path": "images/a.png", "page_idx": 0}]), "images/a.png": "mock-image"}
    files.update(extra or {})
    with ZipFile(buffer, "w") as z:
        for name, body in files.items():
            z.writestr(name, body)
    return buffer.getvalue()


def test_parser_archive_contract():
    result = inspect_result_archive(archive(), 10000)
    assert result["structured_elements"] == 2 and result["missing_image_references"] == 0
    assert result["provider_page_indices"] == [0] and not result["source_mapping_normalized"]


@pytest.mark.parametrize("extra", [{"../escape.txt": "x"}, {"result_content_list.json": "[]"},
    {"result_content_list.json": '[{"page_idx":-1}]'}, {"full.md": ""}])
def test_parser_archive_rejects_unsafe_or_incomplete(extra):
    with pytest.raises(AppError):
        inspect_result_archive(archive(extra), 10000)


@pytest.mark.parametrize("dsn", ["dbname=rag_diagnosis user=rag_agent host=127.0.0.1",
    "dbname=multimodal_rag user=postgres host=127.0.0.1", "dbname=multimodal_rag user=multimodal_rag_app host=remote",
    "password=secret", "bad-dsn-secret"])
def test_database_target_guard(dsn):
    with pytest.raises(AppError) as error:
        checked_dsn(Settings(_env_file=None, postgres_dsn=dsn))
    assert error.value.code == "unsafe_database_target" and "secret" not in str(error.value)
