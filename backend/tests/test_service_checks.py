"""服务验证命令的离线测试：报告、预算、中断续查，不使用个人凭证。"""
import json
from pathlib import Path
import sys

import httpx2 as httpx
import pytest

from multimodal_rag import cli
from multimodal_rag.application import service_checks
from multimodal_rag.infrastructure.http_gateway import HttpGateway
from multimodal_rag.infrastructure.settings import Settings


@pytest.fixture
def setup_check(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, project_root=tmp_path, mode="mock",
                        bailian_workspace_id="ws-test", bailian_api_key="SECRET",
                        parser_api_key="PARSER-SECRET")
    def configure(handler):
        monkeypatch.setattr(service_checks, "HttpGateway", lambda config, service, budget:
            HttpGateway(config, service, budget=budget, transport=httpx.MockTransport(handler), sleep=lambda _: None))
        return settings
    return configure


def test_check_chat_report_is_sanitized(setup_check):
    settings = setup_check(lambda _: httpx.Response(200, json={
        "choices": [{"message": {"content": "PRIVATE-RESPONSE"}, "finish_reason": "stop"}]}))
    report = service_checks.check_service(settings, "chat", 1)
    saved = Path(report["report_path"]).read_text(encoding="utf-8")
    assert report["status"] == "validated" and report["http_requests"] == 1
    assert report["quality_evaluation"] is False and report["cost"] is None
    assert "SECRET" not in saved and "PRIVATE-RESPONSE" not in saved
    assert json.loads(saved)["records"][0]["outcome"] == "validated"


def test_parser_submit_records_task_before_budget_ends(setup_check):
    def handler(request):
        assert request.method == "POST"
        return httpx.Response(200, json={"code": 0, "data": {"task_id": "resume-123"}})
    report = service_checks.check_service(setup_check(handler), "parser", 1)
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert report["status"] == "pending" and saved["task_id"] == "resume-123"
    assert saved["http_requests"] == 1


def test_parser_resume_only_gets_no_resubmit(setup_check):
    def handler(request):
        assert request.method == "GET" and request.url.path.endswith("/resume-123")
        return httpx.Response(200, json={"code": 0, "data": {"task_id": "resume-123", "state": "running"}})
    report = service_checks.check_service(setup_check(handler), "parser", 1, task_id="resume-123")
    assert report["status"] == "pending" and report["state"] == "running"
    assert report["http_requests"] == 1


def test_completed_parser_without_download_is_not_validated(setup_check):
    settings = setup_check(lambda _: httpx.Response(200, json={"code": 0, "data": {
        "task_id": "resume-123", "state": "done",
        "full_zip_url": "https://cdn-mineru.openxlab.org.cn/result.zip?token=SECRET"}}))
    report = service_checks.check_service(settings, "parser", 1, task_id="resume-123")
    saved = Path(report["report_path"]).read_text(encoding="utf-8")
    assert report["status"] == "pending" and report["state"] == "done"
    assert "result.zip" not in saved and "SECRET" not in saved


def test_check_failure_preserves_safe_record(setup_check):
    settings = setup_check(lambda _: httpx.Response(401, text="SECRET-PROVIDER-BODY"))
    report = service_checks.check_service(settings, "chat", 1)
    assert report["status"] == "failed" and report["error_code"] == "authentication_failed"
    assert report["http_requests"] == 1
    assert "SECRET" not in Path(report["report_path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("args", [
    ["db-migrate"], ["check-env", "--real"],
    ["check-service", "--service", "chat", "--max-requests", "1"],
    ["check-service", "--service", "chat", "--max-requests", "1", "--confirm-live", "--task-id", "wrong"],
    ["check-service", "--service", "chat", "--max-requests", "1", "--confirm-live"],
])
def test_cli_requires_explicit_gates(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mrag", *args])
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(_env_file=None))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("url", [None, {}, "https://[bad", "https://cdn-mineru.openxlab.org.cn:bad/a"])
def test_malformed_transfer_url_is_safe_error(url):
    from multimodal_rag.core.errors import AppError
    settings = Settings(_env_file=None, mode="mock", parser_api_key="secret")
    from multimodal_rag.infrastructure.http_gateway import CallBudget
    gateway = HttpGateway(settings, "parser", budget=CallBudget(1), transport=httpx.MockTransport(lambda _: None))
    try:
        with pytest.raises(AppError) as error:
            gateway._transfer_url(url)
        assert error.value.code == "unsafe_transfer_url" and gateway.budget.used == 0
    finally:
        gateway.close()
