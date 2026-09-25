"""受控真实验证入口：固定公开样例，预算按每次HTTP尝试计数。

不在普通健康接口暴露付费能力。报告不保存密钥、原始请求体或签名URL。
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.model_adapters import ChatAdapter, VisionAdapter, EmbeddingAdapter, RerankAdapter
from multimodal_rag.infrastructure.parser_adapter import ParserAdapter
from multimodal_rag.infrastructure.settings import Settings


def vision_sample(settings: Settings) -> tuple[bytes, str, str]:
    preview = preview_document("tidb85_b5ecd98ba8b7dd54", settings)
    root = (settings.project_root / "data/raw/tidb_zh/source").resolve()
    for element in preview.elements:
        reference = element.image_ref
        if not reference or ":" in reference or "\\" in reference:
            continue
        if reference.startswith("/"):
            path = root / reference.lstrip("/")
        else:
            path = (settings.project_root / preview.document.source_path).parent / reference
        path = path.resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if path.is_file() and path.stat().st_size <= settings.api_max_image_bytes:
            content = path.read_bytes()
            return content, "image/png" if path.suffix.lower() == ".png" else "image/jpeg", path.relative_to(settings.project_root).as_posix()
    raise AppError("vision_sample_missing", "未找到符合本批大小限制的开发侧图像", 404)


def check_service(settings: Settings, service: str, max_requests: int, *, task_id: str | None = None) -> dict:
    # 创建Gateway会先检查真实模式/开关/密钥/预算；未通过时不会发起请求。
    gateway = HttpGateway(settings, service, budget=CallBudget(max_requests))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = settings.project_root / "evals/results/service-checks" / f"{run_id}-{service}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"service": service, "run_id": run_id, "status": "started", "mode": "api",
              "max_http_requests": max_requests, "quality_evaluation": False, "cost": None,
              "records": gateway.records}

    def save():
        report["http_requests"] = gateway.budget.used
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        if service == "chat":
            result = ChatAdapter(gateway).complete("这是接口连通测试，请仅回复：连接成功。")
            report.update(model=result["model"], response_characters=len(result["text"]), usage=result["usage"])
        elif service == "vision":
            content, mime, source = vision_sample(settings)
            result = VisionAdapter(gateway).describe("请用一句话描述图中主要内容；看不清时说明不确定。", content, mime)
            report.update(model=result["model"], response_characters=len(result["text"]), usage=result["usage"],
                          sample=source, image_sha256=hashlib.sha256(content).hexdigest(), image_bytes=len(content))
        elif service == "embedding":
            result = EmbeddingAdapter(gateway).embed(["数据库连接超时，应检查网络配置。", "技术文档包含操作步骤。"])
            report.update(model=result["model"], dimensions=result["dimensions"], vectors=len(result["vectors"]), usage=result["usage"])
        elif service == "rerank":
            result = RerankAdapter(gateway).rerank("数据库连接超时", ["连接超时可以检查网络和端口。", "更换文档标题的操作说明。"], 2)
            report.update(result)
        elif service == "parser":
            adapter = ParserAdapter(gateway)
            if task_id is None:
                receipt = adapter.submit_url("https://cdn-mineru.openxlab.org.cn/demo/example.pdf")
                task_id = receipt["task_id"]
            report.update(task_id=task_id, state="submitted_or_resuming", status="pending")
            save()  # 先保存任务编号，再查询；被中断后用该编号继续，不重新提交。
            if gateway.budget.used < max_requests:
                state = adapter.poll(task_id)
                report["state"] = state["state"]
                report["progress"] = state.get("progress")
                if state["state"] == "failed":
                    raise AppError("parser_task_failed", "解析任务失败，保留任务编号供排查", 502)
                if state["state"] == "done" and gateway.budget.used < max_requests:
                    artifact = adapter.fetch_result(state["signed_result_url"])
                    report["artifact"] = artifact
                    report["status"] = "validated" if artifact["status"] == "structure_checked" else "incomplete"
                # pending/running/done但尚未下载都不算已验证，之后显式恢复查询。
        if service != "parser":
            report["status"] = "validated"
    except AppError as exc:
        report.update(status="failed", error_code=exc.code, message=exc.message)
    finally:
        save()
        gateway.close()
    return {**report, "report_path": str(output)}
