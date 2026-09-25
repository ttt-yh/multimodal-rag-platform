"""阶段 6B Web 竣工验收：不提交问答、不调用外部模型。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "evals" / "results" / "stage6b"


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    page_errors: list[str] = []
    http_errors: list[str] = []
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    with sync_playwright() as playwright:
        # 用户机器可能启用系统代理；本地验收明确绕过代理，避免 localhost 被代理拦截。
        browser = playwright.chromium.launch(
            channel="msedge", headless=True, args=["--no-proxy-server"])
        try:
            context = browser.new_context(
                viewport={"width": 1440, "height": 1000},
                permissions=["clipboard-read", "clipboard-write"])
            page = context.new_page()
            page.on("console", lambda message: console_errors.append(message.text)
                    if message.type == "error" else None)
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on("response", lambda result: http_errors.append(
                f"{result.status} {result.request.method} {result.url}")
                if result.status >= 400 else None)

            response = page.goto("http://127.0.0.1:5173", wait_until="domcontentloaded")
            record("前端首页可访问", response is not None and response.ok,
                   str(response.status if response else "no response"))
            page.get_by_role("heading", name="从内部知识中找到").wait_for()
            page.get_by_text("API 已连接", exact=True).wait_for()
            record("后端健康检查已连接", True)

            composer = page.get_by_placeholder("例如：数据库连接超时时应该检查哪些配置？")
            composer.fill("仅检查输入框，不发送模型请求")
            record("问答输入交互可用", composer.input_value() == "仅检查输入框，不发送模型请求")
            page.get_by_role("button", name="复制回答", exact=True).click()
            page.get_by_role("button", name="已复制", exact=True).wait_for()
            record("复制回答操作可用", True)
            record("未实现入口已明确禁用",
                   page.get_by_title("当前版本不包含多租户工作区设置").is_disabled()
                   and page.get_by_title("使用说明请查看仓库 README 与部署文档").is_disabled())

            page.get_by_role("button", name="添加资料", exact=True).click()
            page.get_by_role("heading", name="导入研发资料", exact=True).wait_for()
            record("问答页添加资料入口可达", True)
            page.get_by_title("关闭导入窗口").click()
            page.get_by_role("heading", name="知识库", exact=True).wait_for()
            page.get_by_text("PostgreSQL 实时状态", exact=True).wait_for()
            record("知识库实时状态已加载", True)

            page.get_by_role("button", name="评测记录", exact=True).click()
            page.get_by_role("heading", name="评测记录", exact=True).wait_for()
            page.get_by_text("严格确定性门禁", exact=True).wait_for()
            page.get_by_text("开发级语义评审", exact=True).wait_for()
            page.get_by_text("必要要点覆盖", exact=True).last.wait_for()
            page.get_by_text("0.70", exact=True).wait_for()
            page.get_by_text("0.908", exact=True).wait_for()
            record("Visual Gold v2.4 最终指标已展示", True,
                   "strict=0.70, semantic=1.00, required_terms=0.908")

            page.get_by_role("button", name="刷新评测", exact=True).click()
            page.get_by_role("button", name="刷新评测", exact=True).wait_for()
            record("评测刷新操作可用", True)

            page.screenshot(path=str(RESULT_ROOT / "web_acceptance.png"), full_page=True)
            record("验收截图已生成", True, "evals/results/stage6b/web_acceptance.png")
        finally:
            context.close()
            browser.close()

    checks.append({"name": "无页面运行时异常", "passed": not page_errors,
                   "detail": " | ".join(page_errors)})
    checks.append({"name": "无浏览器控制台错误", "passed": not console_errors,
                   "detail": " | ".join(console_errors)})
    checks.append({"name": "无HTTP错误响应", "passed": not http_errors,
                   "detail": " | ".join(http_errors)})
    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "scope": "no_model_call_web_acceptance",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "console_errors": console_errors,
        "page_errors": page_errors,
        "http_errors": http_errors,
        "external_model_calls": 0,
    }
    (RESULT_ROOT / "web_acceptance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
