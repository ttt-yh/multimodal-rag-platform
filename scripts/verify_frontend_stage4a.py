"""用 Playwright 验收阶段 4A 的 Vue 问答工作台。

脚本由 with_server.py 负责启动前后端，本文件只描述浏览器中的用户路径：
打开问答页、读取知识库和评测记录、提交一个真实问题并核对回答与引用。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "evals" / "results" / "stage4-web"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="msedge", headless=True, args=["--disable-gpu"]
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        try:
            # Vite 在 Windows 上可能只监听 IPv6 loopback，因此使用 localhost
            # 比硬编码 127.0.0.1 更稳健；页面内部 API 仍由 Vite 代理到后端。
            page.goto("http://localhost:5173", wait_until="domcontentloaded")
            expect(page.get_by_role("heading", name="从内部知识中找到").first).to_be_visible()
            report["checks"]["app_loaded"] = True

            # 知识库页确认前端能够读取 PostgreSQL 中的真实索引状态。
            page.get_by_role("button", name="知识库", exact=True).click()
            expect(page.get_by_role("heading", name="知识库").first).to_be_visible()
            page.get_by_role("button", name="刷新").first.click()
            page.wait_for_timeout(500)
            knowledge_text = page.locator("main").inner_text()
            report["checks"]["knowledge_page"] = {
                "passed": "索引状态" in knowledge_text and "PostgreSQL" in knowledge_text,
                "contains_active_index": "active" in knowledge_text,
            }

            # 新增的白名单入库流程：选择文档、创建任务、运行单 worker 并展示最终状态。
            page.get_by_role("button", name="导入资料", exact=True).click()
            expect(page.get_by_role("dialog", name="导入研发资料")).to_be_visible()
            document_select = page.locator("#ingestion-document")
            assert document_select.locator("option").count() > 1, "白名单文档清单为空"
            page.get_by_role("button", name="开始入库", exact=True).click()
            expect(page.get_by_text("最近入库任务").first).to_be_visible(timeout=30000)
            ingestion_text = page.locator("main").inner_text()
            report["checks"]["ingestion_submit"] = {
                "passed": "最近入库任务" in ingestion_text and ("已完成" in ingestion_text or "处理中" in ingestion_text),
                "has_stage": "quality_checked" in ingestion_text or "queued" in ingestion_text,
            }
            page.get_by_role("button", name="关闭", exact=True).first.click()
            release_text = page.locator("main").inner_text()
            report["checks"]["release_panels"] = {
                "passed": "审核队列" in release_text and "索引版本" in release_text,
                "has_build_gate": "我确认调用模型" in release_text,
            }

            # 评测页确认真实联调报告能够列出，并可以打开详情弹窗。
            page.get_by_role("button", name="评测记录", exact=True).click()
            expect(page.get_by_role("heading", name="评测记录").first).to_be_visible()
            page.get_by_role("button", name="刷新").first.click()
            page.wait_for_timeout(500)
            evaluation_text = page.locator("main").inner_text()
            has_report = "运行报告" in evaluation_text and "已验证" in evaluation_text
            report["checks"]["evaluation_page"] = {"passed": has_report}
            detail_buttons = page.get_by_role("button", name="详情")
            if detail_buttons.count():
                detail_buttons.first.click()
                expect(page.get_by_role("dialog", name="运行报告详情")).to_be_visible()
                report["checks"]["evaluation_detail"] = {"passed": True}
                page.get_by_role("button", name="关闭详情").click()
            else:
                report["checks"]["evaluation_detail"] = {
                    "passed": False,
                    "reason": "没有可打开的运行报告",
                }

            # 回到问答页，提交一个资料中已有的中文问题，核对结果与引用。
            page.get_by_role("button", name="问答工作台", exact=True).click()
            # 选择阶段4A离线问答中已通过的题型，避免把“知识库未覆盖”
            # 与前端链路故障混为一谈。
            query = "根据《TiDB Dashboard 监控页面》，关于“Write Traffic”有哪些主要说明或操作要求？"
            page.locator("textarea").fill(query)
            page.get_by_role("button", name="开始检索").click()
            expect(page.get_by_text("本次回答").first).to_be_visible(timeout=120000)
            page.wait_for_timeout(300)
            answer_text = page.locator("main").inner_text()
            has_answer = "引用证据" in answer_text and "本次回答" in answer_text
            report["checks"]["qa_submit"] = {
                "passed": has_answer,
                "query": query,
                "has_citations": "引用证据" in answer_text,
                "has_retrieval_trace": "混合检索" in answer_text,
            }
            page.screenshot(path=str(RESULT_DIR / "qa-workbench.png"), full_page=True)
        except Exception as exc:  # pragma: no cover - 失败时把现场留给调用者
            report["error"] = f"{type(exc).__name__}: {exc}"
            page.screenshot(path=str(RESULT_DIR / "failure.png"), full_page=True)
            raise
        finally:
            browser.close()

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    target = RESULT_DIR / "frontend_stage4a_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
