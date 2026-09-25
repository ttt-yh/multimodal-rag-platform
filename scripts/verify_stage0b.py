"""0B验收：离线契约测试、原数据回归、实际本机HTTP与隔离PostgreSQL。

不会读取.env或连接用户业务库，不调用外部模型；临时数据库检查后停止。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from verify_stage0a import ROOT, run_tests, live_smoke, write_json
from verify_database_fixture import run_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg-bin", type=Path, required=True, help="已安装PostgreSQL工具目录，仅用于临时实例")
    args = parser.parse_args()
    # 清除个人应用配置；数据库fixture使用显式随机凭证、回环随机端口。
    for name in list(os.environ):
        if name.startswith("MRAG_"):
            os.environ.pop(name)
    base = ROOT / "evals/results/stage0b"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = base / run_id
    output.mkdir(parents=True)
    config = {"phase": "0B", "mode": "offline", "external_calls_allowed": False,
              "demo_document_id": "tidb85_b5ecd98ba8b7dd54", "database": "isolated_temporary_cluster"}
    app = run_tests(Path(sys.executable), "backend/tests", "app-tests", output, isolated_temp=True)
    data_python = ROOT / (".venv-data/Scripts/python.exe" if os.name == "nt" else ".venv-data/bin/python")
    data = run_tests(data_python, "tests", "data-tests", output, isolated_temp=True)
    smoke = live_smoke(output, config["demo_document_id"])
    database = run_fixture(args.pg_bin, output)
    tracked = sorted(set((ROOT / "backend").rglob("*.py")) |
                     set((ROOT / "backend").rglob("*.sql")) |
                     set((ROOT / "scripts").glob("verify_*.py")) |
                     {ROOT / p for p in ["pyproject.toml", "uv.lock", ".env.example", "scripts/bootstrap_database.sql"]})
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    passed = all(item["status"] == "passed" for item in [app, data, smoke, database])
    report = {"phase": "0B", "run_id": run_id,
              "status": "offline_passed_live_pending" if passed else "failed",
              "app_tests": app, "data_regression": data, "http_smoke": smoke, "database_fixture": database,
              "config": config, "source_files": hashes, "source_sha256": fingerprint,
              "python": sys.version,
              "versions": {name: importlib.metadata.version(name) for name in
                           ["fastapi", "pydantic", "httpx2", "psycopg", "psycopg-binary", "pytest"]},
              "data_bundle": json.loads((ROOT / "data/manifests/bundle.json").read_text(encoding="utf-8"))["bundle_id"],
              "external_api_calls": 0, "actual_project_database_checked": False, "model_evaluation_run": False,
              "limitations": ["接口响应为人工构造夹具，不代表真实账号、供应商兼容性或模型效果已验证。",
                              "PostgreSQL为真实16版隔离实例，不是用户5432端口上的项目库。",
                              "当前仅具备迁移和外部服务适配，尚未接通批量入库、Chroma、BM25、检索问答。",
                              "中文180条候选仍待人工审核；页码/坐标映射与图文质量在0C逐样例核验。"]}
    write_json(output / "summary.json", report)
    write_json(base / "summary.json", report)
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{item['status']}</td><td>{count}</td></tr>"
                   for name, item, count in [
                       ("应用与外部服务离线契约", app, f"{app['passed']}/{app['total']}"),
                       ("原数据回归", data, f"{data['passed']}/{data['total']}"),
                       ("真实本机HTTP", smoke, str(sum(smoke.get('checks', {}).values()))),
                       ("隔离PostgreSQL", database, str(database['checks_passed']))])
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>0B工程验收</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1020px;margin:40px auto;padding:0 24px;color:#243342}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:10px;text-align:left}}code{{overflow-wrap:anywhere}}a{{color:#126781}}</style>
<h1>0B工程验收：{report['status']}</h1><p>运行：{run_id} · 数据：{html.escape(report['data_bundle'])}</p>
<p>外部API调用0次；未验证用户业务库、真实API和模型质量。临时数据库已停止：{database.get('server_stopped', False)}。</p>
<table><tr><th>检查</th><th>状态</th><th>通过数量</th></tr>{rows}</table>
<p><a href="{run_id}/summary.json">完整快照与源码指纹</a> · <a href="{run_id}/app-tests.xml">应用JUnit</a> · <a href="{run_id}/app-tests.log">应用日志</a> · <a href="{run_id}/data-tests.xml">数据JUnit</a> · <a href="{run_id}/database.json">数据库逐项结果</a> · <a href="{run_id}/preview.json">84元素预览</a></p>
<h2>当前边界</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in report['limitations'])}</ul>
<p>源码SHA256：<code>{fingerprint}</code></p></html>"""
    (base / "report.html").write_text(page, encoding="utf-8")
    # 历史运行入口使用同一页面中的相对资源，不指向后续覆盖的最新快照。
    (output / "report.html").write_text(page.replace(f'href="{run_id}/', 'href="'), encoding="utf-8")
    print(json.dumps({"status": report["status"], "application_tests": app["passed"],
                      "data_tests": data["passed"], "http_smoke": smoke["status"],
                      "database_checks": database["checks_passed"], "database_status": database["status"],
                      "report": str(base / "report.html")}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
