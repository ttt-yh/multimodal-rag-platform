"""0A 可重复验收：应用测试 + 原数据回归 + 本机实际启动检查。

使用应用环境运行本脚本，数据回归仍使用 .venv-data；不会重建/修改数据清单。
每次报告独立保存，顶层 report.html/summary.json 是最新运行的入口。
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_tests(python: Path, target: str, name: str, output: Path, *, isolated_temp=False) -> dict:
    command = [str(python), "-X", "utf8", "-m", "pytest", target, "-q", "--tb=short",
               f"--junitxml={output / (name + '.xml')}"]
    if isolated_temp:
        # 不复用不同Windows执行身份创建的pytest缓存；--basetemp必须是新路径。
        temp = (output / (name + "-tmp")).resolve()
        if not temp.is_relative_to((ROOT / "evals/results").resolve()) or temp.exists():
            raise ValueError("pytest temporary directory must be fresh and under results")
        command.extend([f"--basetemp={temp}", "-p", "no:cacheprovider"])
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8",
                                errors="replace", timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "failed", "reason": "测试解释器不可用或执行超时", "total": 0, "passed": 0}
    (output / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    xml_path = output / f"{name}.xml"
    if not xml_path.exists():
        return {"status": "failed", "reason": "未生成 JUnit 文件", "total": 0, "passed": 0}
    cases = ET.parse(xml_path).findall(".//testcase")
    failures = [c.attrib["name"] for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [c.attrib["name"] for c in cases if c.find("skipped") is not None]
    return {"status": "passed" if result.returncode == 0 and cases and not skipped else "failed",
            "total": len(cases), "passed": len(cases) - len(failures) - len(skipped),
            "failures": failures, "skipped": skipped, "exit_code": result.returncode,
            "seconds": round(sum(float(c.get("time", 0)) for c in cases), 3)}


def live_smoke(output: Path, document_id: str) -> dict:
    """独立子进程启动真正的 HTTP 服务，检查完必定停止，不占用用户常用端口。"""
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    # 清除 MRAG_*；子进程明确不读取 .env，不可能用用户后续填写的凭证发起调用。
    env = {k: v for k, v in os.environ.items() if not k.startswith("MRAG_")}
    command = [sys.executable, "-X", "utf8", "-c",
               "import uvicorn; from multimodal_rag.api.app import create_app; "
               "from multimodal_rag.infrastructure.settings import Settings; "
               f"uvicorn.run(create_app(Settings(_env_file=None)),host='127.0.0.1',port={port},access_log=False)"]
    process = None
    with (output / "server.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            def request(method, path, body=None):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                try:
                    connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
                    response = connection.getresponse()
                    payload = response.read()
                    return response.status, payload, response.getheader("X-Request-ID")
                finally:
                    connection.close()
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    raise RuntimeError("server exited")
                try:
                    status, payload, request_id = request("GET", "/health/live")
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("startup timeout") from None
                    time.sleep(0.1)
            checks = {"live_200": status == 200 and json.loads(payload)["status"] == "alive",
                      "request_id": bool(request_id)}
            status, payload, _ = request("GET", "/health/ready")
            checks["ready_honest_503"] = status == 503 and json.loads(payload)["ready"] is False
            status, payload, _ = request("GET", "/openapi.json")
            checks["openapi_200"] = status == 200 and "/api/v1/preview" in json.loads(payload)["paths"]
            checks["docs_200"] = request("GET", "/docs")[0] == 200
            status, payload, _ = request("POST", "/api/v1/preview", json.dumps({"document_id": document_id}))
            preview = json.loads(payload)
            checks["preview_200"] = status == 200 and preview.get("source_verified") is True
            checks["no_index_write"] = preview.get("persisted") is False
            checks["no_external_calls"] = preview.get("external_calls") == 0
            write_json(output / "preview.json", preview)
            return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                    "elements": len(preview.get("elements", [])),
                    "element_types": dict(Counter(e["kind"] for e in preview.get("elements", [])))}
        except Exception as exc:
            return {"status": "failed", "error_type": type(exc).__name__, "note": "查看 server.log"}
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def main() -> int:
    base = ROOT / "evals/results/stage0a"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = base / run_id
    output.mkdir(parents=True)
    config = json.loads((ROOT / "evals/configs/stage0a.json").read_text(encoding="utf-8"))
    assert config["mode"] == "offline" and config["external_calls_allowed"] is False
    app_tests = run_tests(Path(sys.executable), "backend/tests", "app-tests", output)
    data_python = ROOT / (".venv-data/Scripts/python.exe" if os.name == "nt" else ".venv-data/bin/python")
    data_tests = run_tests(data_python, "tests", "data-tests", output)
    smoke = live_smoke(output, config["demo_document_id"])
    tracked = sorted(list((ROOT / "backend").rglob("*.py")) +
                     [ROOT / p for p in ["pyproject.toml", "uv.lock", ".env.example", "scripts/verify_stage0a.py", "evals/configs/stage0a.json"]])
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    bundle = json.loads((ROOT / "data/manifests/bundle.json").read_text(encoding="utf-8"))
    passed = all(r["status"] == "passed" for r in [app_tests, data_tests, smoke])
    report = {"phase": "0A", "run_id": run_id, "status": "passed" if passed else "failed",
              "python": sys.version, "source_sha256": source_hash, "source_files": hashes,
              "data_bundle": bundle["bundle_id"], "config": config,
              "versions": {name: importlib.metadata.version(name) for name in
                           ["fastapi", "pydantic", "pydantic-settings", "uvicorn", "markdown-it-py", "pytest", "httpx2"]},
              "app_tests": app_tests, "data_regression": data_tests, "http_smoke": smoke,
              "model_evaluation_run": False, "external_api_calls": 0, "database_connected": False,
              "limitations": ["只有离线结构预览，尚无知识索引、数据库、OCR/VLM、RAG 或 Agent。",
                              "图像只保留原文引用；未下载、解码或验证图像资源。",
                              "中文 180 条候选仍待人工审核，程序测试不是模型质量评分。",
                              "Swagger 文档页默认依赖 CDN 静态资源；JSON API 和 OpenAPI 本身可离线工作。"]}
    write_json(output / "summary.json", report)
    write_json(base / "summary.json", report)
    rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{r['status']}</td><td>{r.get('passed', '—')} / {r.get('total', '—')}</td></tr>"
                   for name, r in [("应用单元/接口测试", app_tests), ("原数据回归", data_tests), ("实际 HTTP 启动检查", smoke)])
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>0A 离线工程验收</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#243342}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:10px;text-align:left}}code{{overflow-wrap:anywhere}}a{{color:#126781}}</style>
<h1>0A 离线工程验收：{report['status']}</h1><p>运行：{run_id} · 数据：{html.escape(bundle['bundle_id'])}</p>
<p>本报告只证明工程与来源链路，不包含检索召回率或 Agent 成功率；外部 API 调用 0 次。</p>
<table><tr><th>检查</th><th>状态</th><th>通过 / 总数</th></tr>{rows}</table>
<h2>真实开发文档预览</h2><p>元素数：{smoke.get('elements', 0)}；分类：{html.escape(str(smoke.get('element_types', {})))}</p>
<p><a href="{run_id}/preview.json">元素及原文行号 JSON</a> · <a href="{run_id}/server.log">本机服务日志</a></p>
<h2>可复查记录</h2><p><a href="{run_id}/app-tests.log">应用测试日志</a> · <a href="{run_id}/app-tests.xml">应用 JUnit</a> · <a href="{run_id}/data-tests.log">数据回归日志</a> · <a href="{run_id}/data-tests.xml">数据 JUnit</a> · <a href="{run_id}/summary.json">完整运行快照</a></p>
<p>源码 SHA256：<code>{source_hash}</code></p><h2>当前边界</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in report['limitations'])}</ul></html>"""
    (base / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps({"status": report["status"], "app_tests": app_tests, "data_tests": data_tests,
                      "http_smoke": smoke, "report": str(base / "report.html")}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
