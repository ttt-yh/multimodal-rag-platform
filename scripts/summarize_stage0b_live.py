"""复核已有真实调用记录并只读检查项目库；不发送新的模型或解析请求。

报告是已有记录的审计汇总，不把复核时间冒充供应商调用时间。
"""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

from multimodal_rag.infrastructure.database import connection, MIGRATIONS
from multimodal_rag.infrastructure.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def main():
    settings = load_settings()
    directory = ROOT / "evals/results/service-checks"
    selected = {}
    for service in ("chat", "vision", "embedding", "rerank", "parser"):
        paths = sorted(directory.glob(f"*-{service}.json"))
        if not paths:
            raise ValueError(f"missing {service} report")
        selected[service] = (paths[-1], json.loads(paths[-1].read_text(encoding="utf-8")))
    checks = {}
    sources = {}
    def register(path):
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for service, (path, report) in selected.items():
        register(path)
        records = report["records"]
        checks[service + "_validated"] = (report["status"] == "validated" and report["mode"] == "api"
            and report["quality_evaluation"] is False and report["http_requests"] == len(records)
            and all(r["status_code"] == 200 and r["outcome"].startswith("validated") for r in records))
        if service != "parser":
            checks[service + "_current_model"] = report["model"] == getattr(settings, service + "_model")
            checks[service + "_single_request"] = report["http_requests"] == 1
        rows.append({"service": service, "call_run_id": report["run_id"], "status": report["status"],
                     "model": report.get("model"), "http_requests": report["http_requests"],
                     "duration_ms": sum(r["duration_ms"] for r in records), "usage": report.get("usage"),
                     "source_report": path.relative_to(ROOT).as_posix()})
    embedding = selected["embedding"][1]
    checks["embedding_dimensions"] = embedding["dimensions"] == settings.embedding_dimensions and embedding["vectors"] == 2
    vision = selected["vision"][1]
    image = (ROOT / vision["sample"]).resolve()
    checks["vision_source_hash"] = (image.is_relative_to((ROOT / "data/raw").resolve()) and image.is_file()
        and hashlib.sha256(image.read_bytes()).hexdigest() == vision["image_sha256"])
    parser = selected["parser"][1]
    parser_calls = []
    for path in sorted(directory.glob("*-parser.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("task_id") == parser["task_id"]:
            register(path)
            parser_calls.extend(report["records"])
    checks["parser_single_submission"] = sum(r["method"] == "POST" for r in parser_calls) == 1
    checks["parser_budget"] = len(parser_calls) <= 6
    artifact = parser["artifact"]
    checks["parser_two_pages"] = artifact["provider_page_indices"] == [0, 1]
    checks["parser_resources"] = artifact["status"] == "structure_checked" and artifact["missing_image_references"] == 0
    with connection(settings, read_only=True) as conn:
        identity = conn.execute("SELECT current_database(),current_user,current_setting('server_version')").fetchone()
        tables = [r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='mrag' ORDER BY table_name").fetchall()]
        migrations = conn.execute("SELECT name,sha256 FROM mrag.schema_migrations ORDER BY name").fetchall()
    checks["business_tables"] = {"documents", "document_versions", "elements", "ingestion_jobs", "schema_migrations"}.issubset(tables)
    checks["migration_matches"] = bool(migrations) and all((MIGRATIONS / name).is_file() and
        hashlib.sha256((MIGRATIONS / name).read_bytes()).hexdigest() == sha for name, sha in migrations)
    result = {"status": "verified_existing_records" if all(checks.values()) else "failed",
              "reviewed_at": datetime.now(timezone.utc).isoformat(), "evidence_kind": "existing_live_call_reports",
              "new_external_api_calls": 0, "recorded_model_calls": 4, "recorded_parser_requests": len(parser_calls),
              "quality_evaluation": False, "cost": None, "checks": checks, "services": rows,
              "parser_artifact": artifact, "source_sha256": sources,
              "database": {"database": identity[0], "role": identity[1], "version": identity[2], "tables": tables, "read_only": True},
              "limitations": ["已有记录证明当时接口调用成功，不是此次重新请求或持续可用性保证。",
                              "历史解析报告未记录模型字段；当前配置vlm，不将其反推为历史响应的模型身份。",
                              "解析结果只完成结构检查，页码归一化、OCR准确率和图文语义质量尚待0C验证。",
                              "未做RAG检索、Agent任务或真实业务质量评测；不从调用量推算费用。"]}
    output = ROOT / "evals/results/stage0b-live"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    body = "".join(f"<tr><td>{r['service']}</td><td>{html.escape(r['model'] or '未记录')}</td><td>{r['status']}</td><td>{r['duration_ms']:.0f}</td><td><a href='../../..{('/' + r['source_report'])}'>原记录</a></td></tr>" for r in rows)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>0B真实接口记录复核</title>
<style>body{{font:16px/1.7 system-ui;max-width:1000px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border:1px solid #ccc}}</style>
<h1>0B真实接口记录复核：{result['status']}</h1>
<p>本轮新增外部调用0次。复核已有四次模型调用、{len(parser_calls)}次解析相关请求；数据库为本轮只读实查。</p>
<table><tr><th>服务</th><th>模型</th><th>历史结果</th><th>耗时(ms)</th><th>依据</th></tr>{body}</table>
<p>解析样例：2页、{artifact['structured_elements']}个结构元素、{artifact['image_files']}张图片，缺失图片引用0。</p>
<p><a href="summary.json">检查项、来源哈希与数据库状态</a></p><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in result['limitations'])}</ul></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks_passed": sum(checks.values()), "checks_total": len(checks),
                      "new_external_api_calls": 0, "parser_requests_in_records": len(parser_calls),
                      "report": str(output / "report.html")}, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
