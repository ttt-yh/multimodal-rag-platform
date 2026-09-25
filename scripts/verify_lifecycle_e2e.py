"""Real PostgreSQL lifecycle smoke test without external model calls.

The script creates one uniquely named temporary Markdown document, lets the
server parse it, approves it, retires it, and verifies that the active index
pointer never changes. The document and audit row are intentionally retained
for traceability; use a disposable database if a clean database is required.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def request(base: str, method: str, path: str, *, query: dict | None = None,
            body: bytes | None = None, content_type: str | None = None) -> dict:
    url = base.rstrip("/") + path
    if query:
        url += "?" + urlencode(query)
    headers = {"Content-Type": content_type} if content_type else {}
    try:
        with urlopen(Request(url, data=body, headers=headers, method=method), timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from None
    except URLError as exc:
        raise RuntimeError(f"无法访问 {url}: {exc.reason}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description="验证文档上传、审核和下线的真实数据库闭环")
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--rebuild-retired", action="store_true",
                        help="下线后构建排除退役版本的草稿索引，但不激活")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"lifecycle-e2e-{stamp}.md"
    source = ("# 生命周期验收临时文档\n\n"
              "这是用于验证版本化入库、质量审核和安全下线流程的临时资料。\n"
              "它不调用外部模型，也不会进入当前 active 索引。\n")

    before = request(args.base_url, "GET", "/api/v1/indexes", query={"status": "active"})
    active_before = [item["index_version_id"] for item in before.get("items", [])]
    uploaded = request(args.base_url, "POST", "/api/v1/ingestion/uploads", query={
        "filename": filename, "knowledge_base": "e2e_lifecycle", "title": filename,
    }, body=source.encode("utf-8"), content_type="text/markdown")
    document_id = uploaded["upload"]["document_id"]
    job_id = uploaded["job"]["job_id"]

    job = None
    for _ in range(30):
        job = request(args.base_url, "GET", f"/api/v1/ingestion/jobs/{job_id}")
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.5)
    if not job or job["status"] != "succeeded":
        raise RuntimeError(f"入库任务未成功完成: {job}")
    processing_id = job.get("processing_version_id")
    if not processing_id:
        raise RuntimeError("任务完成但没有 processing_version_id")

    review = request(args.base_url, "POST", f"/api/v1/processing/{processing_id}/review",
                     body=json.dumps({"reviewer": "lifecycle-e2e", "decision": "approved",
                                      "notes": "自动化生命周期验收，已核对临时文档。"}).encode(),
                     content_type="application/json")
    if review.get("quality_status") != "approved":
        raise RuntimeError(f"审核未通过: {review}")

    retired = request(args.base_url, "POST", f"/api/v1/documents/{document_id}/retire",
                      body=json.dumps({"reviewer": "lifecycle-e2e", "notes": "生命周期验收结束，临时文档下线。",
                                       "confirm": True}).encode(),
                      content_type="application/json")
    if not retired.get("requires_index_rebuild"):
        raise RuntimeError(f"下线结果未要求重建索引: {retired}")

    rebuild = None
    if args.rebuild_retired:
        rebuild = request(args.base_url, "POST", "/api/v1/indexes/build", body=json.dumps({
            "max_requests": 1, "confirm_live": True,
            "build_mode": "rebuild_without_retired",
        }).encode(), content_type="application/json")
        if rebuild.get("status") != "built" or rebuild.get("new_external_api_calls") != 0:
            raise RuntimeError(f"退役版本重建未按预期完成: {rebuild}")

    after = request(args.base_url, "GET", "/api/v1/indexes", query={"status": "active"})
    active_after = [item["index_version_id"] for item in after.get("items", [])]
    if active_after != active_before:
        raise RuntimeError(f"生命周期验收不应切换 active 索引: before={active_before}, after={active_after}")
    result = {"status": "passed", "document_id": document_id, "job_id": job_id,
              "processing_version_id": processing_id, "active_index_unchanged": True,
              "draft_rebuild": rebuild["index_version_id"] if rebuild else None,
              "external_model_calls": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
