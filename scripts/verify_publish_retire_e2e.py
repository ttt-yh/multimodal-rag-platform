"""Real publish -> retire -> rebuild lifecycle check.

This intentionally makes one Embedding request to publish a temporary document,
then removes it through the governed retirement path. Use only with an explicit
``--confirm-live`` flag because the active index is switched twice.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def call(base: str, method: str, path: str, *, query=None, payload=None, content_type=None) -> dict:
    url = base.rstrip("/") + path
    if query:
        url += "?" + urlencode(query)
    headers = {"Content-Type": content_type} if content_type else {}
    try:
        with urlopen(Request(url, data=payload, headers=headers, method=method), timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {exc.read().decode(errors='replace')}") from None
    except URLError as exc:
        raise RuntimeError(f"无法访问 {url}: {exc.reason}") from None


def json_call(base: str, method: str, path: str, body: dict) -> dict:
    return call(base, method, path, payload=json.dumps(body).encode(), content_type="application/json")


def main() -> int:
    parser = argparse.ArgumentParser(description="验证真实 active 索引的发布与下线回归")
    parser.add_argument("--base-url", default="http://127.0.0.1:8013")
    parser.add_argument("--confirm-live", action="store_true",
                        help="确认一次 Embedding 调用和两次 active 索引切换")
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("该验收会调用一次Embedding并切换active索引，请提供 --confirm-live")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"publish-retire-e2e-{stamp}.md"
    source = ("# 发布下线回归临时文档\n\n"
              "这是一份只用于验证索引发布和退役流程的临时 Markdown 文档。\n"
              "它会先进入草稿索引并通过激活门禁，再执行安全下线。\n")
    before = call(args.base_url, "GET", "/api/v1/indexes", query={"status": "active"})
    active_before = before["items"][0]
    base_count = active_before["chunk_count"]

    uploaded = call(args.base_url, "POST", "/api/v1/ingestion/uploads", query={
        "filename": filename, "knowledge_base": "publish_retire_e2e", "title": filename,
    }, payload=source.encode(), content_type="text/markdown")
    document_id, job_id = uploaded["upload"]["document_id"], uploaded["job"]["job_id"]
    job = None
    for _ in range(40):
        job = call(args.base_url, "GET", f"/api/v1/ingestion/jobs/{job_id}")
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.5)
    if not job or job["status"] != "succeeded":
        raise RuntimeError(f"临时文档入库失败: {job}")
    processing_id = job["processing_version_id"]
    review = json_call(args.base_url, "POST", f"/api/v1/processing/{processing_id}/review", {
        "reviewer": "publish-retire-e2e", "decision": "approved",
        "notes": "真实 active 索引发布与退役回归，临时文档内容已核验。",
    })
    if review["quality_status"] != "approved":
        raise RuntimeError(f"临时处理版本审核失败: {review}")

    built = json_call(args.base_url, "POST", "/api/v1/indexes/build", {
        "processing_version_ids": [processing_id], "max_requests": 1,
        "confirm_live": True, "build_mode": "incremental_from_active",
    })
    if built.get("status") != "built" or built.get("new_external_api_calls") != 1:
        raise RuntimeError(f"增量索引构建结果不符合预期: {built}")
    published = json_call(args.base_url, "POST",
                          f"/api/v1/indexes/{built['index_version_id']}/activate", {
                              "reviewer": "publish-retire-e2e",
                              "notes": "临时文档发布回归，确认数量一致。", "confirm": True,
                          })
    if published.get("status") != "active":
        raise RuntimeError(f"临时索引激活失败: {published}")

    retired = json_call(args.base_url, "POST", f"/api/v1/documents/{document_id}/retire", {
        "reviewer": "publish-retire-e2e", "notes": "回归测试结束，移除临时资料。", "confirm": True,
    })
    if not retired.get("requires_index_rebuild"):
        raise RuntimeError(f"临时文档下线未要求重建: {retired}")

    rebuilt = json_call(args.base_url, "POST", "/api/v1/indexes/build", {
        "max_requests": 1, "confirm_live": True, "build_mode": "rebuild_without_retired",
    })
    if rebuilt.get("status") != "built" or rebuilt.get("new_external_api_calls") != 0:
        raise RuntimeError(f"排除退役版本重建结果不符合预期: {rebuilt}")
    released = json_call(args.base_url, "POST",
                         f"/api/v1/indexes/{rebuilt['index_version_id']}/activate", {
                             "reviewer": "publish-retire-e2e",
                             "notes": "已确认退役文档不再属于 active 范围。", "confirm": True,
                         })
    if released.get("status") != "active":
        raise RuntimeError(f"退役后索引激活失败: {released}")

    after = call(args.base_url, "GET", "/api/v1/indexes", query={"status": "active"})["items"][0]
    if after["index_version_id"] != rebuilt["index_version_id"] or after["chunk_count"] != base_count:
        raise RuntimeError(f"最终 active 索引范围未恢复: before={base_count}, after={after}")
    print(json.dumps({
        "status": "passed", "document_id": document_id, "processing_version_id": processing_id,
        "published_index": built["index_version_id"], "retired_rebuild_index": rebuilt["index_version_id"],
        "base_chunk_count": base_count, "final_chunk_count": after["chunk_count"],
        "embedding_calls": 1, "other_external_calls": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
