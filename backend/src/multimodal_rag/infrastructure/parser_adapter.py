"""MinerU任务/结果适配。提交与查询分开，超时保留任务ID，不重新提交。

0B验证结果包是否含正文/结构JSON/图片资源；页坐标归一化及正文质量留到0C。
ZIP仅在内存中检查，不对不可信归档执行extractall。
"""
import hashlib
from io import BytesIO
import json
from pathlib import PurePosixPath
import re
import stat
from urllib.parse import urlsplit
from zipfile import ZipFile, BadZipFile

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import HttpGateway
from multimodal_rag.infrastructure.model_adapters import invalid, successful


def checked_task_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise AppError("invalid_task_id", "任务编号格式错误", 422)
    return value


class ParserAdapter:
    def __init__(self, gateway: HttpGateway):
        if gateway.service != "parser":
            raise ValueError("parser gateway required")
        self.gateway = gateway

    def submit_url(self, url: str):
        """0B只开放官方公开样例，最多解析前两页；不允许任意外部/内网URL。"""
        if url != "https://cdn-mineru.openxlab.org.cn/demo/example.pdf":
            raise AppError("unapproved_parser_sample", "0B URL验证仅允许官方公开样例", 403)
        data = self.gateway.request("POST", "/extract/task", payload={"url": url,
            "model_version": self.gateway.settings.parser_model, "page_ranges": "1-2",
            "enable_table": True, "enable_formula": True, "language": "ch"})
        try:
            task_id = checked_task_id(data["data"]["task_id"])
        except (KeyError, TypeError, AppError):
            invalid(self.gateway)
        successful(self.gateway, data)
        return {"task_id": task_id, "kind": "single", "state": "submitted"}

    def request_upload(self, filename: str, data_id: str):
        return self.request_document_upload(filename, data_id, page_count=2, max_pages=2)

    def request_document_upload(self, filename: str, data_id: str, *, page_count: int,
                                max_pages: int = 200):
        """Request one resumable MinerU upload for a validated local PDF.

        The signed URL is deliberately returned only to the current process.  The
        durable job stores the batch id, never the temporary URL.
        """
        checked_task_id(data_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}\.pdf", filename):
            raise AppError("invalid_filename", "上传文件名必须是安全的PDF基名", 422)
        if (not isinstance(page_count, int) or isinstance(page_count, bool)
                or not 1 <= page_count <= max_pages):
            raise AppError("pdf_page_limit_exceeded", "PDF 页数超出入库限制", 422)
        data = self.gateway.request("POST", "/file-urls/batch", payload={
            "files": [{"name": filename, "data_id": data_id,
                       "page_ranges": f"1-{page_count}"}],
            "model_version": self.gateway.settings.parser_model, "enable_table": True,
            "enable_formula": True, "language": "ch"})
        try:
            batch_id = checked_task_id(data["data"]["batch_id"])
            urls = data["data"]["file_urls"]
            if not isinstance(urls, list) or len(urls) != 1 or not isinstance(urls[0], str):
                invalid(self.gateway)
            self.gateway._transfer_url(urls[0])
        except (KeyError, TypeError):
            invalid(self.gateway)
        successful(self.gateway, data)
        # signed_url是敏感运行时值，不能写日志或最终报告。
        return {"batch_id": batch_id, "signed_url": urls[0], "data_id": data_id}

    def upload(self, signed_url: str, content: bytes):
        return self.upload_document(signed_url, content, max_bytes=2_097_152)

    def upload_document(self, signed_url: str, content: bytes, *, max_bytes: int = 20_971_520):
        if not content.startswith(b"%PDF-") or len(content) > max_bytes:
            raise AppError("invalid_parser_sample", "PDF 签名错误或超过本次上传大小限制", 422)
        self.gateway.request("PUT", "", transfer_url=signed_url, content=content, binary=True)
        return {"uploaded": True, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    def poll(self, task_id: str, *, batch=False, data_id: str | None = None):
        task_id = checked_task_id(task_id)
        path = f"/extract-results/batch/{task_id}" if batch else f"/extract/task/{task_id}"
        data = self.gateway.request("GET", path)
        try:
            row = data["data"]
            if batch:
                rows = row["extract_result"]
                rows = [r for r in rows if r.get("data_id") == data_id]
                if len(rows) != 1:
                    invalid(self.gateway)
                row = rows[0]
            elif row.get("task_id") != task_id:
                invalid(self.gateway)
            state = row["state"]
            if state not in {"pending", "running", "converting", "waiting-file", "done", "failed"}:
                invalid(self.gateway)
            result = {"task_id": task_id, "state": state, "kind": "batch" if batch else "single"}
            if state == "done":
                url = row["full_zip_url"]
                self.gateway._transfer_url(url)
                result["signed_result_url"] = url
            progress = row.get("extract_progress")
            if isinstance(progress, dict):
                extracted, total = progress.get("extracted_pages"), progress.get("total_pages")
                if type(extracted) is not int or type(total) is not int or not 0 <= extracted <= total:
                    invalid(self.gateway)
                result["progress"] = {"extracted_pages": extracted, "total_pages": total}
        except (KeyError, TypeError, AttributeError):
            invalid(self.gateway)
        successful(self.gateway, data)
        # 不返回err_msg原文，其中可能包含原始URL或认证信息。
        return result

    def fetch_result(self, signed_url: str):
        content = self.fetch_result_bytes(signed_url)
        return inspect_result_archive(content, self.gateway.settings.parser_max_unpacked_bytes)

    def fetch_result_bytes(self, signed_url: str) -> bytes:
        content = self.gateway.request("GET", "", transfer_url=signed_url, binary=True,
                                       max_bytes=self.gateway.settings.parser_max_zip_bytes)
        try:
            inspect_result_archive(content, self.gateway.settings.parser_max_unpacked_bytes)
        except AppError:
            self.gateway.records[-1]["outcome"] = "invalid_parser_archive"
            raise
        self.gateway.records[-1]["outcome"] = "validated_archive"
        return content


def inspect_result_archive(content: bytes, max_unpacked: int) -> dict:
    try:
        with ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > 5000 or sum(i.file_size for i in entries) > max_unpacked:
                raise ValueError("archive size")
            names = [i.filename for i in entries]
            if len(names) != len(set(names)):
                raise ValueError("duplicate names")
            for item in entries:
                path = PurePosixPath(item.filename)
                if (path.is_absolute() or ".." in path.parts or "\\" in item.filename or ":" in item.filename
                        or stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1):
                    raise ValueError("unsafe entry")
            markdown = [n for n in names if PurePosixPath(n).name == "full.md"]
            structured = [n for n in names if n.endswith("content_list.json")]
            if len(markdown) != 1 or len(structured) != 1:
                raise ValueError("missing or ambiguous structure")
            text = archive.read(markdown[0]).decode("utf-8")
            rows = json.loads(archive.read(structured[0]))
            if not text.strip() or not isinstance(rows, list) or not rows:
                raise ValueError("empty parsed output")
            pages = set()
            for row in rows:
                if not isinstance(row, dict) or type(row.get("page_idx")) is not int or row["page_idx"] < 0:
                    raise ValueError("missing page mapping")
                pages.add(row["page_idx"])
            images = [n for n in names if PurePosixPath(n).suffix.lower() in {".png", ".jpg", ".jpeg"}]
            references = [row["img_path"] for row in rows if row.get("img_path")]
            available = set(names)
            base = PurePosixPath(structured[0]).parent
            missing = sum(1 for ref in references if not isinstance(ref, str) or
                          (base / ref).as_posix() not in available)
            return {"archive_sha256": hashlib.sha256(content).hexdigest(), "markdown_characters": len(text),
                    "structured_elements": len(rows), "provider_page_indices": sorted(pages),
                    "image_files": len(images), "image_references": len(references),
                    "missing_image_references": missing, "source_mapping_normalized": False,
                    "status": "structure_checked" if not missing else "incomplete_resources"}
    except (BadZipFile, ValueError, KeyError, TypeError, UnicodeError, OSError, RuntimeError):
        raise AppError("invalid_parser_archive", "解析结果包不完整、超限或来源结构不合法", 502) from None
