"""受限的网页上传存储。

上传内容落到 data/raw/uploads 的内容寻址目录，运行时清单使用原子替换更新。
API 从不接受服务器路径；同一知识库中的同名文件复用 document_id，从而形成新版本。
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import threading

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.documents import ManifestEntry
from multimodal_rag.infrastructure.settings import Settings


_manifest_lock = threading.Lock()
_FORMATS = {".md": "md", ".txt": "txt", ".pdf": "pdf"}


def _safe_filename(filename: str) -> str:
    value = filename.strip()
    windows = PureWindowsPath(value)
    if (not value or len(value) > 180 or "\x00" in value or Path(value).name != value
            or windows.name != value or windows.drive or value in {".", ".."}):
        raise AppError("invalid_upload_filename", "文件名不合法，请勿包含目录或盘符", 422)
    if Path(value).suffix.lower() not in _FORMATS:
        raise AppError("unsupported_format", "网页上传仅支持 Markdown、TXT 和 PDF", 415)
    return value


def _runtime_manifest(settings: Settings) -> Path:
    root = settings.project_root.resolve()
    allowed = (root / "data/manifests").resolve()
    value = settings.runtime_upload_manifest
    relative = Path(value)
    if (relative.is_absolute() or PureWindowsPath(value).drive or ".." in relative.parts
            or "\\" in value or ":" in value):
        raise AppError("unsafe_manifest", "运行时上传清单路径不安全", 403)
    target = (root / relative).resolve()
    if not target.is_relative_to(allowed) or target.suffix.lower() != ".jsonl":
        raise AppError("unsafe_manifest", "运行时上传清单必须位于 data/manifests 目录", 403)
    return target


def register_upload(settings: Settings, filename: str, payload: bytes, *,
                    knowledge_base: str = "uploaded_documents", title: str = "") -> ManifestEntry:
    """Validate and persist one upload, then atomically publish its current revision."""
    filename = _safe_filename(filename)
    knowledge_base = knowledge_base.strip()
    if not knowledge_base or len(knowledge_base) > 100:
        raise AppError("invalid_knowledge_base", "知识库名称不能为空且不能超过100个字符", 422)
    if not payload:
        raise AppError("empty_document", "上传文件不能为空", 422)
    suffix = Path(filename).suffix.lower()
    format_name = _FORMATS[suffix]
    limit = settings.max_pdf_bytes if format_name == "pdf" else settings.max_document_bytes
    if len(payload) > limit:
        raise AppError("document_too_large", "上传文件超过当前格式的大小限制", 413)
    if format_name == "pdf":
        if not payload.startswith(b"%PDF-"):
            raise AppError("invalid_pdf_signature", "PDF 文件签名不正确", 422)
    else:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise AppError("invalid_encoding", "Markdown/TXT 必须使用 UTF-8 编码", 422) from None
        if not text.strip():
            raise AppError("empty_document", "上传文件不能为空", 422)

    digest = hashlib.sha256(payload).hexdigest()
    identity = f"{knowledge_base.casefold()}:{filename.casefold()}"
    document_id = "upload_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    relative = Path("data/raw/uploads") / document_id / f"{digest[:16]}{suffix}"
    target = (settings.project_root.resolve() / relative).resolve()
    upload_root = (settings.project_root.resolve() / "data/raw/uploads").resolve()
    if not target.is_relative_to(upload_root):
        raise AppError("unsafe_path", "上传目标路径不安全", 403)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        os.replace(temporary, target)

    display_title = title.strip() or filename
    if len(display_title) > 200:
        raise AppError("invalid_title", "文档标题不能超过200个字符", 422)
    entry = ManifestEntry(
        document_id=document_id,
        path=relative.as_posix(),
        format=format_name,
        sha256=digest,
        split="dev",
        title=display_title,
        knowledge_base=knowledge_base,
        source_family="web_upload",
        source_revision=datetime.now(timezone.utc).strftime("upload-%Y%m%dT%H%M%SZ"),
        license="user-provided",
    )
    manifest = _runtime_manifest(settings)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock:
        existing: list[dict] = []
        if manifest.exists():
            try:
                existing = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
                            if line.strip()]
                if not all(isinstance(row, dict) for row in existing):
                    raise ValueError
            except (OSError, ValueError, json.JSONDecodeError):
                raise AppError("manifest_unavailable", "运行时上传清单损坏，拒绝覆盖", 503) from None
        rows = [row for row in existing if row.get("document_id") != document_id]
        rows.append(entry.model_dump())
        temporary = manifest.with_suffix(".jsonl.part")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                             encoding="utf-8")
        os.replace(temporary, manifest)
    return entry
