"""开发侧白名单读取：客户端只传文档编号，不能传任意文件路径。

白名单是本地可信配置，不接受 API 上传；解析前校验路径、格式、分组和哈希。
图片只保留引用，不在预览阶段读取、下载或执行任何资源。
"""
import hashlib
import json
from pathlib import Path, PureWindowsPath

from pydantic import ValidationError

from multimodal_rag.core.errors import AppError
from multimodal_rag.core.models import Identifier, Model, Sha256


class ManifestEntry(Model):
    # 原清单还有许可/分组等字段；显式选择所需字段，不导入 QA 标注。
    document_id: Identifier
    path: str
    format: str
    sha256: Sha256
    split: str
    title: str
    knowledge_base: str
    source_family: str
    source_revision: str
    license: str


class WhitelistDocumentReader:
    def __init__(self, root: Path, max_bytes: int,
                 manifest_path: str = "data/manifests/ingestion_chinese_md.jsonl"):
        self.root = root.resolve()
        self.source_root = (self.root / "data/raw/tidb_zh/source").resolve()
        manifest_root = (self.root / "data/manifests").resolve()
        relative = Path(manifest_path)
        # manifest 只能来自项目内的 manifests 目录，避免配置项变成任意文件读取入口。
        if (relative.is_absolute() or PureWindowsPath(manifest_path).drive or
                ".." in relative.parts or "\\" in manifest_path or ":" in manifest_path):
            raise AppError("unsafe_manifest", "入库白名单路径不在允许范围内", 403)
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(manifest_root) or candidate.suffix.lower() != ".jsonl":
            raise AppError("unsafe_manifest", "入库白名单必须位于 data/manifests 目录且为 JSONL", 403)
        self.manifest = candidate
        self.max_bytes = max_bytes

    def list_allowed(self) -> list[ManifestEntry]:
        """Return discoverable development documents without reading their bodies.

        The catalog is intentionally metadata-only.  ``read`` remains the
        authoritative path for hash, size, encoding and source-root checks
        immediately before a document is queued.
        """
        try:
            lines = self.manifest.read_text(encoding="utf-8").splitlines()
            entries: list[ManifestEntry] = []
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("invalid manifest row")
                entry = ManifestEntry(**{k: row[k] for k in ManifestEntry.model_fields})
                if entry.split == "dev" and entry.format in {"md", "txt"}:
                    entries.append(entry)
            return entries
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            raise AppError("manifest_unavailable", "开发文档白名单缺失或格式异常", 503) from None

    def read(self, document_id: str) -> tuple[ManifestEntry, str]:
        try:
            lines = self.manifest.read_text(encoding="utf-8").splitlines()
            entries = []
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("invalid manifest row")
                if row.get("document_id") == document_id:
                    entries.append(ManifestEntry(**{k: row[k] for k in ManifestEntry.model_fields}))
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            raise AppError("manifest_unavailable", "开发文档白名单缺失或格式异常", 503) from None
        if not entries:
            raise AppError("document_not_allowed", "文档不在预览白名单中", 404)
        if len(entries) != 1:
            raise AppError("manifest_conflict", "白名单文档编号重复", 409)
        entry = entries[0]
        if entry.split != "dev":
            raise AppError("split_not_allowed", "0A 预览只读取开发侧文档", 403)
        # 同时拒绝 Windows 路径、UNC、反斜杠和备用数据流，跨平台行为一致。
        relative = Path(entry.path)
        if (relative.is_absolute() or PureWindowsPath(entry.path).drive or
                ".." in relative.parts or "\\" in entry.path or ":" in entry.path):
            raise AppError("unsafe_path", "文档路径不在允许范围内", 403)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.source_root) or not path.is_relative_to(self.root):
            raise AppError("unsafe_path", "文档路径不在允许范围内", 403)
        if entry.format not in {"md", "txt"} or path.suffix.lower() != f".{entry.format}":
            raise AppError("unsupported_format", "0A 仅支持 Markdown/TXT 离线预览", 415)
        try:
            with path.open("rb") as stream:
                payload = stream.read(self.max_bytes + 1)
        except OSError:
            raise AppError("source_unavailable", "白名单原文不存在或不可读", 404) from None
        if len(payload) > self.max_bytes:
            raise AppError("document_too_large", "文档超过本地预览大小限制", 413)
        if hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise AppError("source_hash_mismatch", "原文与白名单哈希不一致，请先核查数据版本", 409)
        try:
            # BOM 作为字符保留，原始定位可以精确复原；解析器仅屏蔽开头的 BOM。
            return entry, payload.decode("utf-8")
        except UnicodeDecodeError:
            raise AppError("invalid_encoding", "0A 文本文档必须为 UTF-8 编码", 422) from None
