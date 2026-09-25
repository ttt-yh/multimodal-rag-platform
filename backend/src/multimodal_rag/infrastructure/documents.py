"""开发侧白名单读取：客户端只传文档编号，不能传任意文件路径。

白名单是本地可信配置，不接受 API 上传；解析前校验路径、格式、分组和哈希。
Markdown/TXT 以 UTF-8 文本返回；PDF 只返回经过签名、大小和哈希校验的字节。
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
                 manifest_path: str = "data/manifests/ingestion_chinese_md.jsonl",
                 max_pdf_bytes: int = 20_971_520,
                 runtime_manifest_path: str = "data/manifests/runtime_uploads.jsonl"):
        self.root = root.resolve()
        # 正式白名单可以组合多个公开数据源；安全边界是 data/raw，而不是某个
        # TiDB 子目录。客户端仍然不能提交任意本地路径。
        self.source_root = (self.root / "data/raw").resolve()
        self.manifest_root = (self.root / "data/manifests").resolve()
        self.manifest = self._manifest_path(manifest_path)
        self.runtime_manifest = self._manifest_path(runtime_manifest_path)
        self.max_bytes = max_bytes
        self.max_pdf_bytes = max_pdf_bytes

    def _manifest_path(self, value: str) -> Path:
        relative = Path(value)
        # manifest 只能来自项目内的 manifests 目录，避免配置项变成任意文件读取入口。
        if (relative.is_absolute() or PureWindowsPath(value).drive or
                ".." in relative.parts or "\\" in value or ":" in value):
            raise AppError("unsafe_manifest", "入库白名单路径不在允许范围内", 403)
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.manifest_root) or candidate.suffix.lower() != ".jsonl":
            raise AppError("unsafe_manifest", "入库白名单必须位于 data/manifests 目录且为 JSONL", 403)
        return candidate

    def _entries(self) -> list[ManifestEntry]:
        """Load the immutable corpus manifest plus the optional runtime upload catalog."""
        rows: list[ManifestEntry] = []
        paths = ((self.manifest, True), (self.runtime_manifest, False))
        try:
            for path, required in paths:
                if not path.exists() and not required:
                    continue
                lines = path.read_text(encoding="utf-8").splitlines()
                for line in lines:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("invalid manifest row")
                    rows.append(ManifestEntry(**{k: raw[k] for k in ManifestEntry.model_fields}))
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            raise AppError("manifest_unavailable", "开发文档白名单缺失或格式异常", 503) from None
        identifiers = [row.document_id for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise AppError("manifest_conflict", "白名单文档编号重复", 409)
        return rows

    def list_allowed(self) -> list[ManifestEntry]:
        """Return discoverable development documents without reading their bodies.

        The catalog is intentionally metadata-only.  ``read`` remains the
        authoritative path for hash, size, encoding and source-root checks
        immediately before a document is queued.
        """
        return [entry for entry in self._entries()
                if entry.split == "dev" and entry.format in {"md", "txt", "pdf"}]

    def _entry(self, document_id: str) -> ManifestEntry:
        entries = [entry for entry in self._entries() if entry.document_id == document_id]
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
        if entry.format not in {"md", "txt", "pdf"} or path.suffix.lower() != f".{entry.format}":
            raise AppError("unsupported_format", "正式入库仅支持 Markdown、TXT 和 PDF", 415)
        return entry

    def read_source(self, document_id: str) -> tuple[ManifestEntry, bytes]:
        """Return a hash-verified source without decoding binary PDF content."""
        entry = self._entry(document_id)
        path = (self.root / entry.path).resolve()
        limit = self.max_pdf_bytes if entry.format == "pdf" else self.max_bytes
        try:
            with path.open("rb") as stream:
                payload = stream.read(limit + 1)
        except OSError:
            raise AppError("source_unavailable", "白名单原文不存在或不可读", 404) from None
        if len(payload) > limit:
            raise AppError("document_too_large", "文档超过当前格式的入库大小限制", 413)
        if hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise AppError("source_hash_mismatch", "原文与白名单哈希不一致，请先核查数据版本", 409)
        if entry.format == "pdf" and not payload.startswith(b"%PDF-"):
            raise AppError("invalid_pdf_signature", "PDF 文件签名不正确", 422)
        return entry, payload

    def read(self, document_id: str) -> tuple[ManifestEntry, str]:
        """Read a UTF-8 text source for the local preview/text worker."""
        entry, payload = self.read_source(document_id)
        if entry.format not in {"md", "txt"}:
            raise AppError("preview_format_unsupported", "PDF 需通过带预算确认的 MinerU 入库链路处理", 415)
        try:
            # BOM 作为字符保留，原始定位可以精确复原；解析器仅屏蔽开头的 BOM。
            return entry, payload.decode("utf-8")
        except UnicodeDecodeError:
            raise AppError("invalid_encoding", "0A 文本文档必须为 UTF-8 编码", 422) from None
