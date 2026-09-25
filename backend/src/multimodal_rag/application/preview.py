"""一个用例连接四层；预览成功并不代表索引已经发布。"""
import hashlib
from uuid import uuid4

from multimodal_rag.core.models import Document, DocumentVersion, IngestionJob, PreviewResult
from multimodal_rag.infrastructure.documents import WhitelistDocumentReader
from multimodal_rag.infrastructure.settings import Settings
from multimodal_rag.infrastructure.text_parser import PARSER_VERSION, parse_text, verify_sources


def preview_document(document_id: str, settings: Settings) -> PreviewResult:
    entry, text = WhitelistDocumentReader(
        settings.project_root, settings.max_document_bytes, settings.ingestion_manifest,
        settings.max_pdf_bytes, settings.runtime_upload_manifest,
    ).read(document_id)
    document = Document(document_id=entry.document_id, title=entry.title,
                        knowledge_base=entry.knowledge_base, source_path=entry.path,
                        source_family=entry.source_family, format=entry.format, license=entry.license)
    version_key = hashlib.sha256(f"{entry.document_id}:{entry.sha256}".encode()).hexdigest()
    version = DocumentVersion(version_id="ver_" + version_key, document_id=entry.document_id,
                              content_sha256=entry.sha256, source_revision=entry.source_revision)
    elements, warnings = parse_text(document, version, text)
    verify_sources(document, version, text, elements)
    job = IngestionJob(job_id="preview_" + uuid4().hex, document_id=entry.document_id,
                       version_id=version.version_id, operation="preview", status="succeeded", stage="source_verified")
    return PreviewResult(document=document, version=version, job=job, elements=elements,
                         warnings=warnings, parser_version=PARSER_VERSION, source_verified=True)
