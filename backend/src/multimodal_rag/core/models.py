"""统一数据契约：Pydantic 负责校验和 JSON 序列化，不代表已写入数据库。

文档身份与原文版本分开；元素属于版本；定位只记录实际获得的信息。
页码统一为从 1 开始的物理页，坐标统一为左上原点、0～1 的归一化框。
"""
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Document(Model):
    document_id: Identifier
    title: str
    knowledge_base: str
    source_path: str
    source_family: str
    format: Literal["md", "txt", "pdf", "image"]
    license: str


class DocumentVersion(Model):
    version_id: Identifier
    document_id: Identifier
    content_sha256: Sha256
    source_revision: str
    status: Literal["draft", "active", "retired"] = "draft"


class ProcessingVersion(Model):
    processing_version_id: Identifier
    document_id: Identifier
    version_id: Identifier
    parser_version: str
    profile_sha256: Sha256
    quality_status: Literal["candidate", "needs_review", "blocked", "approved"]
    release_status: Literal["draft", "active", "retired"] = "draft"
    element_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)


class IngestionPersistenceResult(Model):
    document_id: Identifier
    version_id: Identifier
    processing_version_id: Identifier
    job_id: Identifier
    quality_status: Literal["candidate", "needs_review", "blocked", "approved"]
    release_status: Literal["draft", "active", "retired"] = "draft"
    element_count: int = Field(ge=0)
    idempotent: bool


class SourceLocation(Model):
    source_path: str
    kind: Literal["text", "pdf", "image"]
    precision: Literal["line", "page", "region", "image"]
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    page_number: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    image_ref: str | None = None
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def consistent_location(self):
        if self.kind == "text":
            if self.precision != "line" or self.line_start is None or self.line_end is None:
                raise ValueError("文本定位必须提供起止行号")
            if self.line_end < self.line_start:
                raise ValueError("结束行不能早于开始行")
            if any(v is not None for v in (self.page_number, self.bbox, self.image_ref,
                                          self.image_width, self.image_height)):
                raise ValueError("文本定位不能包含虚构的页码或图像坐标")
        elif self.kind == "pdf":
            if self.page_number is None or self.precision not in {"page", "region"}:
                raise ValueError("PDF 必须提供物理页码和页级/区域级精度")
            if self.line_start is not None or self.line_end is not None:
                raise ValueError("PDF 不能冒用文本行号")
            if (self.precision == "region") != (self.bbox is not None):
                raise ValueError("区域级定位必须有框；没有框时只能标为页级")
        else:
            if not self.image_ref or not self.image_width or not self.image_height:
                raise ValueError("独立图像必须提供引用和尺寸")
            if self.precision not in {"image", "region"}:
                raise ValueError("独立图像定位精度无效")
            if any(v is not None for v in (self.line_start, self.line_end, self.page_number)):
                raise ValueError("独立图像不能带文本行号或 PDF 页码")
            if (self.precision == "region") != (self.bbox is not None):
                raise ValueError("图像区域精度与坐标必须一致")
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("坐标必须为左上原点的有效归一化框")
        return self


class Element(Model):
    element_id: Identifier
    document_id: Identifier
    version_id: Identifier
    kind: Literal["heading", "text", "table", "image", "code"]
    order: int = Field(ge=0)
    raw_text: str
    heading_path: list[str] = Field(default_factory=list)
    source: SourceLocation
    image_ref: str | None = None
    # 描述与原文分开：0A 不调用 VLM，此字段保持空值。
    generated_description: str | None = None
    processing_version_id: Identifier | None = None
    index_eligible: bool | None = None
    excluded_reason: str | None = None


class Chunk(Model):
    """检索单元；它只引用解析元素，不覆盖元素原文。"""
    chunk_id: Identifier
    document_id: Identifier
    version_id: Identifier
    processing_version_id: Identifier
    ordinal: int = Field(ge=0)
    text: str
    heading_path: list[str] = Field(default_factory=list)
    element_ids: list[Identifier] = Field(min_length=1)
    source_locations: list[SourceLocation] = Field(min_length=1)
    length_unit: Literal["characters"]
    estimated_length: int = Field(ge=0)


class IngestionJob(Model):
    job_id: Identifier
    document_id: Identifier
    version_id: Identifier
    operation: Literal["preview", "ingest"]
    status: Literal["pending", "running", "succeeded", "failed"]
    stage: str


class IngestionJobStatus(Model):
    job_id: Identifier
    document_id: Identifier
    version_id: Identifier
    processing_version_id: Identifier | None = None
    operation: Literal["preview", "ingest"]
    status: Literal["pending", "running", "succeeded", "failed"]
    stage: str
    quality_status: Literal["candidate", "needs_review", "blocked", "approved"] | None = None
    error_code: str | None = None
    attempts: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class QualityReviewResult(Model):
    review_id: Identifier
    processing_version_id: Identifier
    reviewer: str
    decision: Literal["approved", "rejected"]
    quality_status: Literal["approved", "blocked"]
    release_status: Literal["draft", "active", "retired"]


class IndexVersion(Model):
    index_version_id: Identifier
    processing_version_id: Identifier
    embedding_model: str
    embedding_dimensions: int = Field(gt=0)
    bm25_version: str
    chunk_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    keyword_count: int = Field(ge=0)
    artifact_path: str
    status: Literal["draft", "active", "retired", "failed"] = "draft"


class PreviewResult(Model):
    document: Document
    version: DocumentVersion
    job: IngestionJob
    elements: list[Element]
    warnings: list[str]
    parser_version: str
    source_verified: bool
    persisted: Literal[False] = False
    external_calls: Literal[0] = 0

