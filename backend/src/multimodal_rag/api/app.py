"""FastAPI 应用工厂：测试可注入配置，不在 import 时连接数据库或供应商。"""
import time
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from multimodal_rag import __version__
from multimodal_rag.application.dependencies import dependency_report
from multimodal_rag.application.preview import preview_document
from multimodal_rag.application.qa_service import answer
from multimodal_rag.application.multimodal_qa_service import answer_multimodal
from multimodal_rag.application.ingestion_worker import enqueue_document, run_once
from multimodal_rag.application.retrieval_service import retrieve
from multimodal_rag.application.visual_evidence import locate_visual_evidence, read_active_visual_element
from multimodal_rag.core.errors import AppError
from multimodal_rag.core.models import Identifier, IngestionJobStatus, PreviewResult, QualityReviewResult
from multimodal_rag.infrastructure.logging import configure_logging, log_event
from multimodal_rag.infrastructure.settings import Settings, load_settings
from multimodal_rag.infrastructure.review_repository import get_job, record_quality_review
from multimodal_rag.infrastructure.documents import WhitelistDocumentReader
from multimodal_rag.infrastructure.index_lifecycle import activate_index
from multimodal_rag.infrastructure.release_repository import list_indexes, list_review_queue
from multimodal_rag.infrastructure.knowledge_repository import list_knowledge
from multimodal_rag.infrastructure.evaluation_repository import get_evaluation, list_evaluations


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: Identifier


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: str
    decision: str
    notes: str


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    dense_top_k: int = Field(default=10, ge=1, le=20)
    keyword_top_k: int = Field(default=10, ge=1, le=20)
    fusion_top_k: int = Field(default=10, ge=1, le=20)
    final_top_k: int = Field(default=5, ge=1, le=20)
    max_requests: int = Field(default=2, ge=1, le=2)


class QARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    max_context_chars: int = Field(default=12000, ge=1000, le=30000)
    max_requests: int = Field(default=3, ge=3, le=3)


class VisualEvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_ids: list[Identifier] = Field(min_length=1, max_length=20)
    max_images: int = Field(default=3, ge=1, le=5)


class MultimodalQARequest(QARequest):
    max_images: int = Field(default=3, ge=1, le=3)


class IngestionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: Identifier


class IndexBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    processing_version_ids: list[Identifier] = Field(min_length=1, max_length=100)
    max_requests: int = Field(ge=1, le=128)
    confirm_live: bool = False


class IndexActivateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer: str = Field(min_length=1, max_length=128)
    notes: str = Field(min_length=1, max_length=4000)
    confirm: bool = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="Multimodal RAG Platform", version=__version__, debug=False)
    app.state.settings = settings

    @app.get("/", tags=["health"])
    def root():
        return {"service": "multimodal-rag", "phase": "v1-release-candidate", "docs": "/docs",
                "frontend": "http://127.0.0.1:5173"}

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        # 由服务端创建编号，不将任意客户端头部内容直接写入日志。
        request.state.request_id = uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # 不回显或记录可能携带凭证/原文的异常文本。
            log_event("request_failed", request_id=request.state.request_id, error_code="internal_error")
            response = JSONResponse(status_code=500, content={"error": {
                "code": "internal_error", "message": "内部错误，请凭请求编号定位",
                "request_id": request.state.request_id}})
        response.headers["X-Request-ID"] = request.state.request_id
        route = request.scope.get("route")
        log_event("http_request", request_id=request.state.request_id, method=request.method,
                  route=getattr(route, "path", "unmatched"), status=response.status_code,
                  duration_ms=round((time.perf_counter() - started) * 1000, 2))
        return response

    @app.exception_handler(AppError)
    async def application_error(request: Request, exc: AppError):
        log_event("request_failed", request_id=request.state.request_id, error_code=exc.code)
        return JSONResponse(status_code=exc.status, content={"error": {
            "code": exc.code, "message": exc.message, "request_id": request.state.request_id}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": {
            "code": "invalid_request", "message": "请求格式错误：仅接受合法的 document_id",
            "request_id": request.state.request_id}})

    @app.get("/health/live", tags=["health"])
    def live():
        return {"status": "alive", "mode": settings.mode,
                "phase": "v1-release-candidate", "version": __version__}

    @app.get("/health/ready", tags=["health"], responses={503: {"description": "外部依赖尚未接入"}})
    def ready():
        return JSONResponse(status_code=503, content=dependency_report(settings))

    @app.get("/api/v1/dependencies", tags=["diagnostics"])
    def dependencies():
        return dependency_report(settings)

    @app.post("/api/v1/preview", response_model=PreviewResult, tags=["offline-preview"])
    def preview(body: PreviewRequest):
        return preview_document(body.document_id, settings)

    @app.get("/api/v1/ingestion/jobs/{job_id}", response_model=IngestionJobStatus, tags=["ingestion"])
    def ingestion_job(job_id: str):
        return get_job(settings, job_id)

    @app.get("/api/v1/ingestion/catalog", tags=["ingestion"])
    def ingestion_catalog(limit: int = Query(default=50, ge=1, le=200),
                          offset: int = Query(default=0, ge=0)):
        entries = WhitelistDocumentReader(
            settings.project_root, settings.max_document_bytes, settings.ingestion_manifest
        ).list_allowed()
        selected = entries[offset:offset + limit]
        return {
            "documents": [{"document_id": entry.document_id, "title": entry.title,
                           "format": entry.format, "source_path": entry.path,
                           "knowledge_base": entry.knowledge_base}
                          for entry in selected],
            "pagination": {"limit": limit, "offset": offset,
                            "returned": len(selected), "total": len(entries)},
        }

    @app.post("/api/v1/ingestion/jobs", tags=["ingestion"])
    def create_ingestion_job(body: IngestionCreateRequest):
        # enqueue_document revalidates the manifest entry before touching the DB.
        return enqueue_document(settings, body.document_id)

    @app.post("/api/v1/ingestion/jobs/{job_id}/run", tags=["ingestion"])
    def run_ingestion_job(job_id: str):
        worker_id = "api-" + uuid4().hex
        result = run_once(settings, worker_id, job_id=job_id)
        if result is None:
            current = get_job(settings, job_id)
            if current["status"] not in {"running", "succeeded"}:
                raise AppError("job_not_available", "入库任务当前不可执行", 409)
            # 幂等重试或其他worker已领取时，返回数据库中的当前状态。
            return {"result": None, "job": current}
        return {"result": result, "job": get_job(settings, job_id)}

    @app.post("/api/v1/processing/{processing_version_id}/review",
              response_model=QualityReviewResult, tags=["ingestion"])
    def review_processing(processing_version_id: str, body: ReviewRequest):
        return record_quality_review(settings, processing_version_id, body.reviewer, body.decision, body.notes)

    @app.get("/api/v1/review-queue", tags=["ingestion"])
    def review_queue(state: str = Query(default="pending", pattern="^(pending|approved|all)$"),
                     limit: int = Query(default=50, ge=1, le=100),
                     offset: int = Query(default=0, ge=0)):
        return list_review_queue(settings, state=state, limit=limit, offset=offset)

    @app.get("/api/v1/indexes", tags=["indexes"])
    def indexes(status: str = Query(default="all", pattern="^(all|draft|active|retired|failed)$"),
                limit: int = Query(default=50, ge=1, le=100)):
        return list_indexes(settings, status=status, limit=limit)

    @app.post("/api/v1/indexes/build", tags=["indexes"])
    def build_index(body: IndexBuildRequest):
        if not body.confirm_live:
            raise AppError("live_confirmation_required", "构建索引会调用Embedding，请明确确认并提供预算", 422)
        from multimodal_rag.application.index_builder import build_real_indexes_for_processing_versions
        return build_real_indexes_for_processing_versions(
            settings, body.processing_version_ids, max_requests=body.max_requests
        )

    @app.post("/api/v1/indexes/{index_version_id}/activate", tags=["indexes"])
    def activate(index_version_id: str, body: IndexActivateRequest):
        if not body.confirm:
            raise AppError("activation_confirmation_required", "激活会切换当前可读索引，请明确确认", 422)
        result = activate_index(settings, index_version_id)
        return {**result, "reviewer": body.reviewer, "notes": body.notes}

    @app.post("/api/v1/retrieval/search", tags=["retrieval"])
    def retrieval_search(body: RetrievalRequest):
        return retrieve(settings, body.query, dense_top_k=body.dense_top_k,
                        keyword_top_k=body.keyword_top_k, fusion_top_k=body.fusion_top_k,
                        final_top_k=body.final_top_k, max_requests=body.max_requests)

    @app.post("/api/v1/visual-evidence/locate", tags=["retrieval"])
    def visual_evidence(body: VisualEvidenceRequest):
        return locate_visual_evidence(settings, body.chunk_ids, max_images=body.max_images)

    @app.get("/api/v1/visual-elements/{element_id}/content", tags=["retrieval"])
    def visual_element_content(element_id: Identifier):
        asset = read_active_visual_element(settings, element_id)
        return Response(content=asset["content"], media_type=asset["mime_type"], headers={
            "Cache-Control": "private, max-age=3600", "ETag": f'"{asset["sha256"]}"',
            "X-Content-Type-Options": "nosniff",
        })

    @app.get("/api/v1/knowledge", tags=["knowledge"])
    def knowledge(status: str = Query(default="all", pattern="^(all|active|draft|review)$"),
                  limit: int = Query(default=50, ge=1, le=100),
                  offset: int = Query(default=0, ge=0)):
        return list_knowledge(settings, status=status, limit=limit, offset=offset)

    @app.get("/api/v1/evaluations", tags=["evaluations"])
    def evaluations(kind: str = Query(default="all", pattern="^(all|retrieval|qa)$"),
                    limit: int = Query(default=50, ge=1, le=100)):
        return list_evaluations(settings, kind=kind, limit=limit)

    @app.get("/api/v1/evaluations/{run_id}", tags=["evaluations"])
    def evaluation(run_id: str):
        return get_evaluation(settings, run_id)

    @app.post("/api/v1/qa", tags=["qa"])
    def qa(body: QARequest):
        return answer(settings, body.query, max_context_chars=body.max_context_chars,
                      max_requests=body.max_requests)

    @app.post("/api/v1/qa/multimodal", tags=["qa"])
    def multimodal_qa(body: MultimodalQARequest):
        return answer_multimodal(settings, body.query, max_context_chars=body.max_context_chars,
                                 max_images=body.max_images, max_requests=body.max_requests)

    return app
