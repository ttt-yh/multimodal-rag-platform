<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  ArrowUpRight,
  BarChart3,
  BookOpen,
  Check,
  ChevronDown,
  CircleAlert,
  CircleHelp,
  Clock3,
  Database,
  FilePlus2,
  FileText,
  Filter,
  Gauge,
  Eye,
  Library,
  LoaderCircle,
  Menu,
  MoreHorizontal,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  RefreshCw,
  Search,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from 'lucide-vue-next'

type Section = 'qa' | 'knowledge' | 'evals'
type Evidence = {
  citation: string
  chunk_id: string
  text: string
  metadata: {
    source_path: string
    heading_path: string
    line_start?: number
    line_end?: number
  }
  rerank_score?: number | null
  rrf_score?: number
}

type VisualEvidence = {
  image_citation?: string
  element_id: string
  document_title: string
  image_ref: string
  mime_type?: string
  byte_size?: number
  source?: { source_path?: string; line_start?: number; line_end?: number }
}

type QAResult = {
  query: string
  answer: string
  citations: Evidence[]
  retrieval: { dense_count: number; keyword_count: number; fused_count: number; rerank: string }
  index_version_id: string
  degraded: boolean
  external_calls: number
  answer_mode?: 'text' | 'vision' | 'visual_refusal'
  visual_evidence?: VisualEvidence[]
}

type CatalogEntry = {
  document_id: string
  title: string
  format: 'md' | 'txt' | 'pdf'
  source_path: string
  knowledge_base: string
}

type IngestionJob = {
  job_id: string
  document_id: string
  version_id: string
  processing_version_id?: string | null
  operation: 'ingest'
  status: 'pending' | 'running' | 'succeeded' | 'failed'
  stage: string
  quality_status?: 'candidate' | 'needs_review' | 'blocked' | 'approved' | null
  error_code?: string | null
  attempts?: number
  created_at?: string
  updated_at?: string
}

type ReviewItem = {
  processing_version_id: string
  document_id: string
  title: string
  source_path: string
  format: 'md' | 'txt' | 'pdf'
  version_id: string
  quality_status: 'candidate' | 'needs_review' | 'blocked' | 'approved'
  release_status: 'draft' | 'active' | 'retired'
  element_count: number
  warning_count: number
  chunk_count: number
  updated_at?: string
}

type IndexItem = {
  index_version_id: string
  status: 'draft' | 'active' | 'retired' | 'failed'
  processing_version_id: string
  embedding_model: string
  embedding_dimensions: number
  bm25_version: string
  chunk_count: number
  vector_count: number
  keyword_count: number
  artifact_path: string
}

const section = ref<Section>('qa')
const sidebarOpen = ref(typeof window === 'undefined' ? true : window.innerWidth > 680)
const query = ref('')
const isLoading = ref(false)
const apiStatus = ref<'unknown' | 'online' | 'offline'>('unknown')
const errorMessage = ref('')
const expandedEvidence = ref<string | null>(null)
const answerCopied = ref(false)
const lastResult = ref<QAResult | null>(null)
const recentQueries = ref([
  '如何排查数据库连接超时？',
  'TiKV Region 大小应该如何调整？',
  'TSO 客户端攒批参数有什么作用？',
])
const knowledgeLoading = ref(false)
const catalogLoading = ref(false)
const catalogEntries = ref<CatalogEntry[]>([])
const importOpen = ref(false)
const importDocumentId = ref('')
const uploadFile = ref<File | null>(null)
const uploadKnowledgeBase = ref('uploaded_documents')
const importLoading = ref(false)
const importError = ref('')
const importJob = ref<IngestionJob | null>(null)
const parserConfirm = ref(false)
const parserBudget = ref(4)
const reviewLoading = ref(false)
const reviewItems = ref<ReviewItem[]>([])
const reviewDialogOpen = ref(false)
const reviewTarget = ref<ReviewItem | null>(null)
const reviewDecision = ref<'approved' | 'rejected'>('approved')
const reviewNotes = ref('已核对解析内容、来源定位和警告信息。')
const reviewSubmitting = ref(false)
const reviewError = ref('')
const selectedProcessingIds = ref<string[]>([])
const buildBudget = ref(1)
const buildConfirm = ref(false)
const buildLoading = ref(false)
const buildError = ref('')
const indexItems = ref<IndexItem[]>([])
const hasActiveIndex = computed(() => indexItems.value.some((item) => item.status === 'active'))
const indexLoading = ref(false)
const activateDialogOpen = ref(false)
const activateTarget = ref<IndexItem | null>(null)
const activateReviewer = ref('human-review')
const activateNotes = ref('已核对Chunk、向量和关键词索引数量一致，确认发布。')
const activateConfirm = ref(false)
const activateLoading = ref(false)
const activateError = ref('')
const knowledgeFilter = ref<'all' | 'active' | 'draft' | 'review'>('all')
const knowledgeSummary = ref({ total_documents: 1109, approved_processing_versions: 1, active_indexes: 1 })
const knowledgeDocuments = ref([
  { document_id: 'demo_01', title: 'best-practices-on-public-cloud.md', source_path: 'data/raw/tidb_zh/source/best-practices/best-practices-on-public-cloud.md', format: 'md', knowledge_base: '研发知识库', chunk_count: 20, version: { status: 'active', created_at: '2026-09-24T08:12:00Z' }, processing: { quality_status: 'approved', release_status: 'active', warning_count: 0 }, index: { status: 'active', embedding_model: 'text-embedding-v4' } },
])
type EvaluationRun = { run_id: string; kind: 'retrieval' | 'qa'; query: string; index_version_id?: string; status: string; degraded: boolean; external_calls: number; result_count: number; citation_count: number; created_at: string }
type QualityMetrics = { recall_at_k: number; precision_at_k: number; mrr: number; k?: number }
type QaQualityMetrics = { sample_count: number; answerable_count: number; unanswerable_count: number; citation_recall_average: number; citation_validity_rate: number; full_evidence_coverage_rate: number; unanswerable_refusal_rate: number; index_version_id?: string }
type SemanticQualityMetrics = { sample_count: number; required_term_coverage_average: number; correctness_average_4: number; completeness_average_4: number; groundedness_average_4: number; semantic_pass_rate: number; human_review_queue_count: number; evaluation_level: string }
type VisualQualityMetrics = { sample_count: number; visual_answer_rate: number; required_term_coverage_average: number; image_citation_rate: number; image_citation_validity_rate?: number; visual_evidence_group_recall_average?: number; excluded_image_violation_rate?: number; formal_pass_rate?: number; sample_size_warning?: string; known_dataset_issue_count: number; evaluation_level: string }
type VisualSemanticQualityMetrics = { sample_count: number; visual_semantic_pass_rate: number; combined_pass_rate: number; correctness_average_4: number; completeness_average_4: number; groundedness_average_4: number; human_review_queue_count: number; evaluation_level: string; metric_notice?: string }
type FinalTestSummary = { status: string; dataset_version: string; sample_count: number; metrics: { strict_deterministic_pass_rate: number; llm_judge_semantic_pass_rate: number; visual_evidence_group_recall: number; image_citation_validity_rate: number; excluded_image_violation_rate?: number; required_term_coverage_average?: number }; reporting_notice?: string }
type EvaluationDetail = EvaluationRun & {
  answer?: string
  results?: Array<{ chunk_id: string; text: string; metadata?: { source_path?: string; heading_path?: string; line_start?: number; line_end?: number }; rerank_score?: number; rrf_score?: number }>
  citations?: Array<{ citation: string; chunk_id: string; source_path?: string; heading_path?: string; line_start?: number; line_end?: number }>
  retrieval?: { dense_count: number; keyword_count: number; fused_count: number; rerank: string }
  records?: Array<{ service: string; status_code: number; outcome: string; duration_ms?: number; usage?: { total_tokens?: number } }>
}
const evaluationLoading = ref(false)
const evaluationKind = ref<'all' | 'retrieval' | 'qa'>('all')
const evaluationRuns = ref<EvaluationRun[]>([])
const evaluationSummary = ref({ total_runs: 0, retrieval_runs: 0, qa_runs: 0, validated_runs: 0, degraded_runs: 0, external_calls: 0, quality_metrics_available: false })
const evaluationNote = ref('正在读取本地评测与最终测试状态。')
const qualityMetrics = ref<QualityMetrics | null>(null)
const qaQualityMetrics = ref<QaQualityMetrics | null>(null)
const semanticQualityMetrics = ref<SemanticQualityMetrics | null>(null)
const visualQualityMetrics = ref<VisualQualityMetrics | null>(null)
const visualSemanticQualityMetrics = ref<VisualSemanticQualityMetrics | null>(null)
const finalTestSummary = ref<FinalTestSummary | null>(null)
const evaluationDetail = ref<EvaluationDetail | null>(null)
const evaluationDetailLoading = ref(false)
const evaluationDetailError = ref('')

const demoEvidence: Evidence[] = [
  {
    citation: '[1]',
    chunk_id: 'chk_demo_01',
    text: '当连接异常出现时，先确认客户端连接地址、端口和目标实例状态，再检查网络策略与超时配置。',
    metadata: { source_path: 'runbooks/database-connection.md', heading_path: '故障排查 / 连接超时', line_start: 42, line_end: 48 },
    rerank_score: 0.94,
    rrf_score: 0.031,
  },
  {
    citation: '[2]',
    chunk_id: 'chk_demo_02',
    text: '对于反复出现的连接超时，需要记录发生时间、客户端版本、网络路径和服务端日志，再与历史案例进行比对。',
    metadata: { source_path: 'cases/connection-timeout-2025.md', heading_path: '历史案例 / 定位信息', line_start: 18, line_end: 25 },
    rerank_score: 0.87,
    rrf_score: 0.029,
  },
]

const demoResult: QAResult = {
  query: '如何排查数据库连接超时？',
  answer: '建议先按连接地址、端口、实例状态和网络策略的顺序检查，并记录客户端版本与发生时间，最后对照历史案例确认是否存在版本相关问题。[1][2]',
  citations: demoEvidence,
  retrieval: { dense_count: 10, keyword_count: 10, fused_count: 10, rerank: 'validated' },
  index_version_id: 'idx_f7720ce41c33…',
  degraded: false,
  external_calls: 3,
}

const displayResult = computed(() => lastResult.value ?? demoResult)
const isDemo = computed(() => !lastResult.value)
const answerLines = computed(() => displayResult.value.answer.split('\n').filter(Boolean))
const evidenceItems = computed<Evidence[]>(() => displayResult.value.citations.map((item) => ({
  ...item,
  text: item.text ?? '',
  metadata: item.metadata ?? {
    source_path: (item as Evidence & { source_path?: string }).source_path ?? 'unknown',
    heading_path: (item as Evidence & { heading_path?: string }).heading_path ?? '',
    line_start: (item as Evidence & { line_start?: number }).line_start,
    line_end: (item as Evidence & { line_end?: number }).line_end,
  },
})))
const visualItems = computed<VisualEvidence[]>(() => displayResult.value.visual_evidence ?? [])
const selectedImportEntry = computed(() => catalogEntries.value.find(
  (entry) => entry.document_id === importDocumentId.value,
))
const importRequiresParser = computed(() => (
  uploadFile.value?.name.toLowerCase().endsWith('.pdf') || selectedImportEntry.value?.format === 'pdf'
))

function chooseUpload(event: Event) {
  const input = event.target as HTMLInputElement
  uploadFile.value = input.files?.[0] ?? null
  importJob.value = null
  importError.value = ''
}

function navigate(next: Section) {
  section.value = next
  errorMessage.value = ''
  if (next === 'knowledge') {
    loadKnowledge()
    loadCatalog()
    loadReviewQueue()
    loadIndexes()
  }
  if (next === 'evals') loadEvaluations()
}

async function loadCatalog() {
  catalogLoading.value = true
  try {
    const response = await fetch('/api/v1/ingestion/catalog?limit=200&offset=0')
    if (!response.ok) throw new Error(`API ${response.status}`)
    const data = await response.json()
    catalogEntries.value = data.documents ?? []
    if (!importDocumentId.value && catalogEntries.value.length) {
      importDocumentId.value = catalogEntries.value[0].document_id
    }
  } catch {
    importError.value = '无法读取可导入文档清单，请确认后端和白名单配置正常。'
  } finally {
    catalogLoading.value = false
  }
}

function openImportDialog() {
  importOpen.value = true
  importError.value = ''
  importJob.value = null
  uploadFile.value = null
  parserConfirm.value = false
  if (!catalogEntries.value.length) loadCatalog()
}

function openImportFromQA() {
  navigate('knowledge')
  openImportDialog()
}

function closeImportDialog() {
  if (!importLoading.value) importOpen.value = false
}

async function submitImport() {
  const documentId = importDocumentId.value.trim()
  if ((!documentId && !uploadFile.value) || importLoading.value) return
  importLoading.value = true
  importError.value = ''
  if (importRequiresParser.value && !parserConfirm.value) {
    importError.value = 'PDF 入库会调用 MinerU，请先确认本次外部解析请求预算。'
    importLoading.value = false
    return
  }
  try {
    let createResponse: Response
    if (uploadFile.value) {
      const params = new URLSearchParams({
        filename: uploadFile.value.name,
        knowledge_base: uploadKnowledgeBase.value.trim() || 'uploaded_documents',
      })
      createResponse = await fetch(`/api/v1/ingestion/uploads?${params.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: uploadFile.value,
      })
    } else {
      createResponse = await fetch('/api/v1/ingestion/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: documentId }),
      })
    }
    const responseBody = await createResponse.json()
    if (!createResponse.ok) {
      throw new Error(responseBody?.error?.message || `创建入库任务失败（${createResponse.status}）`)
    }
    const created = responseBody.job ?? responseBody
    if (responseBody.upload) {
      const uploaded = responseBody.upload as CatalogEntry
      catalogEntries.value = [uploaded, ...catalogEntries.value.filter(
        (entry) => entry.document_id !== uploaded.document_id,
      )]
      importDocumentId.value = uploaded.document_id
      uploadFile.value = null
    }
    importJob.value = {
      ...created,
      operation: 'ingest',
      status: 'pending',
      stage: 'queued',
    }

    if (responseBody.processing === 'background') {
      await waitForBackgroundImport()
    } else {
      await advanceImportJob()
    }
    apiStatus.value = 'online'
    await loadKnowledge()
  } catch (error) {
    apiStatus.value = 'offline'
    importError.value = error instanceof Error
      ? error.message
      : '入库任务未完成，请检查文件、后端服务和数据库连接。'
  } finally {
    importLoading.value = false
  }
}

async function waitForBackgroundImport() {
  if (!importJob.value) return
  for (let attempt = 0; attempt < 12; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 500))
    const response = await fetch(`/api/v1/ingestion/jobs/${encodeURIComponent(importJob.value.job_id)}`)
    if (!response.ok) throw new Error(`job ${response.status}`)
    importJob.value = await response.json() as IngestionJob
    if (importJob.value.status === 'succeeded' || importJob.value.status === 'failed') return
  }
  importError.value = '后台任务仍在处理中，可关闭窗口并稍后在知识库列表中刷新状态。'
}

async function advanceImportJob() {
  if (!importJob.value || importLoading.value && importJob.value.stage !== 'queued') return
  const isPdf = selectedImportEntry.value?.format === 'pdf'
  if (isPdf && !parserConfirm.value) {
    importError.value = '继续 PDF 解析前需要确认 MinerU 调用预算。'
    return
  }
  importLoading.value = true
  importError.value = ''
  try {
    const response = await fetch(`/api/v1/ingestion/jobs/${encodeURIComponent(importJob.value.job_id)}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(isPdf ? { confirm_live: true, max_requests: parserBudget.value } : {}),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data?.error?.message || `run ${response.status}`)
    importJob.value = data.job as IngestionJob
    apiStatus.value = 'online'
    await loadKnowledge()
    await loadReviewQueue()
  } catch (error) {
    importError.value = error instanceof Error ? error.message : '入库任务未完成。'
  } finally {
    importLoading.value = false
  }
}

async function loadKnowledge() {
  knowledgeLoading.value = true
  try {
    const response = await fetch(`/api/v1/knowledge?status=${knowledgeFilter.value}&limit=50&offset=0`)
    if (!response.ok) throw new Error(`API ${response.status}`)
    const data = await response.json()
    knowledgeDocuments.value = data.documents ?? []
    knowledgeSummary.value = data.summary ?? knowledgeSummary.value
    apiStatus.value = 'online'
  } catch {
    // Keep the clearly labelled sample row when the read-only endpoint is unavailable.
    apiStatus.value = 'offline'
  } finally {
    knowledgeLoading.value = false
  }
}

async function retireKnowledgeDocument(documentId: string, title: string) {
  if (!window.confirm(`确认下线“${title}”吗？当前索引不会立即变化。`)) return
  const notes = window.prompt('请输入下线原因：', '资料已过期或不再适用')
  if (!notes?.trim()) return
  try {
    const response = await fetch(`/api/v1/documents/${encodeURIComponent(documentId)}/retire`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer: 'human-review', notes: notes.trim(), confirm: true }),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data?.error?.message || `retire ${response.status}`)
    await loadKnowledge()
    errorMessage.value = '文档已下线；请构建并激活排除退役版本的新索引。'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '文档下线失败。'
  }
}

async function loadReviewQueue() {
  reviewLoading.value = true
  try {
    const response = await fetch('/api/v1/review-queue?state=all&limit=100&offset=0')
    if (!response.ok) throw new Error(`API ${response.status}`)
    const data = await response.json()
    reviewItems.value = data.items ?? []
    apiStatus.value = 'online'
  } catch {
    apiStatus.value = 'offline'
    reviewError.value = '暂时无法读取审核队列。'
  } finally {
    reviewLoading.value = false
  }
}

async function loadIndexes() {
  indexLoading.value = true
  try {
    const response = await fetch('/api/v1/indexes?status=all&limit=50')
    if (!response.ok) throw new Error(`API ${response.status}`)
    const data = await response.json()
    indexItems.value = data.items ?? []
    apiStatus.value = 'online'
  } catch {
    apiStatus.value = 'offline'
    buildError.value = '暂时无法读取索引版本。'
  } finally {
    indexLoading.value = false
  }
}

function toggleProcessing(processingId: string) {
  selectedProcessingIds.value = selectedProcessingIds.value.includes(processingId)
    ? selectedProcessingIds.value.filter((id) => id !== processingId)
    : [...selectedProcessingIds.value, processingId]
}

function openReview(item: ReviewItem) {
  reviewTarget.value = item
  reviewDecision.value = item.quality_status === 'blocked' ? 'rejected' : 'approved'
  reviewNotes.value = '已核对解析内容、来源定位和警告信息。'
  reviewError.value = ''
  reviewDialogOpen.value = true
}

function closeReview() {
  if (!reviewSubmitting.value) reviewDialogOpen.value = false
}

async function submitReview() {
  if (!reviewTarget.value || reviewSubmitting.value) return
  reviewSubmitting.value = true
  reviewError.value = ''
  try {
    const response = await fetch(`/api/v1/processing/${encodeURIComponent(reviewTarget.value.processing_version_id)}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer: 'human-review', decision: reviewDecision.value, notes: reviewNotes.value }),
    })
    if (!response.ok) throw new Error(`API ${response.status}`)
    reviewDialogOpen.value = false
    await Promise.all([loadReviewQueue(), loadKnowledge()])
  } catch {
    reviewError.value = '审核提交失败，请检查处理版本和数据库连接。'
  } finally {
    reviewSubmitting.value = false
  }
}

async function buildIndex() {
  if (!selectedProcessingIds.value.length || !buildConfirm.value || buildLoading.value) return
  buildLoading.value = true
  buildError.value = ''
  try {
    const response = await fetch('/api/v1/indexes/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        processing_version_ids: selectedProcessingIds.value,
        max_requests: buildBudget.value,
        confirm_live: true,
        build_mode: hasActiveIndex.value ? 'incremental_from_active' : 'full',
      }),
    })
    if (!response.ok) throw new Error(`API ${response.status}`)
    selectedProcessingIds.value = []
    buildConfirm.value = false
    await loadIndexes()
  } catch {
    buildError.value = '索引构建失败，可能是Embedding预算不足或处理版本未准备好。'
  } finally {
    buildLoading.value = false
  }
}

function openActivate(item: IndexItem) {
  activateTarget.value = item
  activateConfirm.value = false
  activateError.value = ''
  activateDialogOpen.value = true
}

function closeActivate() {
  if (!activateLoading.value) activateDialogOpen.value = false
}

async function activateIndex() {
  if (!activateTarget.value || !activateConfirm.value || activateLoading.value) return
  activateLoading.value = true
  activateError.value = ''
  try {
    const response = await fetch(`/api/v1/indexes/${encodeURIComponent(activateTarget.value.index_version_id)}/activate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer: activateReviewer.value, notes: activateNotes.value, confirm: true }),
    })
    if (!response.ok) throw new Error(`API ${response.status}`)
    activateDialogOpen.value = false
    await Promise.all([loadIndexes(), loadKnowledge(), loadReviewQueue()])
  } catch {
    activateError.value = '激活失败：请确认该索引范围内所有处理版本均已审核通过，且索引产物完整。'
  } finally {
    activateLoading.value = false
  }
}

watch(knowledgeFilter, () => {
  if (section.value === 'knowledge') loadKnowledge()
})

async function loadEvaluations() {
  evaluationLoading.value = true
  try {
    const response = await fetch(`/api/v1/evaluations?kind=${evaluationKind.value}&limit=50`)
    if (!response.ok) throw new Error(`API ${response.status}`)
    const data = await response.json()
    evaluationRuns.value = data.runs ?? []
    evaluationSummary.value = data.summary ?? evaluationSummary.value
    evaluationNote.value = data.note ?? evaluationNote.value
    qualityMetrics.value = data.quality_metrics ?? null
    qaQualityMetrics.value = data.qa_quality_metrics ?? null
    semanticQualityMetrics.value = data.semantic_quality_metrics ?? null
    visualQualityMetrics.value = data.visual_quality_metrics ?? null
    visualSemanticQualityMetrics.value = data.visual_semantic_quality_metrics ?? null
    finalTestSummary.value = data.final_test_summary ?? null
    apiStatus.value = 'online'
  } catch {
    apiStatus.value = 'offline'
  } finally {
    evaluationLoading.value = false
  }
}

async function openEvaluation(run: EvaluationRun) {
  evaluationDetailLoading.value = true
  evaluationDetailError.value = ''
  evaluationDetail.value = null
  try {
    const response = await fetch(`/api/v1/evaluations/${encodeURIComponent(run.run_id)}`)
    if (!response.ok) throw new Error(`API ${response.status}`)
    evaluationDetail.value = await response.json() as EvaluationDetail
  } catch {
    evaluationDetailError.value = '暂时无法读取这份运行报告，请确认后端仍在运行。'
  } finally {
    evaluationDetailLoading.value = false
  }
}

function closeEvaluation() {
  evaluationDetail.value = null
  evaluationDetailError.value = ''
}

watch(evaluationKind, () => {
  if (section.value === 'evals') loadEvaluations()
})

function useRecent(value: string) {
  query.value = value
  section.value = 'qa'
}

async function runQA() {
  const text = query.value.trim()
  if (!text || isLoading.value) return
  isLoading.value = true
  errorMessage.value = ''
  try {
    const response = await fetch('/api/v1/qa/multimodal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: text, max_context_chars: 12000, max_images: 3, max_requests: 3 }),
    })
    if (!response.ok) throw new Error(`API ${response.status}`)
    lastResult.value = await response.json() as QAResult
    apiStatus.value = 'online'
    recentQueries.value = [text, ...recentQueries.value.filter((item) => item !== text)].slice(0, 3)
  } catch (error) {
    apiStatus.value = 'offline'
    errorMessage.value = '暂时无法连接问答服务，当前展示的是演示证据。请确认后端已启动并检查 API 配置。'
  } finally {
    isLoading.value = false
  }
}

function handleComposerKey(event: KeyboardEvent) {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) runQA()
}

async function copyAnswer() {
  try {
    await navigator.clipboard.writeText(displayResult.value.answer)
    answerCopied.value = true
    window.setTimeout(() => { answerCopied.value = false }, 1600)
  } catch {
    errorMessage.value = '复制失败，请手动选择回答文本。'
  }
}

async function checkHealth() {
  try {
    const response = await fetch('/health/live')
    apiStatus.value = response.ok ? 'online' : 'offline'
  } catch {
    apiStatus.value = 'offline'
  }
}

checkHealth()
</script>

<template>
  <div class="app-shell" :class="{ 'sidebar-collapsed': !sidebarOpen }">
    <aside class="sidebar" :class="{ 'sidebar--hidden': !sidebarOpen }">
      <div class="brand-lockup">
        <div class="brand-mark"><Sparkles :size="18" /></div>
        <div v-if="sidebarOpen">
          <strong>Evidence Lab</strong>
          <span>Multimodal RAG</span>
        </div>
      </div>

      <div v-if="sidebarOpen" class="workspace-switcher">
        <span class="label">当前工作区</span>
        <button class="workspace-button" type="button" disabled title="当前版本仅支持一个本地工作区"><span><span class="status-dot" />研发知识库</span><ChevronDown :size="14" /></button>
      </div>

      <nav class="primary-nav" aria-label="主导航">
        <button :class="['nav-item', { active: section === 'qa' }]" type="button" @click="navigate('qa')">
          <MessageSquareText :size="18" /><span v-if="sidebarOpen">问答工作台</span>
        </button>
        <button :class="['nav-item', { active: section === 'knowledge' }]" type="button" @click="navigate('knowledge')">
          <Library :size="18" /><span v-if="sidebarOpen">知识库</span>
        </button>
        <button :class="['nav-item', { active: section === 'evals' }]" type="button" @click="navigate('evals')">
          <BarChart3 :size="18" /><span v-if="sidebarOpen">评测记录</span>
        </button>
      </nav>

      <div v-if="sidebarOpen" class="sidebar-section">
        <div class="section-label"><span>最近查询</span><button class="icon-button icon-button--dark" title="新建查询" type="button" @click="query = ''; section = 'qa'"><Plus :size="15" /></button></div>
        <button v-for="item in recentQueries" :key="item" class="recent-query" type="button" @click="useRecent(item)">
          <Clock3 :size="14" /><span>{{ item }}</span>
        </button>
      </div>

      <div class="sidebar-footer">
        <button class="nav-item" type="button" disabled title="当前版本不包含多租户工作区设置"><Settings2 :size="18" /><span v-if="sidebarOpen">工作区设置</span></button>
        <div v-if="sidebarOpen" class="profile-row"><div class="avatar">TY</div><div><strong>本地开发环境</strong><span>个人工作区</span></div><MoreHorizontal :size="16" /></div>
      </div>
    </aside>

    <main class="main-area">
      <header class="topbar">
        <div class="topbar-left">
          <button class="icon-button" :title="sidebarOpen ? '收起导航' : '展开导航'" type="button" @click="sidebarOpen = !sidebarOpen">
            <PanelLeftClose v-if="sidebarOpen" :size="19" /><PanelLeftOpen v-else :size="19" />
          </button>
          <span class="breadcrumb">工作台 <span>/</span> {{ section === 'qa' ? '问答' : section === 'knowledge' ? '知识库' : '评测记录' }}</span>
        </div>
        <div class="topbar-actions">
          <span class="connection-state"><span :class="['status-dot', apiStatus]" />{{ apiStatus === 'online' ? 'API 已连接' : apiStatus === 'offline' ? '演示模式' : '检查连接中' }}</span>
          <button class="help-button" type="button" disabled title="使用说明请查看仓库 README 与部署文档"><CircleHelp :size="18" /></button>
        </div>
      </header>

      <div class="content-wrap">
        <section v-if="section === 'qa'" class="qa-view">
          <div class="view-heading">
            <div>
              <p class="eyebrow">证据驱动的研发助手</p>
              <h1>从内部知识中找到<br /><em>可追溯的答案</em></h1>
              <p class="lede">检索已审核的研发资料，回答会附带原文位置，方便复核和继续排查。</p>
            </div>
            <div class="index-status"><ShieldCheck :size="18" /><div><span>索引已激活</span><strong>{{ displayResult.index_version_id }}</strong></div></div>
          </div>

          <div class="composer-wrap">
            <div class="composer-head"><span>输入研发问题</span><span class="shortcut">⌘ + Enter 发送</span></div>
            <textarea v-model="query" placeholder="例如：数据库连接超时时应该检查哪些配置？" rows="3" @keydown="handleComposerKey" />
            <div class="composer-foot"><div class="composer-tools"><button class="tool-button" type="button" @click="openImportFromQA"><Upload :size="15" />添加资料</button><span class="hint">支持基于当前激活知识库检索</span></div><button class="send-button" type="button" :disabled="!query.trim() || isLoading" @click="runQA"><LoaderCircle v-if="isLoading" class="spin" :size="16" /><Send v-else :size="16" />{{ isLoading ? '检索中' : '开始检索' }}</button></div>
            <div v-if="errorMessage" class="inline-error"><X :size="15" />{{ errorMessage }}</div>
          </div>

          <div class="answer-layout">
            <div class="answer-column">
              <div class="section-heading"><div><span class="section-kicker">回答</span><h2>{{ isDemo ? '示例回答' : '本次回答' }}</h2></div></div>
              <article class="answer-panel">
                <div class="answer-meta"><span class="answer-badge"><Check :size="14" />{{ displayResult.answer_mode === 'vision' ? '图文证据已读取' : displayResult.answer_mode === 'visual_refusal' ? '图片证据不可用' : '文本证据已核验' }}</span><span>{{ displayResult.external_calls }} 次模型调用</span></div>
                <p v-for="line in answerLines" :key="line" class="answer-text">{{ line }}</p>
                <div class="answer-footer"><span>基于 {{ evidenceItems.length }} 条引用生成</span><button class="text-action" type="button" @click="copyAnswer">{{ answerCopied ? '已复制' : '复制回答' }} <Check v-if="answerCopied" :size="14" /><ArrowUpRight v-else :size="14" /></button></div>
              </article>
            </div>
            <aside class="run-column">
              <div class="section-heading"><div><span class="section-kicker">本次运行</span><h2>检索轨迹</h2></div><span class="run-id">3 / 3</span></div>
              <div class="trace-panel">
                <div class="trace-step"><span class="trace-icon trace-icon--done"><Check :size="13" /></span><div><strong>查询理解</strong><span>已生成向量表达</span></div><small>0.2s</small></div>
                <div class="trace-line" />
                <div class="trace-step"><span class="trace-icon trace-icon--done"><Check :size="13" /></span><div><strong>混合检索</strong><span>Dense 10 · BM25 10</span></div><small>0.1s</small></div>
                <div class="trace-line" />
                <div class="trace-step"><span class="trace-icon trace-icon--done"><Check :size="13" /></span><div><strong>重排证据</strong><span>{{ displayResult.retrieval.rerank === 'validated' ? 'Rerank 已完成' : '已降级到 RRF' }}</span></div><small>0.3s</small></div>
                <div class="trace-line" />
                <div class="trace-step"><span class="trace-icon trace-icon--active"><Sparkles :size="13" /></span><div><strong>生成回答</strong><span>{{ displayResult.answer_mode === 'vision' ? `VLM读取 ${visualItems.length} 张原图` : '已附加来源引用' }}</span></div><small>2.8s</small></div>
              </div>
              <div class="metric-strip"><div><span>候选证据</span><strong>{{ displayResult.retrieval.fused_count }}</strong></div><div><span>最终引用</span><strong>{{ displayResult.citations.length }}</strong></div><div><span>状态</span><strong class="text-green">{{ displayResult.degraded ? '降级' : '正常' }}</strong></div></div>
            </aside>
          </div>

          <section v-if="visualItems.length" class="evidence-section">
            <div class="section-heading"><div><span class="section-kicker">原图核验</span><h2>视觉证据 <span>{{ visualItems.length }}</span></h2></div></div>
            <div class="visual-evidence-grid">
              <article v-for="item in visualItems" :key="item.element_id" class="visual-evidence-card">
                <img :src="`/api/v1/visual-elements/${encodeURIComponent(item.element_id)}/content`" :alt="`${item.document_title} ${item.image_citation || ''}`" loading="lazy" />
                <div><strong>{{ item.image_citation || '[图]' }} {{ item.document_title }}</strong><span>{{ item.image_ref }}</span><small>{{ item.source?.source_path }} · 第 {{ item.source?.line_start || '—' }} 行</small></div>
              </article>
            </div>
          </section>

          <section class="evidence-section">
            <div class="section-heading"><div><span class="section-kicker">来源核验</span><h2>引用证据 <span>{{ evidenceItems.length }}</span></h2></div><button class="quiet-button" type="button" disabled title="当前版本固定按重排相关性排序"><Filter :size="15" />排序：相关性</button></div>
            <div class="evidence-list">
              <article v-for="item in evidenceItems" :key="item.chunk_id" class="evidence-item" :class="{ expanded: expandedEvidence === item.chunk_id }">
                <button class="evidence-main" type="button" @click="expandedEvidence = expandedEvidence === item.chunk_id ? null : item.chunk_id">
                  <span class="citation-number">{{ item.citation }}</span><div class="evidence-copy"><div class="evidence-title">{{ item.metadata.heading_path || '未命名章节' }}</div><div class="evidence-source"><FileText :size="14" /><span>{{ item.metadata.source_path }}</span><span class="source-location">第 {{ item.metadata.line_start || '—' }}–{{ item.metadata.line_end || '—' }} 行</span></div><p v-if="expandedEvidence === item.chunk_id" class="evidence-excerpt">{{ item.text }}</p></div><div class="evidence-score"><span>重排分数</span><strong>{{ item.rerank_score ? item.rerank_score.toFixed(2) : '—' }}</strong></div><ChevronDown :class="['chevron', { rotated: expandedEvidence === item.chunk_id }]" :size="17" />
                </button>
              </article>
            </div>
          </section>
        </section>

        <section v-else-if="section === 'knowledge'" class="simple-view">
          <div class="view-heading compact">
            <div><p class="eyebrow">内容资产</p><h1>知识库</h1><p class="lede">查看已解析、已审核并参与检索的研发资料。</p></div>
            <button class="primary-button" type="button" @click="openImportDialog"><FilePlus2 :size="16" />导入资料</button>
          </div>
          <div class="summary-grid"><div class="summary-card"><div class="summary-icon teal"><Database :size="18" /></div><span>当前文档</span><strong>{{ knowledgeSummary.total_documents.toLocaleString() }}</strong><small>来自 PostgreSQL</small></div><div class="summary-card"><div class="summary-icon blue"><BookOpen :size="18" /></div><span>审核通过版本</span><strong>{{ knowledgeSummary.approved_processing_versions }}</strong><small>可进入发布流程</small></div><div class="summary-card"><div class="summary-icon amber"><ShieldCheck :size="18" /></div><span>激活索引</span><strong class="text-green">{{ knowledgeSummary.active_indexes }}</strong><small>当前可检索版本</small></div></div>
          <div v-if="importJob" class="ingestion-status" :class="`ingestion-status--${importJob.status}`">
            <div class="ingestion-status-icon"><Check v-if="importJob.status === 'succeeded'" :size="16" /><CircleAlert v-else-if="importJob.status === 'failed'" :size="16" /><LoaderCircle v-else class="spin" :size="16" /></div>
            <div class="ingestion-status-copy"><strong>最近入库任务</strong><span>{{ importJob.document_id }} · {{ importJob.stage }}</span><small v-if="importJob.error_code">错误：{{ importJob.error_code }}</small><small v-else>{{ importJob.status === 'succeeded' ? '解析完成，等待质量审核和索引发布。' : '任务正在处理，页面会保留当前状态。' }}</small></div>
            <span class="status-pill" :class="importJob.status === 'succeeded' ? 'approved' : 'muted'">{{ importJob.status === 'succeeded' ? '已完成' : importJob.status === 'failed' ? '失败' : '处理中' }}</span>
            <button v-if="importJob.status === 'pending' && importRequiresParser" class="quiet-button" type="button" :disabled="importLoading" @click="advanceImportJob"><RefreshCw :class="{ spin: importLoading }" :size="15" />继续 PDF 解析</button>
          </div>
          <div class="table-panel"><div class="table-toolbar"><div class="table-title"><h2>入库资料</h2><span>{{ knowledgeLoading ? '同步中…' : 'PostgreSQL 实时状态' }}</span></div><div class="table-actions"><select v-model="knowledgeFilter" class="filter-select" aria-label="知识库状态筛选"><option value="all">全部状态</option><option value="active">已激活</option><option value="draft">草稿 / 退役</option><option value="review">待审核</option></select><button class="quiet-button" type="button" @click="loadKnowledge"><Search :size="15" />刷新</button></div></div><table><thead><tr><th>文档</th><th>格式</th><th>Chunk</th><th>处理质量</th><th>索引状态</th><th>操作</th></tr></thead><tbody><tr v-for="document in knowledgeDocuments" :key="document.document_id"><td><div class="doc-cell"><span class="file-icon"><FileText :size="16" /></span><div><strong>{{ document.title }}</strong><small>{{ document.source_path }}</small></div></div></td><td>{{ document.format.toUpperCase() }}</td><td>{{ document.chunk_count || '—' }}</td><td><span :class="['status-pill', document.processing.quality_status === 'approved' ? 'approved' : 'muted']">{{ document.processing.quality_status || '未处理' }}</span></td><td><span :class="['table-status', { 'muted-status': document.index.status !== 'active' }]" ><span class="status-dot" />{{ document.index.status || '未建索引' }}</span></td><td><button v-if="document.processing.release_status !== 'retired'" class="table-action table-action--danger" type="button" title="下线文档" @click="retireKnowledgeDocument(document.document_id, document.title)"><X :size="14" />下线</button><span v-else class="muted-status">已下线</span></td></tr><tr v-if="!knowledgeDocuments.length"><td colspan="6" class="empty-cell">当前筛选没有资料</td></tr></tbody></table></div>

          <section class="release-panel">
            <div class="release-panel-head"><div><span class="section-kicker">质量门禁</span><h2>审核队列</h2><p>审核通过后，处理版本才可以进入索引构建。</p></div><button class="quiet-button" type="button" @click="loadReviewQueue"><RefreshCw :size="15" />刷新</button></div>
            <div v-if="reviewError" class="inline-error"><CircleAlert :size="15" />{{ reviewError }}</div>
            <div v-if="reviewLoading" class="release-empty"><LoaderCircle class="spin" :size="17" />正在读取审核队列</div>
            <div v-else-if="!reviewItems.length" class="release-empty"><Check :size="17" />当前没有待审核处理版本</div>
            <div v-else class="review-list">
              <article v-for="item in reviewItems" :key="item.processing_version_id" class="review-row">
                <label class="review-select"><input type="checkbox" :checked="selectedProcessingIds.includes(item.processing_version_id)" :disabled="item.quality_status !== 'approved'" @change="toggleProcessing(item.processing_version_id)" /><span /></label>
                <div class="review-copy"><strong>{{ item.title }}</strong><span>{{ item.processing_version_id }} · {{ item.chunk_count }} Chunks · {{ item.warning_count }} 个警告</span></div>
                <span class="status-pill" :class="item.quality_status === 'approved' ? 'approved' : 'muted'">{{ item.quality_status }}</span>
                <button class="table-action" type="button" @click="openReview(item)">{{ item.quality_status === 'approved' ? '查看审核' : '审核' }}</button>
              </article>
            </div>
            <div class="build-strip"><div><strong>{{ hasActiveIndex ? '增量扩展草稿索引' : '构建首个草稿索引' }}</strong><span>{{ hasActiveIndex ? '复用活动索引已有向量，仅对所选新版本调用 Embedding。' : '仅对已审核版本执行；Embedding 调用按批次计费。' }}</span></div><label class="budget-field">预算<input v-model.number="buildBudget" type="number" min="1" max="128" aria-label="Embedding调用预算" /></label><label class="confirm-check"><input v-model="buildConfirm" type="checkbox" />我确认调用模型</label><button class="primary-button" type="button" :disabled="!selectedProcessingIds.length || !buildConfirm || buildLoading" @click="buildIndex"><LoaderCircle v-if="buildLoading" class="spin" :size="15" /><Database v-else :size="15" />{{ buildLoading ? '构建中' : `构建 ${selectedProcessingIds.length} 个版本` }}</button></div>
            <div v-if="buildError" class="inline-error"><CircleAlert :size="15" />{{ buildError }}</div>
          </section>

          <section class="release-panel">
            <div class="release-panel-head"><div><span class="section-kicker">索引发布</span><h2>索引版本</h2><p>激活前会再次校验审核状态、Chunk数量和索引产物。</p></div><button class="quiet-button" type="button" @click="loadIndexes"><RefreshCw :size="15" />刷新</button></div>
            <div v-if="indexLoading" class="release-empty"><LoaderCircle class="spin" :size="17" />正在读取索引版本</div>
            <div v-else-if="!indexItems.length" class="release-empty">暂无索引版本</div>
            <div v-else class="index-list"><article v-for="item in indexItems" :key="item.index_version_id" class="index-row"><div class="index-marker"><ShieldCheck :size="15" /></div><div class="index-copy"><strong>{{ item.index_version_id }}</strong><span>{{ item.embedding_model }} · {{ item.chunk_count }} / {{ item.vector_count }} / {{ item.keyword_count }}</span></div><span class="status-pill" :class="item.status === 'active' ? 'approved' : 'muted'">{{ item.status }}</span><button v-if="item.status === 'draft'" class="table-action" type="button" @click="openActivate(item)">审核并激活</button></article></div>
            <div v-if="activateError" class="inline-error"><CircleAlert :size="15" />{{ activateError }}</div>
          </section>

          <div v-if="reviewDialogOpen" class="import-overlay" @click.self="closeReview">
            <section class="import-dialog" role="dialog" aria-modal="true" aria-label="审核处理版本"><header class="import-dialog-head"><div><span class="section-kicker">质量审核</span><h2>审核处理版本</h2><p>{{ reviewTarget?.title }} · {{ reviewTarget?.processing_version_id }}</p></div><button class="icon-button" type="button" title="关闭审核窗口" @click="closeReview"><X :size="17" /></button></header><div class="import-dialog-body"><div class="review-facts"><span>元素 {{ reviewTarget?.element_count }}</span><span>Chunk {{ reviewTarget?.chunk_count }}</span><span>警告 {{ reviewTarget?.warning_count }}</span></div><label class="field-label">审核结论</label><div class="decision-toggle"><button type="button" :class="{ selected: reviewDecision === 'approved' }" @click="reviewDecision = 'approved'"><Check :size="14" />通过</button><button type="button" :class="{ selected: reviewDecision === 'rejected' }" @click="reviewDecision = 'rejected'"><CircleAlert :size="14" />驳回</button></div><label class="field-label" for="review-notes">审核说明</label><textarea id="review-notes" v-model="reviewNotes" class="review-notes" rows="4" /><div v-if="reviewError" class="inline-error"><CircleAlert :size="15" />{{ reviewError }}</div></div><footer class="import-dialog-foot"><button class="quiet-button" type="button" :disabled="reviewSubmitting" @click="closeReview">取消</button><button class="primary-button" type="button" :disabled="reviewSubmitting || !reviewNotes.trim()" @click="submitReview"><LoaderCircle v-if="reviewSubmitting" class="spin" :size="15" /><Check v-else :size="15" />提交审核</button></footer></section>
          </div>

          <div v-if="activateDialogOpen" class="import-overlay" @click.self="closeActivate">
            <section class="import-dialog" role="dialog" aria-modal="true" aria-label="激活索引版本"><header class="import-dialog-head"><div><span class="section-kicker">发布确认</span><h2>激活索引版本</h2><p>{{ activateTarget?.index_version_id }} 将成为当前唯一可读索引。</p></div><button class="icon-button" type="button" title="关闭激活窗口" @click="closeActivate"><X :size="17" /></button></header><div class="import-dialog-body"><div class="review-facts"><span>Chunks {{ activateTarget?.chunk_count }}</span><span>Vectors {{ activateTarget?.vector_count }}</span><span>BM25 {{ activateTarget?.keyword_count }}</span></div><label class="field-label" for="activate-notes">发布说明</label><textarea id="activate-notes" v-model="activateNotes" class="review-notes" rows="3" /><label class="confirm-check confirm-check--large"><input v-model="activateConfirm" type="checkbox" />我确认切换当前可读索引</label><div v-if="activateError" class="inline-error"><CircleAlert :size="15" />{{ activateError }}</div></div><footer class="import-dialog-foot"><button class="quiet-button" type="button" :disabled="activateLoading" @click="closeActivate">取消</button><button class="primary-button" type="button" :disabled="activateLoading || !activateConfirm || !activateNotes.trim()" @click="activateIndex"><LoaderCircle v-if="activateLoading" class="spin" :size="15" /><ShieldCheck v-else :size="15" />{{ activateLoading ? '激活中' : '确认激活' }}</button></footer></section>
          </div>

          <div v-if="importOpen" class="import-overlay" @click.self="closeImportDialog">
            <section class="import-dialog" role="dialog" aria-modal="true" aria-label="导入研发资料">
              <header class="import-dialog-head"><div><span class="section-kicker">入库任务</span><h2>导入研发资料</h2><p>上传本地文件或选择预置资料，创建可追踪的版本化入库任务。</p></div><button class="icon-button" type="button" title="关闭导入窗口" @click="closeImportDialog"><X :size="17" /></button></header>
              <div class="import-dialog-body">
                <label class="field-label" for="upload-document">上传本地文档</label>
                <input id="upload-document" class="file-input" type="file" accept=".md,.txt,.pdf,text/markdown,text/plain,application/pdf" :disabled="importLoading || !!importJob" @change="chooseUpload" />
                <div class="upload-meta-row"><label><span>知识库</span><input v-model="uploadKnowledgeBase" maxlength="100" :disabled="importLoading || !!importJob" /></label><small>同一知识库中的同名文件会形成新版本，不会覆盖历史原文。</small></div>
                <div class="import-divider"><span>或使用预置资料</span></div>
                <label class="field-label" for="ingestion-document">预置文档</label>
                <select id="ingestion-document" v-model="importDocumentId" class="import-select" :disabled="catalogLoading || importLoading || !!importJob || !!uploadFile">
                  <option value="" disabled>{{ catalogLoading ? '读取文档清单…' : '请选择一个文档' }}</option>
                  <option v-for="entry in catalogEntries" :key="entry.document_id" :value="entry.document_id">{{ entry.title }} · {{ entry.format.toUpperCase() }}</option>
                </select>
                <p v-if="uploadFile" class="field-hint">待上传：{{ uploadFile.name }} · {{ Math.ceil(uploadFile.size / 1024) }} KB，后端会校验格式、大小和内容签名。</p>
                <p v-else-if="importDocumentId" class="field-hint">编号：{{ importDocumentId }} · 后端会再次校验哈希和文件格式。</p>
                <div v-if="importRequiresParser" class="build-strip pdf-budget-strip"><div><strong>MinerU PDF 解析</strong><span>每次点击只推进一个异步阶段，不会在后台无限轮询。</span></div><label class="budget-field">请求预算<input v-model.number="parserBudget" type="number" min="2" max="8" aria-label="MinerU请求预算" /></label><label class="confirm-check"><input v-model="parserConfirm" type="checkbox" />我确认调用 MinerU</label></div>
                <div v-if="importError" class="inline-error"><CircleAlert :size="15" />{{ importError }}</div>
                <div v-if="importJob" class="import-result"><Check v-if="importJob.status === 'succeeded'" :size="16" /><LoaderCircle v-else class="spin" :size="16" /><span>{{ importJob.status === 'succeeded' ? '任务完成，可在列表中查看质量状态。' : `任务状态：${importJob.stage}` }}</span></div>
              </div>
              <footer class="import-dialog-foot"><button class="quiet-button" type="button" :disabled="importLoading" @click="closeImportDialog">关闭</button><button class="primary-button" type="button" :disabled="(!importDocumentId && !uploadFile) || importLoading || catalogLoading || importJob?.status === 'succeeded'" @click="importJob?.status === 'pending' ? advanceImportJob() : submitImport()"><LoaderCircle v-if="importLoading" class="spin" :size="15" /><Upload v-else :size="15" />{{ importLoading ? '处理中' : importJob?.status === 'pending' ? '继续解析' : uploadFile ? '上传并入库' : '开始入库' }}</button></footer>
            </section>
          </div>
        </section>

        <section v-else class="simple-view">
          <div class="view-heading compact"><div><p class="eyebrow">质量反馈</p><h1>评测记录</h1><p class="lede">查看检索、文本问答与正式视觉测试的真实指标和运行报告。</p></div><button class="primary-button" type="button" :disabled="evaluationLoading" @click="loadEvaluations"><Gauge :size="16" />{{ evaluationLoading ? '刷新中' : '刷新评测' }}</button></div>
          <div class="summary-grid"><div class="summary-card"><div class="summary-icon teal"><BarChart3 :size="18" /></div><span>问答联调</span><strong>{{ evaluationSummary.qa_runs }}</strong><small>已保存报告</small></div><div class="summary-card"><div class="summary-icon blue"><ArrowUpRight :size="18" /></div><span>检索联调</span><strong>{{ evaluationSummary.retrieval_runs }}</strong><small>Dense / BM25 / Rerank</small></div><div class="summary-card"><div class="summary-icon amber"><ShieldCheck :size="18" /></div><span>外部调用</span><strong>{{ evaluationSummary.external_calls }}</strong><small>{{ evaluationSummary.quality_metrics_available ? '评测批次累计' : '联调请求累计' }}</small></div></div>
          <div class="metric-notice"><ShieldCheck :size="16" /><span>{{ evaluationNote }}</span></div>
          <div v-if="qualityMetrics" class="quality-strip"><div><span>Recall@{{ qualityMetrics.k || 5 }}</span><strong>{{ qualityMetrics.recall_at_k.toFixed(2) }}</strong></div><div><span>Precision@{{ qualityMetrics.k || 5 }}</span><strong>{{ qualityMetrics.precision_at_k.toFixed(2) }}</strong></div><div><span>MRR</span><strong>{{ qualityMetrics.mrr.toFixed(2) }}</strong></div></div>
          <div v-if="qaQualityMetrics" class="quality-strip"><div><span>实际引用证据召回</span><strong>{{ qaQualityMetrics.citation_recall_average.toFixed(2) }}</strong></div><div><span>完整证据覆盖</span><strong>{{ qaQualityMetrics.full_evidence_coverage_rate.toFixed(2) }}</strong></div><div><span>无答案拒答率</span><strong>{{ qaQualityMetrics.unanswerable_refusal_rate.toFixed(2) }}</strong></div></div>
          <div v-if="semanticQualityMetrics" class="quality-strip"><div><span>必要要点覆盖</span><strong>{{ semanticQualityMetrics.required_term_coverage_average.toFixed(3) }}</strong></div><div><span>语义完整性 / 4</span><strong>{{ semanticQualityMetrics.completeness_average_4.toFixed(2) }}</strong></div><div><span>待人工复核</span><strong>{{ semanticQualityMetrics.human_review_queue_count }}</strong></div></div>
          <div v-if="visualQualityMetrics" class="quality-strip"><div><span>图文回答完成率</span><strong>{{ visualQualityMetrics.visual_answer_rate.toFixed(2) }}</strong></div><div><span>视觉证据组召回</span><strong>{{ (visualQualityMetrics.visual_evidence_group_recall_average ?? visualQualityMetrics.required_term_coverage_average).toFixed(2) }}</strong></div><div><span>排除图片误选率</span><strong>{{ (visualQualityMetrics.excluded_image_violation_rate ?? 0).toFixed(2) }}</strong></div></div>
          <p v-if="visualQualityMetrics?.sample_size_warning" class="metric-note">{{ visualQualityMetrics.sample_size_warning }}</p>
          <div v-if="finalTestSummary" class="quality-strip"><div><span>严格确定性门禁</span><strong>{{ finalTestSummary.metrics.strict_deterministic_pass_rate.toFixed(2) }}</strong></div><div><span>开发级语义评审</span><strong>{{ finalTestSummary.metrics.llm_judge_semantic_pass_rate.toFixed(2) }}</strong></div><div><span>必要要点覆盖</span><strong>{{ (finalTestSummary.metrics.required_term_coverage_average ?? 0).toFixed(3) }}</strong></div></div>
          <p v-if="finalTestSummary?.reporting_notice" class="metric-note">{{ finalTestSummary.reporting_notice }}</p>
          <p v-else-if="visualSemanticQualityMetrics?.metric_notice" class="metric-note">{{ visualSemanticQualityMetrics.metric_notice }}</p>
          <div class="table-panel"><div class="table-toolbar"><div class="table-title"><h2>运行报告</h2><span>{{ evaluationLoading ? '同步中…' : '来自 evals/results' }}</span></div><div class="table-actions"><select v-model="evaluationKind" class="filter-select" aria-label="评测类型筛选"><option value="all">全部类型</option><option value="retrieval">检索联调</option><option value="qa">问答联调</option></select><button class="quiet-button" type="button" @click="loadEvaluations"><Search :size="15" />刷新</button></div></div><table><thead><tr><th>运行编号</th><th>类型 / 查询</th><th>候选 / 引用</th><th>调用次数</th><th>状态</th><th>操作</th></tr></thead><tbody><tr v-for="run in evaluationRuns" :key="`${run.kind}-${run.run_id}`"><td><strong>{{ run.run_id }}</strong><small class="mono-sub">{{ run.created_at }}</small></td><td><span class="status-pill" :class="run.kind === 'qa' ? 'approved' : 'muted'">{{ run.kind === 'qa' ? '问答' : '检索' }}</span><small class="query-cell">{{ run.query || '无文本查询' }}</small></td><td>{{ run.result_count }} / {{ run.citation_count }}</td><td>{{ run.external_calls }}</td><td><span :class="['table-status', { 'muted-status': run.degraded }]" ><span class="status-dot" />{{ run.degraded ? '降级' : '已验证' }}</span></td><td><button class="table-action" type="button" @click="openEvaluation(run)" title="查看完整运行报告"><Eye :size="14" />详情</button></td></tr><tr v-if="!evaluationRuns.length"><td colspan="6" class="empty-cell">暂无符合条件的报告</td></tr></tbody></table></div>
          <div v-if="evaluationDetailLoading || evaluationDetail || evaluationDetailError" class="evaluation-detail-overlay" @click.self="closeEvaluation"><section class="evaluation-detail" role="dialog" aria-modal="true" aria-label="运行报告详情"><header class="evaluation-detail-head"><div><span class="section-kicker">运行详情</span><h2>{{ evaluationDetail?.kind === 'qa' ? '问答联调报告' : '检索联调报告' }}</h2><p>{{ evaluationDetail?.run_id || '正在读取…' }}</p></div><button class="icon-button" type="button" title="关闭详情" @click="closeEvaluation"><X :size="17" /></button></header><div v-if="evaluationDetailLoading" class="detail-loading"><LoaderCircle class="spin" :size="18" />正在读取运行报告</div><div v-else-if="evaluationDetailError" class="detail-error"><X :size="16" />{{ evaluationDetailError }}</div><template v-else-if="evaluationDetail"><div class="detail-query"><span>查询</span><strong>{{ evaluationDetail.query }}</strong></div><div class="detail-stats"><div><span>索引版本</span><strong>{{ evaluationDetail.index_version_id || '—' }}</strong></div><div><span>外部调用</span><strong>{{ evaluationDetail.external_calls }}</strong></div><div><span>状态</span><strong :class="evaluationDetail.degraded ? 'text-amber' : 'text-green'">{{ evaluationDetail.degraded ? '降级' : '已验证' }}</strong></div></div><div v-if="evaluationDetail.retrieval" class="detail-section"><h3>检索统计</h3><p>Dense {{ evaluationDetail.retrieval.dense_count }} · BM25 {{ evaluationDetail.retrieval.keyword_count }} · RRF候选 {{ evaluationDetail.retrieval.fused_count }} · {{ evaluationDetail.retrieval.rerank === 'validated' ? 'Rerank 已完成' : 'Rerank 已降级' }}</p></div><div v-if="evaluationDetail.answer" class="detail-section"><h3>模型回答</h3><p class="detail-answer">{{ evaluationDetail.answer }}</p></div><div v-if="evaluationDetail.citations?.length" class="detail-section"><h3>引用证据（{{ evaluationDetail.citations.length }}）</h3><ul class="detail-list"><li v-for="citation in evaluationDetail.citations" :key="citation.chunk_id"><strong>{{ citation.citation }}</strong><span>{{ citation.heading_path || citation.source_path || '未命名证据' }}</span><small>{{ citation.source_path }} · 第 {{ citation.line_start || '—' }}–{{ citation.line_end || '—' }} 行</small></li></ul></div><div v-if="evaluationDetail.results?.length" class="detail-section"><h3>检索结果（{{ evaluationDetail.results.length }}）</h3><ul class="detail-list"><li v-for="result in evaluationDetail.results.slice(0, 5)" :key="result.chunk_id"><strong>{{ result.rerank_score?.toFixed(2) || '—' }}</strong><span>{{ result.metadata?.heading_path || '未命名章节' }}</span><small>{{ result.metadata?.source_path }} · 第 {{ result.metadata?.line_start || '—' }}–{{ result.metadata?.line_end || '—' }} 行</small></li></ul></div><div v-if="evaluationDetail.records?.length" class="detail-section"><h3>调用记录</h3><div class="record-list"><span v-for="record in evaluationDetail.records" :key="`${record.service}-${record.status_code}`"><strong>{{ record.service }}</strong><small>{{ record.status_code }} · {{ record.duration_ms || '—' }} ms · {{ record.usage?.total_tokens || '—' }} tokens</small></span></div></div></template></section></div>
        </section>
      </div>
    </main>
  </div>
</template>
