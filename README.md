# Enterprise Multimodal RAG Platform

面向企业研发资料的多模态 RAG 知识库。项目围绕“资料可治理、检索可复现、回答可追溯、效果可评测”实现了从文档入库到图文问答的完整链路；异常诊断 Agent 不在本仓库范围内，将作为独立项目开发。

## 当前状态

项目已进入 **v1 release candidate**：

- 支持 Markdown/TXT 白名单入库、版本与哈希管理、结构感知切分、人工质量审核和索引发布门禁；
- 使用 Chroma Dense 检索与本地持久化 BM25 进行混合召回，经 RRF 融合和在线 Rerank 后生成带来源引用的回答；
- 能从命中 Chunk 关联图片原件，按需调用 VLM 完成图文联合回答，并返回可核验的文字及图片证据；
- 提供 FastAPI 后端与 Vue 3 工作台，覆盖知识问答、资料导入、审核、索引构建/激活和评测结果查看；
- PostgreSQL 保存文档、处理版本、结构元素、Chunk、审核记录、索引版本和任务状态，索引产物与业务状态分离。

当前唯一 active 索引为 `idx_a20fa855d568a759b7fac6d3b9823f7ae89c58db900ff477`，覆盖 69 个已审核处理版本，Chunk、Dense 向量和 BM25 条目均为 1,078。

## 评测结果

文本检索 Gold 包含 8 条可回答题和 3 条无答案/澄清题：Recall@5 为 0.9375、Precision@5 为 0.20、MRR 为 1.00；问答引用有效率、完整证据覆盖率和无答案拒答率均为 1.00。该小规模数据集用于工程回归，不代表通用领域效果。

视觉 Gold v2.4 包含 30 条人工逐图审核样本，固定划分为 20 条 dev 和 10 条 test。方案冻结后仅执行一次 test：图片证据组召回率与图片引用有效率均为 1.00，排除图片误选率为 0，必要要点覆盖率为 0.9083，严格确定性通过率为 0.70；开发级 LLM Judge 语义通过率为 1.00。两套指标必须同时披露，test 样本量仅 10 条，LLM Judge 结果也不等同于正式人工正确率。

详细状态与指标口径见 [阶段6A收口说明](docs/6A_项目状态与最终评测收口.md)，浏览器和公开化检查见 [阶段6B竣工验收报告](docs/6B_正式竣工验收报告.md)，运行问题见 [部署运行与故障排查](docs/部署运行与故障排查.md)。本机原始报告保存在 `evals/results/`，该目录默认不提交 Git。

## 技术架构

```text
Vue 3 + TypeScript
        ↓
FastAPI 应用与质量门禁
        ↓
入库：白名单 → 解析 → 结构元素 → Chunk → 人工审核 → 版本化发布
检索：Query Embedding → Dense + BM25 → RRF → Rerank → 邻接补全
问答：文本证据 / 图片定位与选图 → LLM 或 VLM → 文字与图片引用
        ↓
PostgreSQL（业务元数据） + Chroma（向量） + 持久化 BM25（关键词）
```

模型通过统一 HTTP 网关调用阿里云百炼北京地域服务，密钥和代理只从本地 `.env` 读取，不进入代码、日志或 Git。

## 数据边界

当前正式入库 worker 支持 Markdown/TXT 及文档中关联的本地图片。PDF、MinerU、OCR 与 VLM 解析已完成阶段性验证，但尚未接入正式版本化入库 worker，因此不能将当前版本描述为“已支持任意 PDF 自动入库”。数据池包含 1,109 份中文公开文档；active 索引采用经过审核的 69 个处理版本，并未把全部数据无差别入库。

原始资料、派生数据、索引产物、API 密钥和本地评测报告默认不提交 Git。评测问题与参考答案不会进入知识索引，dev/test 用于不同阶段，冻结后的 test 不重复用于调参。

## 本地运行

后端要求 Python 3.12、PostgreSQL 16，并已在当前机器使用 `uv.lock` 固定依赖：

```powershell
# 在项目根目录执行
uv sync --locked --extra test --extra index --python 3.12
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -X utf8 -m multimodal_rag.cli serve
```

后端地址为 `http://127.0.0.1:8010`，OpenAPI 页面为 `http://127.0.0.1:8010/docs`。

前端要求 Node.js 20 和 pnpm 10：

```powershell
Set-Location frontend
corepack prepare pnpm@10.12.4 --activate
pnpm.cmd install
pnpm.cmd dev
```

前端地址为 `http://127.0.0.1:5173`，开发服务器会将 `/api` 和 `/health` 代理到后端 8010 端口。

## 验证

```powershell
# 在项目根目录执行
.\.venv\Scripts\python.exe -X utf8 -m pytest backend/tests -q
Set-Location frontend
pnpm.cmd build
```

真实 Embedding、Rerank、LLM 和 VLM 命令均带显式确认与请求预算，禁止在不确认成本的情况下自动运行。

## 许可证

项目自有代码采用 [MIT License](LICENSE)。基于 TiDB 中文文档形成的评测数据保留 `CC-BY-SA-3.0` 的署名和相同方式共享要求，详细边界见 [评测数据许可](evals/datasets/LICENSE.md) 与 [第三方声明](THIRD_PARTY_NOTICES.md)。MIT 许可证不覆盖第三方文档、图片、数据集、商标或外部服务。

## 竣工状态

v1 核心功能和工程验收已经闭环：后端 234 项测试通过，Vue 类型检查与生产构建通过，无模型浏览器端到端验收通过，公开化扫描未发现疑似真实凭证。当前只剩初始化独立 Git 仓库、确认首次提交清单及推送 GitHub 等发布动作。PDF/OCR 正式入库、扩大人工 Gold、权限系统和异常诊断 Agent 属于后续增强，不作为本版本竣工阻塞项。
