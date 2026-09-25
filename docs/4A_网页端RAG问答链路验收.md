# 阶段4A：网页端 RAG 问答链路验收

## 1. 验收目标

阶段4A不是重新评测检索算法，而是验证用户从浏览器进入系统后，能够完成一条真实的 RAG 问答路径：打开 Vue 工作台、读取当前激活索引、提交问题、看到带引用的回答，并能够查看检索轨迹和历史运行报告。

## 2. 页面与接口对应关系

| 用户操作 | 前端页面 | 后端接口 | 验收重点 |
| --- | --- | --- | --- |
| 打开工作台 | 问答工作台 | `GET /health/live` | 页面可加载，服务状态可识别 |
| 查看知识资产 | 知识库 | `GET /api/v1/knowledge` | 能看到 PostgreSQL 返回的文档和激活索引 |
| 查看联调结果 | 评测记录 | `GET /api/v1/evaluations`、`GET /api/v1/evaluations/{run_id}` | 能列出报告并打开详情 |
| 提交问题 | 问答工作台 | `POST /api/v1/qa` | 返回回答、引用、检索统计和调用记录 |

问答接口内部仍然执行“查询向量生成 → Dense 与 BM25 混合检索 → RRF 融合 → Rerank → Context Builder → Chat 生成 → 引用回填”。浏览器只负责展示结果，不在前端重复实现检索逻辑。

## 3. 可重复的验收方式

先分别启动后端和前端：

```powershell
# 在项目根目录执行
.\.venv\Scripts\python.exe -X utf8 -m multimodal_rag.cli serve
```

```powershell
Set-Location frontend
pnpm.cmd dev -- --host 127.0.0.1
```

再运行 Playwright 验收脚本：

```powershell
Set-Location ..
& $env:WEBAPP_TEST_PYTHON scripts/verify_frontend_stage4a.py
```

脚本会将截图和结构化结果保存到 `evals/results/stage4-web/`，其中 `frontend_stage4a_report.json` 是验收摘要，`qa-workbench.png` 用于回看页面状态。

## 4. 本次真实验收结果

本次使用一条已在阶段4A离线问答中通过的问题进行网页提交：

> 根据《TiDB Dashboard 监控页面》，关于“Write Traffic”有哪些主要说明或操作要求？

结果如下：

- 页面加载：通过；
- 知识库页读取当前激活索引：通过；
- 评测记录列表及详情弹窗：通过；
- 问答提交：通过；
- 回答引用展示：通过；
- 检索轨迹展示：通过，页面显示 Dense、BM25、Rerank 和引用数量。

这次验证证明前端已经能够承载真实 RAG 问答链路，但它只是一条网页联调样例，不替代阶段3C的离线 Recall@5、Precision@5 和 MRR 评测，也不代表前端已经接入文档上传或异常诊断页面。

## 5. 问题与修正记录

第一次自动化验收使用 `127.0.0.1:5173` 时，Windows 本机的 Vite 进程只监听了 IPv6 回环地址，浏览器出现连接拒绝。验收脚本改用 `localhost:5173` 后恢复正常；后端 API 仍由 Vite 代理到 `127.0.0.1:8010`。

第一次测试提交了索引中没有充分覆盖的问题，后端返回“响应不满足服务契约”，前端正确进入演示模式。这不是前端吞错，而是 Chat 模型响应未满足严格契约时的安全降级。随后改用已通过离线问答验证、且当前索引确实覆盖的问题，验证了真实回答和引用路径。

## 6. 当前边界与下一步

当前前端已经具备问答、知识库只读查看和评测记录查看能力。资料上传按钮仍为占位，异常诊断工作流和 Human-in-the-Loop 审核尚未接入网页。下一步应优先把真实导入任务状态接入知识库页面，再为异常诊断增加独立的任务输入、日志查看、人工确认和经验回流界面。
