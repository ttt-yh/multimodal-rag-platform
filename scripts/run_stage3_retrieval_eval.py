"""在阶段3B draft索引上执行真实查询Embedding/Rerank评测。

检索本身直接读取指定索引产物，不依赖 active 指针；结果按 case 保存，便于复核
和离线重算 Recall@5、Precision@5、MRR。视觉题的指标只表示是否找到了图片所在
Chunk，不把文本Chunk当作图中答案。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.index_backends import build_chroma  # noqa: F401 (dependency gate only)
from multimodal_rag.infrastructure.model_adapters import EmbeddingAdapter, RerankAdapter
from multimodal_rag.infrastructure.retrieval import bm25_search, dense_search, hydrate_chunks, rrf_fuse
from multimodal_rag.infrastructure.settings import load_settings


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(case: dict, selected: list[dict]) -> dict:
    if case.get("answerable") is not True:
        return {"answerable": False, "retrieved_count": len(selected),
                "nonempty_retrieval": bool(selected)}
    expected = set(case.get("evidence_chunk_ids", []))
    result_ids = [row["chunk_id"] for row in selected[:5]]
    matched = expected.intersection(result_ids)
    first = next((index + 1 for index, item in enumerate(result_ids) if item in expected), None)
    return {"answerable": True, "relevant_count": len(expected), "matched_count": len(matched),
            "recall_at_5": len(matched) / len(expected) if expected else 0.0,
            "precision_at_5": len(matched) / len(result_ids) if result_ids else 0.0,
            "mrr": 1 / first if first else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段3C：draft索引真实检索评测")
    parser.add_argument("--dataset", default="evals/datasets/retrieval_stage3_eval_ready.jsonl")
    parser.add_argument("--index-report", default="evals/results/index-build/idx_38f91e95358bd534c5e046fa17e88bcd657e0f3432337e43.json")
    parser.add_argument("--confirm-live", action="store_true")
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error("检索评测会调用Embedding/Rerank，请提供 --confirm-live；未发起网络请求")

    settings = load_settings()
    root = settings.project_root
    cases = _rows(root / args.dataset)
    unreleased = [case.get("case_id") for case in cases
                  if case.get("gold_eligible") is not True or case.get("review_status") != "approved"]
    if unreleased:
        raise SystemExit("评测集包含未发布样本，拒绝生成正式指标：" + ", ".join(unreleased[:5]))
    build_report = json.loads((root / args.index_report).read_text(encoding="utf-8"))
    index_id = build_report["index_version_id"]
    artifact_root = (root / "indexes" / index_id).resolve()
    if not artifact_root.is_relative_to(root) or not artifact_root.is_dir():
        raise SystemExit("draft索引产物目录不存在或不在项目目录内")

    report_dir = root / "evals/results/retrieval-stage3" / index_id
    report_dir.mkdir(parents=True, exist_ok=True)
    details = []
    total_calls = 0
    for case in cases:
        budget = CallBudget(2 if case.get("answerable") is True else 1)
        records: list[dict] = []
        embedding_gateway = HttpGateway(settings, "embedding", budget=budget)
        try:
            embedding = EmbeddingAdapter(embedding_gateway).embed([case["question"]])
        finally:
            records.extend(embedding_gateway.records)
            embedding_gateway.close()

        dense = dense_search(artifact_root, index_id, embedding["vectors"][0], top_k=10)
        keyword = bm25_search(artifact_root, index_id, case["question"], top_k=10)
        fused = rrf_fuse(dense, keyword, top_k=10)
        hydrated = hydrate_chunks(artifact_root, index_id, [row["chunk_id"] for row in fused])
        candidates = [{**row, "text": hydrated[row["chunk_id"]]["text"],
                       "metadata": hydrated[row["chunk_id"]]["metadata"]}
                      for row in fused if row["chunk_id"] in hydrated]
        selected = candidates[:5]
        rerank_status = "skipped_unanswerable"
        if case.get("answerable") is True and candidates:
            rerank_gateway = HttpGateway(settings, "rerank", budget=budget)
            try:
                reranked = RerankAdapter(rerank_gateway).rerank(
                    case["question"], [row["text"] for row in candidates], min(5, len(candidates)))
                selected = [{**candidates[item["index"]], "rerank_score": item["score"]}
                            for item in reranked["results"]]
                rerank_status = "validated"
            except AppError as exc:
                rerank_status = f"fallback:{exc.code}"
                selected = [{**row, "rerank_score": None} for row in selected]
            finally:
                records.extend(rerank_gateway.records)
                rerank_gateway.close()

        result = {"case_id": case["case_id"], "query": case["question"],
                  "index_version_id": index_id,
                  "retrieval": {"dense_count": len(dense), "keyword_count": len(keyword),
                                 "fused_count": len(candidates), "rerank": rerank_status},
                  "results": [{"chunk_id": row["chunk_id"], "rrf_score": row.get("rrf_score"),
                               "rerank_score": row.get("rerank_score"), "metadata": row["metadata"],
                               "text": row["text"]} for row in selected],
                  "metrics": _metrics(case, selected), "external_calls": len(records),
                  "records": records, "created_at": datetime.now(timezone.utc).isoformat()}
        (report_dir / f"{case['case_id']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        details.append(result)
        total_calls += len(records)

    answerable = [row["metrics"] for row in details if row["metrics"].get("answerable")]
    text_answerable = [row["metrics"] for row, case in zip(details, cases)
                       if case.get("answerable") is True and case.get("type") != "visual"]
    visual_answerable = [row["metrics"] for row, case in zip(details, cases)
                         if case.get("answerable") is True and case.get("type") == "visual"]

    def aggregate(items):
        if not items:
            return None
        return {key: round(sum(item[key] for item in items) / len(items), 4)
                for key in ("recall_at_5", "precision_at_5", "mrr")}

    summary = {"status": "completed", "index_version_id": index_id,
               "dataset": str(Path(args.dataset).as_posix()), "gold_eligible": True,
               "sample_count": len(details), "answerable_count": len(answerable),
               "text_answerable_count": len(text_answerable), "visual_answerable_count": len(visual_answerable),
               "unanswerable_count": len(details) - len(answerable),
               "metrics_all_evidence": aggregate(answerable),
               "metrics_text_evidence": aggregate(text_answerable),
               "metrics_visual_location": aggregate(visual_answerable),
               "external_api_calls": total_calls, "reports_dir": str(report_dir),
               "created_at": datetime.now(timezone.utc).isoformat()}
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
