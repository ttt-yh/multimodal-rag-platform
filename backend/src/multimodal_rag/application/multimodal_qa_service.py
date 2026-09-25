"""Route image-dependent questions through retrieved local evidence and a VLM."""
from __future__ import annotations

import re

from multimodal_rag.application.context_builder import build_context, expand_with_adjacent_chunks
from multimodal_rag.application.qa_service import answer as answer_text
from multimodal_rag.application.retrieval_service import retrieve
from multimodal_rag.application.visual_evidence import (locate_visual_evidence,
    locate_visual_evidence_for_title, read_visual_candidate)
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.model_adapters import VisionAdapter
from multimodal_rag.infrastructure.settings import Settings


_VISUAL_CUES = ("图片", "图中", "截图", "示意图", "架构图", "流量图", "热力图",
                "原图", "图表", "曲线", "图例", "看图")
_TITLE = re.compile(r"《([^》]{1,200})》")


def requires_visual(query: str) -> bool:
    return any(cue in query for cue in _VISUAL_CUES)


def _task_check_text(query: str) -> str:
    checks = []
    if any(cue in query for cue in ("最大", "最小", "最高", "最低")):
        checks.append(
            "这是比较型问题：先把所有候选数值换算为同一单位并逐一比较，最终只给出与比较结果一致的对象和数值。")
    if any(cue in query for cue in ("方向", "分布", "走势", "形态")):
        checks.append(
            "描述视觉分布时，同时核对方向与形态（例如横向、纵向、斜带或阶梯），不要遗漏图中明显特征。")
    return ("\n本题专项核对：" + "".join(checks)) if checks else ""


def answer_multimodal(settings: Settings, query: str, *, max_context_chars: int = 12000,
                      max_images: int = 3, max_requests: int = 3) -> dict:
    if not requires_visual(query):
        return {**answer_text(settings, query, max_context_chars=max_context_chars,
                              max_requests=max_requests), "answer_mode": "text"}
    if not isinstance(max_images, int) or not 1 <= max_images <= 3:
        raise AppError("invalid_image_limit", "图文问答每轮最多读取1到3张图片", 422)
    if max_requests != 3:
        raise AppError("invalid_call_budget", "图文问答预算必须为3次：Embedding、Rerank和Vision", 422)

    retrieval = retrieve(settings, query, max_requests=2)
    expansion = expand_with_adjacent_chunks(settings, retrieval["results"], radius=1, max_primary=3)
    context_results = expansion["results"]
    context = build_context(query, context_results, max_chars=max_context_chars)
    chunk_ids = [item.get("chunk_id") for item in context_results if item.get("chunk_id")]
    title_match = _TITLE.search(query)
    if title_match:
        located = locate_visual_evidence_for_title(
            settings, title_match.group(1).strip(), query, max_images=max_images)
    else:
        located = locate_visual_evidence(
            settings, list(dict.fromkeys(chunk_ids)), max_images=max_images)
    ready = [item for item in located["candidates"] if item.get("vision_eligible")][:max_images]
    if not ready:
        return {
            "query": query, "answer": "已定位到相关文字章节，但没有找到可安全读取的原始图片，无法确认图中信息。",
            "answer_mode": "visual_refusal", "citations": context["citations"],
            "visual_evidence": located["candidates"], "retrieval": retrieval["retrieval"],
            "index_version_id": retrieval["index_version_id"], "degraded": retrieval["degraded"],
            "context_expansion": {"anchor_count": expansion["anchor_count"],
                                  "adjacent_count": expansion["adjacent_count"]},
            "external_calls": retrieval["external_calls"],
            "records": retrieval["records"],
        }

    image_lines = []
    images = []
    for index, item in enumerate(ready, start=1):
        images.append(read_visual_candidate(settings, item))
        source = item.get("source") or {}
        location = f"{source.get('source_path', '')}:{source.get('line_start', '')}"
        image_lines.append(f"[图{index}] {item.get('document_title', '')}；来源 {location}；原始引用 {item.get('image_ref', '')}")
        item["image_citation"] = f"[图{index}]"
    task_check_text = _task_check_text(query)
    prompt = (
        context["prompt"] + "\n\n图片清单：\n" + "\n".join(image_lines) +
        "\n\n请联合阅读文字证据与按清单顺序提供的原图。图片结论使用[图1]、[图2]标注，"
        "文字结论继续使用[1]、[2]标注。若多张图片内容不同，要明确区分，不能自行选择更符合问题预期的一张；"
        "看不清的数值、对象或关系必须说明无法确认。凡是依据图片得到的结论都必须使用[图N]引用，"
        "不能只写文字证据编号[1]；回答中至少出现一个有效的[图N]。涉及多个数值比较时，先统一单位再判断最大值或最小值，"
        "并检查结论是否与列出的数值一致。" + task_check_text +
        "\n请在200字以内直接回答问题，不复述任务要求。"
    )
    gateway = HttpGateway(settings, "vision", budget=CallBudget(1))
    try:
        response = VisionAdapter(gateway).describe_many(prompt, images)
    finally:
        vision_records = list(gateway.records)
        gateway.close()
    return {
        "query": query, "answer": response["text"], "answer_mode": "vision",
        "citations": context["citations"], "visual_evidence": ready,
        "retrieval": retrieval["retrieval"], "index_version_id": retrieval["index_version_id"],
        "degraded": retrieval["degraded"],
        "context_expansion": {"anchor_count": expansion["anchor_count"],
                              "adjacent_count": expansion["adjacent_count"]},
        "external_calls": retrieval["external_calls"] + len(vision_records),
        "records": retrieval["records"] + vision_records,
    }
