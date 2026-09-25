"""Resolve retrieved image elements to bounded, local visual evidence."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.settings import Settings
from multimodal_rag.infrastructure.visual_repository import (load_active_image_element,
    load_active_image_elements_for_title, load_image_elements_for_chunks)


_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_GENERIC_VISUAL_TERMS = {"原始", "截图", "示意", "图中", "依据", "中的", "分别", "多少",
                         "什么", "页面", "显示", "哪个", "哪些", "是否", "如何", "图片"}


def _corpus_root(source_file: Path) -> Path | None:
    for parent in (source_file.parent, *source_file.parents):
        if parent.name.lower() == "source":
            return parent
    return None


def _resolve(settings: Settings, source_path: str, image_ref: str) -> tuple[Path | None, str]:
    parsed = urlsplit(image_ref)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or "\\" in image_ref:
        return None, "external_or_invalid_reference"
    pure = PurePosixPath(image_ref)
    if any(part == ".." for part in pure.parts):
        return None, "unsafe_reference"
    source_file = (settings.project_root / source_path).resolve()
    if image_ref.startswith("/"):
        root = _corpus_root(source_file)
        if root is None:
            return None, "corpus_root_unknown"
        candidate = (root / image_ref.lstrip("/")).resolve()
    else:
        candidate = (source_file.parent / image_ref).resolve()
    allowed = (settings.project_root / "data" / "raw").resolve()
    if not candidate.is_relative_to(allowed):
        return None, "unsafe_reference"
    return candidate, "resolved"


def _signature_matches(path: Path, mime: str) -> bool:
    prefix = path.read_bytes()[:8]
    return ((mime == "image/png" and prefix.startswith(b"\x89PNG\r\n\x1a\n")) or
            (mime == "image/jpeg" and prefix.startswith(b"\xff\xd8\xff")))


def _resolve_rows(settings: Settings, rows: list[dict]) -> list[dict]:
    resolved_items: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        path, status = _resolve(settings, row["document_source_path"], row["image_ref"])
        item = {key: row[key] for key in (
            "chunk_id", "element_id", "document_id", "document_title", "heading_path", "image_ref")}
        item["source"] = row.get("source")
        item["raw_text"] = row.get("raw_text") or ""
        item["chunk_text"] = row.get("chunk_text") or ""
        item.update({"status": status, "vision_eligible": False})
        if path is not None:
            suffix = path.suffix.lower()
            mime = _MIME_BY_SUFFIX.get(suffix)
            if not path.is_file():
                item["status"] = "asset_missing"
            elif mime is None:
                item["status"] = "unsupported_image_format"
            elif path.stat().st_size > settings.api_max_image_bytes:
                item["status"] = "image_too_large"
            elif not _signature_matches(path, mime):
                item["status"] = "invalid_image_signature"
            else:
                relative = path.relative_to(settings.project_root).as_posix()
                if relative in seen:
                    continue
                content = path.read_bytes()
                seen.add(relative)
                item.update({"status": "ready", "vision_eligible": True,
                             "asset_path": relative, "mime_type": mime,
                             "byte_size": len(content),
                             "sha256": hashlib.sha256(content).hexdigest()})
        resolved_items.append(item)
    return resolved_items


def _tokens(value: str) -> set[str]:
    value = unicodedata.normalize("NFKC", value or "").lower()
    # Hyphens and slashes delimit filename/label words, so a question token
    # such as ``time`` can match ``detail-plans-time.png``.
    result = {token for token in re.findall(r"[a-z][a-z0-9_.]*|\d+(?:\.\d+)?", value)
              if len(token) >= 2}
    for run in re.findall(r"[\u4e00-\u9fff]+", value):
        result.update(run[index:index + 2] for index in range(len(run) - 1)
                      if run[index:index + 2] not in _GENERIC_VISUAL_TERMS)
    return result


def _query_relevance(query: str, title: str, item: dict) -> float:
    scoped_query = query.replace(f"《{title}》", " ").replace(title, " ")
    query_tokens = _tokens(scoped_query)
    raw = f"{item.get('raw_text', '')} {item.get('image_ref', '')} {' '.join(item.get('heading_path') or [])}"
    context = item.get("chunk_text", "")
    raw_tokens, context_tokens = _tokens(raw), _tokens(context)
    direct = query_tokens & raw_tokens
    contextual = query_tokens & context_tokens
    # Alt text and file names are stronger image-level signals than surrounding
    # section prose; numbers and Latin identifiers are especially selective.
    direct_score = sum(3.0 if re.fullmatch(r"[a-z0-9_.-]+", token) else 1.5
                       for token in direct)
    context_score = sum(1.0 if re.fullmatch(r"[a-z0-9_.-]+", token) else 0.35
                        for token in contextual)
    quoted = [next(part for part in groups if part) for groups in
              re.findall(r"“([^”]+)”|'([^']+)'|\"([^\"]+)\"", scoped_query)]
    raw_normalised = unicodedata.normalize("NFKC", raw).lower()
    exact_label_score = 12.0 * sum(
        unicodedata.normalize("NFKC", phrase).lower() in raw_normalised
        for phrase in quoted if len(phrase.strip()) >= 2)
    return round(direct_score + context_score + exact_label_score, 4)


def locate_visual_evidence_for_title(settings: Settings, title: str, query: str,
                                     *, max_images: int = 3) -> dict:
    if not title.strip() or len(title) > 200:
        raise AppError("invalid_document_title", "图片文档标题格式不正确", 422)
    rows = load_active_image_elements_for_title(settings, title)
    items = _resolve_rows(settings, rows)
    ready = [item for item in items if item["vision_eligible"]]
    for item in ready:
        item["visual_relevance_score"] = _query_relevance(query, title, item)
    ready.sort(key=lambda item: (-item["visual_relevance_score"],
                                 item.get("source", {}).get("line_start") or 0,
                                 item["element_id"]))
    # When one image has a strong, clearly separated image-level match, avoid
    # sending lower-ranked sibling screenshots as distractors.  Otherwise keep
    # up to three candidates for the VLM to compare.
    if (ready and ready[0]["visual_relevance_score"] >= 8
            and (len(ready) == 1 or ready[0]["visual_relevance_score"]
                 - ready[1]["visual_relevance_score"] >= 2.5)):
        selected = ready[:1]
    else:
        selected = ready[:max_images]
    diagnostics = [item for item in items if not item["vision_eligible"]]
    return {"document_title": title, "candidates": selected + diagnostics,
            "ready_count": len(selected), "candidate_count": len(selected) + len(diagnostics),
            "discovered_image_count": len(items), "max_images": max_images,
            "selection_strategy": "exact_title_then_local_visual_relevance"}


def locate_visual_evidence(settings: Settings, chunk_ids: list[str], *, max_images: int = 3) -> dict:
    if not chunk_ids or len(chunk_ids) > 20 or len(chunk_ids) != len(set(chunk_ids)):
        raise AppError("invalid_chunk_scope", "图片定位需要1到20个不重复的Chunk编号", 422)
    if not isinstance(max_images, int) or not 1 <= max_images <= 5:
        raise AppError("invalid_image_limit", "单次最多定位1到5张图片", 422)
    rows = load_image_elements_for_chunks(settings, chunk_ids)
    resolved_items = _resolve_rows(settings, rows)

    # Prefer coverage across retrieved Chunks before taking a second image from
    # one Chunk.  This prevents a multi-image section from exhausting the VLM
    # budget and hiding a relevant image in another highly ranked section.
    by_chunk = {chunk_id: [] for chunk_id in chunk_ids}
    for item in resolved_items:
        if item["vision_eligible"]:
            by_chunk.setdefault(item["chunk_id"], []).append(item)
    selected: list[dict] = []
    depth = 0
    while len(selected) < max_images and any(depth < len(values) for values in by_chunk.values()):
        for chunk_id in chunk_ids:
            values = by_chunk.get(chunk_id, [])
            if depth < len(values):
                selected.append(values[depth])
                if len(selected) >= max_images:
                    break
        depth += 1
    diagnostics = [item for item in resolved_items if not item["vision_eligible"]]
    items = selected + diagnostics
    return {"chunk_ids": chunk_ids, "candidates": items,
            "ready_count": len(selected), "candidate_count": len(items),
            "discovered_image_count": len(resolved_items), "max_images": max_images}


def read_visual_candidate(settings: Settings, candidate: dict) -> tuple[bytes, str]:
    """Revalidate a locator-produced asset immediately before a VLM request."""
    relative = candidate.get("asset_path")
    mime = candidate.get("mime_type")
    expected_hash = candidate.get("sha256")
    if not isinstance(relative, str) or mime not in _MIME_BY_SUFFIX.values():
        raise AppError("invalid_visual_candidate", "图片候选缺少有效的本地资源信息", 422)
    path = (settings.project_root / relative).resolve()
    allowed = (settings.project_root / "data" / "raw").resolve()
    if not path.is_relative_to(allowed) or not path.is_file():
        raise AppError("visual_asset_unavailable", "图片资源不存在或超出允许目录", 409)
    content = path.read_bytes()
    if len(content) > settings.api_max_image_bytes:
        raise AppError("image_too_large", "图片超过视觉请求大小限制", 413)
    if not _signature_matches(path, mime) or hashlib.sha256(content).hexdigest() != expected_hash:
        raise AppError("visual_asset_changed", "图片内容与定位时的哈希不一致", 409)
    return content, mime


def read_active_visual_element(settings: Settings, element_id: str) -> dict:
    row = load_active_image_element(settings, element_id)
    if row is None:
        raise AppError("visual_element_not_found", "当前激活知识版本中不存在该图片元素", 404)
    path, status = _resolve(settings, row["document_source_path"], row["image_ref"])
    if path is None or status != "resolved" or not path.is_file():
        raise AppError("visual_asset_unavailable", "图片元素对应的本地资源不可用", 409)
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None or path.stat().st_size > settings.api_max_image_bytes or not _signature_matches(path, mime):
        raise AppError("visual_asset_invalid", "图片格式、签名或大小不符合读取约束", 409)
    content = path.read_bytes()
    return {"content": content, "mime_type": mime,
            "sha256": hashlib.sha256(content).hexdigest(), "element_id": element_id}
