"""结构感知的第一版 Chunk 切分器。

先按标题路径分组，再仅对超长章节递归处理。长度明确使用字符数，
避免在没有固定 Tokenizer 时把字符数误称为 Token 数。
"""
from __future__ import annotations

import hashlib

from multimodal_rag.core.models import Chunk, Element


def _split_text(text: str, limit: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind("。", start, end),
                           text.rfind("；", start, end), text.rfind(" ", start, end))
            if boundary > start + limit // 2:
                end = boundary + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [piece for piece in pieces if piece]


def split_elements(elements: list[Element], processing_version_id: str,
                   *, max_characters: int = 1200, overlap_characters: int = 160,
                   min_characters: int = 100) -> list[Chunk]:
    if not processing_version_id or max_characters <= 0 or not 0 <= overlap_characters < max_characters:
        raise ValueError("切分参数不合法")
    groups: list[tuple[tuple[str, ...], list[Element]]] = []
    for element in elements:
        path = tuple(element.heading_path)
        if not groups or groups[-1][0] != path:
            groups.append((path, []))
        groups[-1][1].append(element)
    chunks: list[Chunk] = []
    for path, group in groups:
        pending_text: list[str] = []
        pending_elements: list[Element] = []

        def flush() -> None:
            if not pending_text:
                return
            text = "\n".join(pending_text).strip()
            if not text:
                pending_text.clear(); pending_elements.clear(); return
            ordinal = len(chunks)
            digest = hashlib.sha256(f"{processing_version_id}:{ordinal}:{text}".encode()).hexdigest()[:32]
            chunks.append(Chunk(
                chunk_id="chk_" + digest,
                document_id=pending_elements[0].document_id,
                version_id=pending_elements[0].version_id,
                processing_version_id=processing_version_id,
                ordinal=ordinal,
                text=text,
                heading_path=list(path),
                element_ids=[e.element_id for e in pending_elements],
                source_locations=[e.source for e in pending_elements],
                length_unit="characters",
                estimated_length=len(text),
            ))
            pending_text.clear(); pending_elements.clear()

        for element in group:
            raw = element.raw_text.strip()
            if not raw:
                continue
            for piece in _split_text(raw, max_characters, overlap_characters):
                if pending_text and len("\n".join(pending_text)) + len(piece) + 1 > max_characters:
                    flush()
                # A small adjacent fragment stays with the same section unless adding it
                # would exceed the hard budget.
                pending_text.append(piece)
                pending_elements.append(element)
                if len("\n".join(pending_text)) >= max_characters:
                    flush()
        if (pending_text and len("\n".join(pending_text)) < min_characters and chunks
                and chunks[-1].heading_path == list(path)
                and len(chunks[-1].text) + 1 + len("\n".join(pending_text)) <= max_characters):
            previous = chunks.pop()
            tail_elements = list(pending_elements)
            pending_text.insert(0, previous.text)
            # Preserve both the previous chunk and the short tail's source mapping.
            ids = previous.element_ids + [e.element_id for e in tail_elements]
            locations = previous.source_locations + [e.source for e in tail_elements]
            text = "\n".join(pending_text).strip()
            digest = hashlib.sha256(f"{processing_version_id}:{previous.ordinal}:{text}".encode()).hexdigest()[:32]
            chunks.append(previous.model_copy(update={"chunk_id": "chk_" + digest, "text": text,
                "element_ids": ids, "source_locations": locations, "estimated_length": len(text)}))
            pending_text.clear(); pending_elements.clear()
        flush()
    return [chunk.model_copy(update={"ordinal": i}) for i, chunk in enumerate(chunks)]
