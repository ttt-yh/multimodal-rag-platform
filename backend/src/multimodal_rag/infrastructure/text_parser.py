"""最小离线结构解析器，不是最终 Chunk 切分器，也不执行 OCR/VLM。

markdown-it token.map 为 0-based、右端不含；输出行号为 1-based、两端包含。
每个元素 raw_text 始终直接截取原文，避免解析后的格式变化破坏来源追溯。
"""
import hashlib

from markdown_it import MarkdownIt

from multimodal_rag.core.errors import AppError
from multimodal_rag.core.models import Document, DocumentVersion, Element, SourceLocation

PARSER_VERSION = "local-text-preview-v1"


def parse_text(document: Document, version: DocumentVersion, text: str) -> tuple[list[Element], list[str]]:
    original_lines = text.splitlines(keepends=True)
    elements: list[Element] = []
    warnings = ["0A 仅进行结构预览，尚未切分、向量化或入库；图片仅保留引用。"]
    headings: list[tuple[int, str]] = []

    def add(kind: str, start: int, end: int, image_ref: str | None = None):
        raw = "".join(original_lines[start:end])
        order = len(elements)
        identity = f"{version.version_id}:{PARSER_VERSION}:{order}:{kind}:{start}:{end}:{image_ref}"
        elements.append(Element(
            element_id="el_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            document_id=document.document_id, version_id=version.version_id,
            kind=kind, order=order, raw_text=raw,
            heading_path=[title for _, title in headings],
            source=SourceLocation(source_path=document.source_path, kind="text", precision="line",
                                  line_start=start + 1, line_end=end), image_ref=image_ref,
        ))

    if not text.strip():
        raise AppError("empty_document", "原文没有可解析内容", 422)
    if document.format == "txt":
        # TXT 仅按空行形成段落，不假定任意数字编号必然是标题。
        start = None
        for i, line in enumerate(original_lines + [""]):
            if line.strip() and start is None:
                start = i
            if not line.strip() and start is not None:
                add("text", start, i)
                start = None
    else:
        # 屏蔽 YAML front matter 但保留同样的行数，不改变后续行号。
        parse_lines = original_lines.copy()
        if parse_lines:
            parse_lines[0] = parse_lines[0].lstrip("\ufeff")
        if parse_lines and parse_lines[0].strip() == "---":
            end = next((i for i in range(1, len(parse_lines)) if parse_lines[i].strip() == "---"), None)
            if end is not None:
                parse_lines[:end + 1] = ["\n"] * (end + 1)
                warnings.append("已跳过 YAML 文档头，正文行号保持不变。")
        tokens = MarkdownIt("commonmark", {"html": True}).enable("table").parse("".join(parse_lines))
        table_until = -1
        for i, token in enumerate(tokens):
            if token.map is None:
                continue
            start, end = token.map
            if start < table_until:
                continue
            if token.type == "heading_open":
                level = int(token.tag[1:])
                title = tokens[i + 1].content
                headings[:] = [(lv, t) for lv, t in headings if lv < level]
                headings.append((level, title))
                add("heading", start, end)
            elif token.type == "table_open":
                add("table", start, end)
                table_until = end
            elif token.type in {"fence", "code_block"}:
                add("code", start, end)
            elif token.type == "paragraph_open":
                inline = tokens[i + 1]
                children = inline.children or []
                images = [c for c in children if c.type == "image"]
                if not images or any(c.type == "text" and c.content.strip() for c in children):
                    add("text", start, end)
                for img in images:
                    add("image", start, end, img.attrGet("src"))
            elif token.type == "html_block":
                add("text", start, end)
                warnings.append("存在 HTML/模板内容：保留原文，不执行或解释其中资源。")
        warnings = list(dict.fromkeys(warnings))
    if not elements:
        raise AppError("no_elements", "未识别出正文元素，请检查文档内容", 422)
    return elements, warnings


def verify_sources(document: Document, version: DocumentVersion, text: str, elements: list[Element]):
    """独立回查每个元素：版本、顺序、行范围与原文必须一致。"""
    lines = text.splitlines(keepends=True)
    ids = set()
    for order, element in enumerate(elements):
        source = element.source
        valid = (element.document_id == document.document_id and
                 element.version_id == version.version_id and element.order == order and
                 element.element_id not in ids and source.source_path == document.source_path and
                 source.kind == "text" and source.line_start is not None and source.line_end is not None)
        if not valid or not (1 <= source.line_start <= source.line_end <= len(lines)):
            raise AppError("invalid_source_mapping", "元素来源关联校验失败", 500)
        if "".join(lines[source.line_start - 1:source.line_end]) != element.raw_text:
            raise AppError("invalid_source_mapping", "元素原文与来源行号不一致", 500)
        ids.add(element.element_id)

