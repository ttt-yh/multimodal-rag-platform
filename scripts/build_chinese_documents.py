"""Render complete source documents into two split-preserving PDF anthologies.

These are project-rendered format variants, NOT official PDFs or new knowledge.
No QA answers are rendered. Markdown source and source-line maps are retained.
"""
from __future__ import annotations

from collections import defaultdict
from html import escape
import json
from pathlib import Path
import re

from markdown_it import MarkdownIt
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, Flowable

from download_datasets import ROOT, hashes, write_json
from download_chinese import REVISION
from prepare_datasets import jsonl, rel
from prepare_chinese import SOURCE, clean
from build_chinese_cases import rows

FONT = Path("C:/Windows/Fonts/msyh.ttc")
WIDTH = A4[0] - 84


class Marker(Flowable):
    def __init__(self, key, position):
        super().__init__()
        self.key, self.position = key, position
        self.width = self.height = 0

    def draw(self):
        pass


class MappedDocument(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locations = defaultdict(dict)

    def afterFlowable(self, flowable):
        if isinstance(flowable, Marker):
            self.locations[flowable.key][flowable.position] = self.page


def plain_inline(token):
    if not token.children:
        return clean(token.content)
    return "".join("\n" if c.type in {"softbreak", "hardbreak"} else c.content
                   for c in token.children if c.type in {"text", "code_inline", "softbreak", "hardbreak"})


def main():
    if not FONT.is_file():
        raise RuntimeError("Set FONT to an installed Chinese TrueType font before rendering.")
    pdfmetrics.registerFont(TTFont("CJK", str(FONT), subfontIndex=0))
    styles = {"body": ParagraphStyle("body", fontName="CJK", fontSize=9, leading=15, wordWrap="CJK", spaceAfter=7),
              "small": ParagraphStyle("small", fontName="CJK", fontSize=7, leading=11, wordWrap="CJK", spaceAfter=5),
              "heading": ParagraphStyle("heading", fontName="CJK", fontSize=13, leading=19, wordWrap="CJK", spaceBefore=10, spaceAfter=8, keepWithNext=True),
              "title": ParagraphStyle("title", fontName="CJK", fontSize=20, leading=28, wordWrap="CJK", spaceAfter=15),
              "code": ParagraphStyle("code", fontName="CJK", fontSize=7, leading=11, wordWrap="CJK", backColor=colors.HexColor("#f0f3f5"), spaceAfter=2)}
    def para(text, style="body"):
        safe = escape(text).replace("\n", "<br/>")
        if style == "code":
            safe = safe.replace(" ", "&#160;")
        return Paragraph(safe, styles[style])
    docs = rows("data/manifests/ingestion_chinese_md.jsonl")
    qa = rows("data/annotations/tidb_zh/qa_candidates.jsonl")
    sections = rows("data/annotations/tidb_zh/sections.jsonl")
    assets = rows("data/annotations/tidb_zh/assets.jsonl")
    selected_ids = {e["document_id"] for q in qa if q["type"] in {"visual", "table"} for e in q["evidence"]}
    core = {"overview.md", "tidb-architecture.md", "hardware-and-software-requirements.md", "production-deployment-using-tiup.md", "maintain-tidb-using-tiup.md", "troubleshoot-tidb-cluster.md"}
    selected = [d for d in docs if d["document_id"] in selected_ids or d["source_path"] in core]
    by_asset = {(a["source_path"], a["target"]): a for a in assets}
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    out = ROOT / "output/pdf"
    out.mkdir(parents=True, exist_ok=True)
    manifest, page_maps, warnings, pdf_summaries = [], [], [], []
    for split in ("dev", "test"):
        chosen = [d for d in selected if d["split"] == split]
        target = out / f"tidb_zh_85_{split}.pdf"
        story = [para(f"TiDB 中文技术文档专题册 | {split}", "title"),
                 para("项目生成的格式测试资料，不是官方 PDF 发布物。", "heading"),
                 para(f"来源：PingCAP / docs-cn / release-8.5\n固定提交：{REVISION}\n许可：CC BY-SA 3.0；归属原文作者及贡献者。\n本册包含 {len(chosen)} 篇完整 Markdown 源文档的重排版。图片可能展示历史版本界面，须以原文适用条件为准。"),
                 para("变更说明：将 Markdown 重排为分页文本、表格和图片；移除导航标记与 HTML 包装，保留代码内容。无法渲染的资源会明确标记并记录，不填造图像。不同格式不得作为独立知识重复入库。\n精确原文及链接见同目录来源清单和 data/raw/tidb_zh/source。"),
                 para("来源许可：https://creativecommons.org/licenses/by-sa/3.0/", "small")]
        block_meta = {}
        for d in chosen:
            story += [PageBreak(), Marker(d["document_id"], "start"), para(d["title"], "title"), para(d["source_url"], "small")]
            text = (ROOT / d["path"]).read_text(encoding="utf-8")
            # Blank rather than remove frontmatter to preserve original line numbers.
            text = re.sub(r"\A---\r?\n.*?\r?\n---[^\n]*(?:\n|$)", lambda m: "\n" * m.group().count("\n"), text, count=1, flags=re.S)
            text = re.sub(r"(?m)^[ \t]*\{\{[<%].*?[>%]\}\}[^\S\n]*$", "", text)
            tokens = md.parse(text)
            i, bullet = 0, False
            while i < len(tokens):
                tok = tokens[i]
                start_i = i
                flows = []
                mapping = tok.map
                if tok.type == "table_open":
                    table_rows, current = [], []
                    i += 1
                    while tokens[i].type != "table_close":
                        t = tokens[i]
                        if t.type == "tr_open": current = []
                        elif t.type == "inline": current.append(para(plain_inline(t), "small"))
                        elif t.type == "tr_close": table_rows.append(current)
                        i += 1
                    columns = max(map(len, table_rows))
                    for row in table_rows:
                        row.extend([""] * (columns - len(row)))
                    table = Table(table_rows, colWidths=[WIDTH / columns] * columns, repeatRows=1, splitByRow=1, splitInRow=1)
                    table.setStyle(TableStyle([("GRID", (0,0), (-1,-1), .3, colors.HexColor("#bdcbd2")), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8f0f4")), ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING",(0,0),(-1,-1),5), ("RIGHTPADDING",(0,0),(-1,-1),5)]))
                    flows = [table, Spacer(1, 8)]
                elif tok.type in {"fence", "code_block"}:
                    flows = [para(line.expandtabs(4) or " ", "code") for line in tok.content.splitlines()]
                elif tok.type == "inline":
                    mapping = tok.map
                    preceding = tokens[i-1].type if i else ""
                    if preceding == "heading_open":
                        flows.append(para(plain_inline(tok), "heading"))
                    else:
                        words = plain_inline(tok)
                        if words.strip(): flows.append(para(("• " if bullet else "") + words))
                        bullet = False
                    for c in tok.children or []:
                        if c.type != "image": continue
                        a = by_asset.get((d["source_path"], c.attrGet("src")))
                        if a and a["status"] == "available" and a["format"] in {"png", "jpg", "jpeg", "gif", "webp"}:
                            image_path = ROOT / a["asset_path"]
                            with PILImage.open(image_path) as im: w, h = im.size
                            scale = min(WIDTH / w, 430 / h, 1)
                            flows += [Image(str(image_path), width=w*scale, height=h*scale), para(c.content or "原文插图", "small")]
                        else:
                            warnings.append({"document_id": d["document_id"], "image": c.attrGet("src"), "reason": a["status"] if a else "unsupported_or_unmapped"})
                            flows.append(para(f"[此图请查看原始资源：{c.attrGet('src')}]", "small"))
                elif tok.type == "list_item_open": bullet = True
                elif tok.type == "html_block":
                    value = clean(tok.content)
                    if value: flows.append(para(value))
                    if "<img" in tok.content:
                        warnings.append({"document_id":d["document_id"], "reason":"html_image_preserved_in_md_not_pdf"})
                if flows:
                    key = f"{d['document_id']}:{start_i}"
                    block_meta[key] = {"document_id":d["document_id"], "source_path":d["source_path"], "line_start": (mapping[0]+1 if mapping else 1), "line_end": (mapping[1] if mapping else 1)}
                    story += [Marker(key, "start"), *flows, Marker(key, "end")]
                i += 1
            story.append(Marker(d["document_id"], "end"))
        def footer(canvas, doc):
            canvas.setFont("CJK", 7)
            canvas.setFillColor(colors.HexColor("#526675"))
            canvas.drawString(42, 25, f"PingCAP docs-cn | {REVISION[:12]} | CC BY-SA 3.0 | {split}")
            canvas.drawRightString(A4[0]-42, 25, str(doc.page))
        pdf = MappedDocument(str(target), pagesize=A4, leftMargin=42, rightMargin=42, topMargin=40, bottomMargin=42,
                             title=f"TiDB Chinese corpus {split}", author="PingCAP contributors; project format conversion", invariant=1)
        pdf.build(story, onFirstPage=footer, onLaterPages=footer)
        page_count = len(PdfReader(target).pages)
        sha = hashes(target)[0]
        pdf_summaries.append({"path":rel(target), "pages":page_count, "documents":len(chosen), "split":split, "sha256":sha})
        for d in chosen:
            span = pdf.locations[d["document_id"]]
            manifest.append({**d, "format":"pdf", "path":rel(target), "sha256":sha, "source_sha256":d["sha256"], "page_start":span["start"], "page_end":span["end"], "derived_from":d["path"], "variant_policy":"replace_md_source_family_never_add_duplicate", "rendered_by_project":True})
        for s in sections:
            if s["document_id"] not in {d["document_id"] for d in chosen}: continue
            overlaps = [pdf.locations[k] for k,b in block_meta.items() if b["document_id"]==s["document_id"] and b["line_end"] >= s["line_start"] and b["line_start"] <= s["line_end"]]
            if overlaps:
                page_maps.append({"evidence_id":s["evidence_id"], "pdf_path":rel(target), "page_start":min(x["start"] for x in overlaps), "page_end":max(x["end"] for x in overlaps), "mapping_precision":"conservative_block_page_range_not_bbox", "split":split})
    jsonl(ROOT / "data/manifests/chinese_pdf_variants.jsonl", manifest)
    jsonl(ROOT / "data/annotations/tidb_zh/pdf_evidence_map.jsonl", page_maps)
    write_json(ROOT / "data/manifests/chinese_pdf_summary.json", {"pdfs":pdf_summaries, "source_documents":len(selected), "warnings":warnings, "pdf_evidence_ranges":len(page_maps), "font_sha256":hashes(FONT)[0]})
    write_json(out / "sources.json", manifest)
    print(json.dumps({"pdfs":pdf_summaries, "warnings":len(warnings)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
