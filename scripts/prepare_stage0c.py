"""用数据环境固定0C开发样本。只选dev，原文件不改写，参考标注不进入上传包。"""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/stage0c"
ART = ROOT / "evals/results/stage0c/preparation"


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    manifest_path = ROOT / "evals/configs/stage0c_samples.json"
    if manifest_path.exists():
        raise RuntimeError("Fixed sample manifest already exists; inspect rather than overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    pdfs = [x for x in rows("data/manifests/chinese_pdf_variants.jsonl") if x["split"] == "dev"]
    sections = {x["evidence_id"]: x for x in rows("data/annotations/tidb_zh/sections.jsonl") if x["split"] == "dev"}
    mappings = [x for x in rows("data/annotations/tidb_zh/pdf_evidence_map.jsonl") if x["split"] == "dev"]
    source_path = ROOT / pdfs[0]["path"]
    source_sha = sha(source_path)
    assert all(x["sha256"] == source_sha for x in pdfs)
    reader = PdfReader(source_path)
    selected, used = [], set()

    def add_zh(page, category):
        owner = next(x for x in pdfs if x["page_start"] <= page <= x["page_end"])
        selected.append({"category": category, "source_path": relative(source_path), "source_sha256": source_sha,
                         "source_page": page, "document_id": owner["document_id"], "split": "dev",
                         "language": "zh", "source_kind": "pdf", "project_rendered": True,
                         "license": owner["license"], "title": owner["title"]})
        used.add(page)
    # 连续4页，用两个2页样例检验局部页码到原文物理页的映射。
    for page in range(pdfs[0]["page_start"], pdfs[0]["page_start"] + 4):
        add_zh(page, "zh_native")
    table_candidates = sorted({m["page_start"] for m in mappings if m["page_start"] == m["page_end"]
        and m["evidence_id"] in sections and re.search(r"\|\s*:?-{3,}", sections[m["evidence_id"]]["text"])})
    for page in [p for p in table_candidates if p not in used][:4]:
        add_zh(page, "zh_table")
    for page in range(2, len(reader.pages) + 1):
        if page not in used and len(reader.pages[page-1].images):
            add_zh(page, "zh_visual")
            if sum(x["category"] == "zh_visual" for x in selected) == 4:
                break
    assert Counter(x["category"] for x in selected) == {"zh_native":4, "zh_table":4, "zh_visual":4}
    omni = sorted([x for x in rows("data/splits/omnidocbench/dev.jsonl")
        if x["page_attributes"]["language"] == "simplified_chinese"
        and x["page_attributes"]["layout"] != "single_column"
        and sum(len(d.get("text", "")) for d in x["layout_dets"] if not d.get("ignore")) > 150], key=lambda x:x["case_id"])[:4]
    references = {}
    for x in omni:
        path = ROOT / x["image_path"]
        assert sha(path) == x["sha256"]
        selected.append({"category":"zh_complex_image", "source_path": x["image_path"], "source_sha256":x["sha256"],
            "source_page":None, "document_id":x["case_id"], "split":"dev", "language":"zh", "source_kind":"image",
            "width":x["width"], "height":x["height"], "project_rendered":False,
            "license":"upstream OmniDocBench; see data source manifest", "title":x["case_id"]})
        references[x["case_id"]] = {"provenance":"upstream_official_not_locally_human_reviewed", "layout_dets":x["layout_dets"]}
    # 英文页必须属于开发证据，且不与测试证据页重叠。
    test_pages = {(e["document_id"],e["page_number"]) for x in rows("data/splits/vidore/test_english.jsonl") for e in x["evidence"]}
    candidates = sorted({(e["document_id"],e["page_number"]) for x in rows("data/splits/vidore/dev_english.jsonl") for e in x["evidence"]} - test_pages)
    docs = {x["doc_id"]: x for x in rows("data/manifests/documents.jsonl")}
    for doc_id, page in candidates[:4]:
        x=docs[doc_id]
        selected.append({"category":"en_pdf", "source_path":x["path"], "source_sha256":x["sha256"], "source_page":page,
            "document_id":doc_id,"split":"dev","language":"en","source_kind":"pdf","project_rendered":False,
            "license":x["license"],"title":doc_id})
    assert len(selected) == 20
    # 原件校验只做一次，避免大PDF反复计算。
    for path, digest in {(x["source_path"],x["source_sha256"]) for x in selected}:
        assert sha(ROOT / path) == digest
    groups = [selected[:2],selected[2:4]] + [[x] for x in selected[4:]]
    samples = []
    readers = {relative(source_path):reader}
    for i, group in enumerate(groups, 1):
        sample_id = f"s0c_{i:02d}"
        target = OUT / f"{sample_id}.pdf"
        if target.exists():
            raise RuntimeError("Refusing to overwrite existing sample PDF")
        if group[0]["source_kind"] == "image":
            image = ROOT / group[0]["source_path"]
            with Image.open(image) as im:
                w,h = im.size
                c = canvas.Canvas(str(target), pagesize=(w*.5,h*.5), invariant=1)
                c.drawImage(ImageReader(im),0,0,width=w*.5,height=h*.5)
                c.showPage(); c.save()
        else:
            writer = PdfWriter()
            for item in group:
                if item["source_path"] not in readers:
                    readers[item["source_path"]] = PdfReader(ROOT / item["source_path"])
                writer.add_page(readers[item["source_path"]].pages[item["source_page"]-1])
            with target.open("wb") as stream:
                writer.write(stream)
        # 复杂扫描页保持原分辨率，不为沿用0B的2MB样例限制而有损压缩。
        assert len(PdfReader(target).pages) == len(group) and target.stat().st_size <= 20_971_520
        sample = {"sample_id":sample_id,"input_path":relative(target),"input_sha256":sha(target),
                  "page_count":len(group),"category":group[0]["category"],"pages":group}
        # 渲染真实待上传文件，验证生成页而非仅抽取文本；保留PNG供对照。
        prefix = ART / sample_id
        subprocess.run([shutil.which("pdftoppm"),"-scale-to","1400","-png",str(target),str(prefix)],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=45)
        sample["previews"] = [relative(p) for p in sorted(ART.glob(sample_id+"-*.png"))]
        assert len(sample["previews"]) == len(group)
        samples.append(sample)
    md = [x for x in rows("data/manifests/ingestion_chinese_md.jsonl") if x["split"] == "dev"][:3]
    txt = [x for x in rows("data/manifests/chinese_txt_variants.jsonl") if x["split"] == "dev" and x["document_id"] not in {d["document_id"] for d in md}][:3]
    text_samples = [{k:x[k] for k in ("document_id","path","sha256","format","split","title","source_family","license")} for x in md+txt]
    payload = {"schema_version":1,"selection":"fixed_dev_only_before_api","pages_total":20,"pdf_files":len(samples),
               "samples":samples,"text_samples":text_samples,"quality_status":"not_run",
               "notes":["中文PDF为已有MD重排版，不冒充真实扫描文档。","复杂版面来自OmniDocBench原始页图，包装PDF不增加OCR文字层。",
                        "TXT为原MD逐字格式副本，不冒充弱结构手写文档。","参考标注单独保存，上传包只含原文页面。"]}
    manifest_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    (ART/"reference_only.json").write_text(json.dumps(references,ensure_ascii=False,indent=2),encoding="utf-8")
    cards = ''.join(f'<h2>{x["sample_id"]} {x["category"]}</h2><p>{html.escape(str([(p["title"],p["source_page"]) for p in x["pages"]]))}</p>'+''.join(f'<img style="max-width:650px" src="{Path(p).name}">' for p in x['previews']) for x in samples)
    (ART/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>0C固定样本</title><h1>0C固定开发样本：20页 + 6份文本</h1><p>尚未调用解析或视觉接口，不代表质量评分。</p>'+cards,encoding="utf-8")
    # 联系表用于人工查看各页类型；不作为模型输入。
    thumbnails=[]
    for sample in samples:
        for path in sample['previews']:
            with Image.open(ROOT/path) as im:
                tile=Image.new('RGB',(280,410),'white'); im.thumbnail((280,385)); tile.paste(im,(0,20))
                from PIL import ImageDraw
                ImageDraw.Draw(tile).text((6,2),sample['sample_id']+' '+Path(path).stem,fill='black'); thumbnails.append(tile)
    sheet=Image.new('RGB',(280*5,410*4),'#dddddd')
    for i,tile in enumerate(thumbnails): sheet.paste(tile,((i%5)*280,(i//5)*410))
    sheet.save(ART/'contact-sheet.png')
    print(json.dumps({'pages':20,'pdf_files':len(samples),'text_files':len(text_samples),'categories':dict(Counter(x['category'] for x in selected)), 'manifest':relative(manifest_path)},ensure_ascii=False))


if __name__ == '__main__':
    main()
