"""Prepare a reviewable 30-case visual QA expansion without model calls.

The output remains candidate data.  This script validates source documents and
original image bytes, creates the exact ingestion allowlist, and builds a local
review worksheet.  It never changes PostgreSQL, an active index, or Gold data.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QA_SOURCE = ROOT / "data/annotations/tidb_zh/qa_candidates.jsonl"
DOC_SOURCE = ROOT / "data/manifests/ingestion_chinese_md.jsonl"
MANIFEST_TARGET = ROOT / "data/manifests/ingestion_chinese_visual_stage5.jsonl"
CANDIDATE_TARGET = ROOT / "evals/datasets/retrieval_stage5_visual_candidates_v2.jsonl"
REPORT_TARGET = ROOT / "evals/results/stage5/visual_expansion_readiness.json"
REVIEW_TARGET = ROOT / "evals/results/stage5/visual_gold_v2_review.html"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _category(row: dict) -> str:
    text = " ".join((row["evidence"][0]["source_path"],
                     row["visual_evidence"]["asset_path"], row["question"])).lower()
    if "hot-spot" in text:
        return "heatmap"
    if "autocommit" in text or "terminal" in text:
        return "terminal_screenshot"
    if any(token in text for token in ("architecture", "computing", "storage-1", "deploy-3dc",
                                        "data-centers", "raft log", "timer driven")):
        return "architecture_or_flow_diagram"
    if any(token in text for token in ("grafana", "monitor", "metrics", "qps", "local-reader")):
        return "monitoring_chart"
    if any(token in text for token in ("table", "ownership", "diagnostics")):
        return "table_screenshot"
    return "product_ui_screenshot"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")


def _review_page(rows: list[dict]) -> str:
    cards = []
    for row in rows:
        image = Path(row["visual_evidence"]["asset_path"])
        image_url = "../../../" + image.as_posix()
        facts = "；".join(row.get("required_facts") or [])
        mapping = row.get("active_visual_mapping") or {}
        mapping_html = (f"<dt>活动证据</dt><dd>Element：{escape(', '.join(mapping.get('element_ids', [])))}"
                        f"<br>Chunk：{escape(', '.join(mapping.get('chunk_ids', [])))}</dd>"
                        if mapping else "")
        cards.append(f'''<article data-case="{escape(row['case_id'])}" data-split="{row['split']}">
<header><span>{escape(row['split'])}</span><span>{escape(row['visual_category'])}</span><code>{escape(row['case_id'])}</code></header>
<h2>{escape(row['question'])}</h2>
<a href="{escape(image_url)}" target="_blank"><img loading="lazy" src="{escape(image_url)}" alt="原始图片"></a>
<dl><dt>候选答案</dt><dd>{escape(row['reference_answer'])}</dd><dt>必要要点</dt><dd>{escape(facts)}</dd>
<dt>来源</dt><dd>{escape(row['evidence'][0]['source_path'])} · 第 {row['evidence'][0]['line_start']}–{row['evidence'][0]['line_end']} 行</dd>{mapping_html}</dl>
<div class="decision"><label><input type="radio" name="{row['case_id']}" value="approved">通过</label>
<label><input type="radio" name="{row['case_id']}" value="needs_changes">需修改</label>
<label><input type="radio" name="{row['case_id']}" value="rejected">拒绝</label>
<textarea placeholder="审核备注；若需修改，请写明问题、答案或证据如何调整"></textarea></div></article>''')
    payload = json.dumps([{"case_id": row["case_id"], "split": row["split"]} for row in rows],
                         ensure_ascii=False)
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>视觉 Gold v2 人工审核</title>
<style>body{{max-width:1180px;margin:28px auto;padding:0 22px;background:#f4f7f8;color:#18313b;font:15px/1.65 system-ui,"Microsoft YaHei"}}.notice{{background:#fff4d8;border:1px solid #e7c980;padding:16px}}.toolbar{{position:sticky;top:0;background:#f4f7f8ee;padding:12px 0;z-index:2;display:flex;gap:12px;align-items:center}}button{{padding:9px 15px;border:0;border-radius:8px;background:#086879;color:white;cursor:pointer}}article{{background:white;border:1px solid #cfdbdf;border-radius:12px;padding:20px;margin:20px 0}}header{{display:flex;gap:10px;align-items:center}}header span{{background:#e5f3f5;padding:2px 8px;border-radius:12px}}h2{{font-size:18px}}img{{display:block;max-width:100%;max-height:650px;margin:16px auto;object-fit:contain;border:1px solid #dde5e8}}dl{{display:grid;grid-template-columns:86px 1fr;gap:8px}}dt{{font-weight:700}}dd{{margin:0}}.decision{{border-top:1px solid #e1e8ea;margin-top:16px;padding-top:14px}}label{{margin-right:20px}}textarea{{box-sizing:border-box;width:100%;min-height:72px;margin-top:12px;padding:10px}}code{{color:#526970}}.done{{border-color:#39a078;box-shadow:0 0 0 2px #39a07822}}</style>
<h1>视觉 Gold v2 人工审核 · {len(rows)} 条</h1><p class="notice">请必须查看原图后再决定。候选问题与答案由助手看图起草，不是正式Gold；只有导出的审核决定再次校验并发布后，才能进入正式评测。test样本只做最终评测，不用于调参。</p>
<div class="toolbar"><strong id="progress">已审核 0 / {len(rows)}</strong><button onclick="exportReview()">导出审核决定 JSON</button><button onclick="clearReview()">清空本页记录</button></div>{''.join(cards)}
<script>const cases={payload};const key='mrag_visual_gold_v2_review';function read(){{try{{return JSON.parse(localStorage.getItem(key)||'{{}}')}}catch{{return {{}}}}}}function refresh(){{const s=read();let n=0;document.querySelectorAll('article').forEach(a=>{{const v=s[a.dataset.case];a.classList.toggle('done',!!v?.status);if(v?.status){{n++;const r=a.querySelector(`input[value="${{v.status}}"]`);if(r)r.checked=true;a.querySelector('textarea').value=v.notes||''}}}});document.querySelector('#progress').textContent=`已审核 ${{n}} / {len(rows)}`}}document.querySelectorAll('article').forEach(a=>{{a.addEventListener('change',save);a.querySelector('textarea').addEventListener('input',save);function save(){{const s=read(),checked=a.querySelector('input:checked');s[a.dataset.case]={{case_id:a.dataset.case,status:checked?.value||null,notes:a.querySelector('textarea').value,reviewer:'human-review',reviewed_at:new Date().toISOString()}};localStorage.setItem(key,JSON.stringify(s));refresh()}}}});function exportReview(){{const s=read();const rows=cases.map(c=>s[c.case_id]).filter(Boolean);const b=new Blob([JSON.stringify(rows,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='visual_gold_v2_review_decisions.json';a.click();URL.revokeObjectURL(a.href)}}function clearReview(){{if(confirm('确认清空本页审核记录？')){{localStorage.removeItem(key);location.reload()}}}}refresh();</script></html>'''


def main() -> int:
    visual = [row for row in _jsonl(QA_SOURCE) if row.get("type") == "visual"]
    documents = {row["document_id"]: row for row in _jsonl(DOC_SOURCE)}
    if len(visual) != 30 or len({row["case_id"] for row in visual}) != 30:
        raise ValueError("视觉候选必须恰好为30条且编号唯一")
    if Counter(row["split"] for row in visual) != {"dev": 20, "test": 10}:
        raise ValueError("视觉候选应保持20条dev与10条test的既定划分")

    candidates: list[dict] = []
    manifest: list[dict] = []
    seen_documents: set[str] = set()
    for source in sorted(visual, key=lambda row: (row["split"], row["case_id"])):
        row = dict(source)
        document_id = row["reference_document_ids"][0]
        document = documents.get(document_id)
        if document is None:
            raise ValueError(f"缺少文档清单：{document_id}")
        document_path = ROOT / document["path"]
        image_path = ROOT / row["visual_evidence"]["asset_path"]
        if not document_path.is_file() or _digest(document_path) != document["sha256"]:
            raise ValueError(f"文档不存在或哈希变化：{document['path']}")
        if not image_path.is_file() or _digest(image_path) != row["visual_evidence"]["sha256"]:
            raise ValueError(f"图片不存在或哈希变化：{image_path}")
        prefix = image_path.read_bytes()[:8]
        if not (prefix.startswith(b"\x89PNG\r\n\x1a\n") or prefix.startswith(b"\xff\xd8\xff")):
            raise ValueError(f"图片签名非法：{image_path}")
        if document_id not in seen_documents:
            # The source document belongs to the searchable knowledge corpus in
            # both cases.  Only QA labels keep the dev/test boundary.  The
            # reader's ``split`` field is an ingestion exposure gate, so retain
            # the original evaluation partition separately and mark the
            # allowlisted source as development-ingestable.
            manifest.append({**document, "source_evaluation_split": document["split"],
                             "split": "dev", "stage5_visual_subset": True,
                             "selection_reason": "contains_visual_gold_v2_candidate"})
            seen_documents.add(document_id)
        row.update({
            "visual_category": _category(row),
            "evaluation_status": "visual_candidate_v2_pending_human_review",
            "review_status": "pending_human_review",
            "gold_eligible": False,
            "review_requirements": [
                "人工实际查看原图", "核对问题没有歧义", "核对答案数值、单位和关系",
                "确认答案必须依赖图片", "登记可替代或应排除的图片",
            ],
        })
        candidates.append(row)

    _write_jsonl(MANIFEST_TARGET, manifest)
    _write_jsonl(CANDIDATE_TARGET, candidates)
    REVIEW_TARGET.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_TARGET.write_text(_review_page(candidates), encoding="utf-8")
    report = {
        "status": "ready_for_human_review_and_ingestion",
        "candidate_count": len(candidates),
        "document_count": len(manifest),
        "image_count": len(candidates),
        "splits": dict(Counter(row["split"] for row in candidates)),
        "knowledge_documents_by_evaluation_split": dict(Counter(
            row["source_evaluation_split"] for row in manifest)),
        "categories": dict(Counter(row["visual_category"] for row in candidates)),
        "source_and_image_hashes_valid": True,
        "model_calls": 0,
        "database_writes": 0,
        "active_index_changed": False,
        "candidate_dataset": CANDIDATE_TARGET.relative_to(ROOT).as_posix(),
        "ingestion_manifest": MANIFEST_TARGET.relative_to(ROOT).as_posix(),
        "review_page": REVIEW_TARGET.relative_to(ROOT).as_posix(),
        "warning": "候选仍需逐图人工审核，不能作为正式Gold或准确率统计。",
    }
    REPORT_TARGET.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
