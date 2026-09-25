"""Data acceptance only: none of these tests measure RAG or Agent quality."""
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
import sys

import pytest
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_chinese_cases import rows
from download_datasets import hashes
from prepare_chinese import SOURCE, sections, clean
from export_chinese_gold import validate_reviews


def test_chinese_source_complete_and_unchanged():
    report=json.loads((ROOT/"data/manifests/chinese_download_report.json").read_text(encoding="utf-8"))
    assert report["complete"] and len(report["files"])==2849
    for f in report["files"]:
        p=ROOT/f["path"]
        assert p.stat().st_size==f["bytes"]
        assert hashes(p)==(f["sha256"],f["git_blob"])


def test_chinese_primary_manifest_excludes_support_and_answers():
    docs=rows("data/manifests/ingestion_chinese_md.jsonl")
    assert len(docs)==len({d["document_id"] for d in docs})==1109
    for d in docs:
        assert d["path"].startswith("data/raw/tidb_zh/source/")
        assert all(k not in d for k in ("question","reference_answer","required_facts"))
        assert not any(part in {"releases","resources","scripts",".agents"} for part in Path(d["source_path"]).parts)


def test_all_chinese_section_line_evidence_reconstructs_exactly():
    cache={}
    ss=rows("data/annotations/tidb_zh/sections.jsonl")
    assert len(ss)==12122
    for s in ss:
        if s["source_path"] not in cache:
            cache[s["source_path"]]=(SOURCE/s["source_path"]).read_text(encoding="utf-8").splitlines()
        text="\n".join(cache[s["source_path"]][s["line_start"]-1:s["line_end"]]).strip()
        assert text==s["text"]
        assert hashlib.sha256(text.encode()).hexdigest()==s["text_sha256"]


def test_markdown_headings_inside_fence_are_not_sections():
    ss=sections("fixture.md","---\ntitle: test\n---\n# 主标题\n正文\n```python\n# not heading\n```\n## 次标题\n内容")
    assert [s["heading_path"] for s in ss]==[["主标题"],["主标题","次标题"]]


def test_code_placeholders_not_mistaken_for_html():
    assert clean('日志 `[INFO] [<unknown>]` 和 <b>正文</b>')=='日志 `[INFO] [<unknown>]` 和 正文'


def test_table_anchors_exist_in_mapped_pdf_pages():
    maps={p["evidence_id"]:p for p in rows("data/annotations/tidb_zh/pdf_evidence_map.jsonl")}
    readers={}
    compact=lambda text: re.sub(r"[\s`*_]", "", text)
    for q in rows("data/annotations/tidb_zh/qa_candidates.jsonl"):
        if q["type"]!="table": continue
        p=maps[q["evidence"][0]["evidence_id"]]
        if p["pdf_path"] not in readers: readers[p["pdf_path"]]=PdfReader(ROOT/p["pdf_path"])
        text="".join(page.extract_text() for page in readers[p["pdf_path"]].pages[p["page_start"]-1:p["page_end"]])
        assert compact(q["table_evidence"]["cells"][0]) in compact(text),q["case_id"]


def test_qa_counts_references_and_pending_status():
    qa=rows("data/annotations/tidb_zh/qa_candidates.jsonl")
    ss={s["evidence_id"]:s for s in rows("data/annotations/tidb_zh/sections.jsonl")}
    assert len(qa)==len({q["case_id"] for q in qa})==180
    assert Counter(q["split"] for q in qa)=={"dev":120,"test":60}
    assert Counter(q["type"] for q in qa)=={"concept_procedure":40,"configuration":30,"table":30,"visual":30,"multi_document":30,"unanswerable_clarify":20}
    for q in qa:
        assert "document_ids" not in q["scope"]  # No oracle filtering during retrieval.
        assert q["reference_answer"] and not q["gold_eligible"]
        assert q["review_status"]=="pending_human_review"
        assert set(q["group_ids"])=={ss[e["evidence_id"]]["group_id"] for e in q["evidence"]}
        for e in q["evidence"]:
            assert ss[e["evidence_id"]]["split"]==q["split"]
            assert e["text_sha256"]==ss[e["evidence_id"]]["text_sha256"]


def test_dev_test_source_families_and_visual_assets_do_not_overlap():
    docs=rows("data/manifests/ingestion_chinese_md.jsonl")
    groups=defaultdict(set)
    for d in docs: groups[d["group_id"]].add(d["split"])
    assert all(len(v)==1 for v in groups.values())
    qa=rows("data/annotations/tidb_zh/qa_candidates.jsonl")
    for key in ("document_id","evidence_id"):
        ids={sp:{e[key] for q in qa if q["split"]==sp for e in q["evidence"]} for sp in ("dev","test")}
        assert not ids["dev"] & ids["test"]
    assets={sp:{q["visual_evidence"]["sha256"] for q in qa if q["split"]==sp and q["type"]=="visual"} for sp in ("dev","test")}
    assert not assets["dev"] & assets["test"]


def test_visual_drafts_grounded_in_original_files_not_gold():
    visual=[q for q in rows("data/annotations/tidb_zh/qa_candidates.jsonl") if q["type"]=="visual"]
    assert len(visual)==30
    for q in visual:
        assert hashes(ROOT/q["visual_evidence"]["asset_path"])[0]==q["visual_evidence"]["sha256"]
        assert q["required_facts"] and q["answer_status"]=="assistant_visual_reviewed_draft_pending_human"


def test_table_evidence_preserves_source_row():
    for q in rows("data/annotations/tidb_zh/qa_candidates.jsonl"):
        if q["type"]!="table": continue
        t=q["table_evidence"]
        lines=(SOURCE/q["evidence"][0]["source_path"]).read_text(encoding="utf-8").splitlines()
        assert "|" in lines[t["row_line"]-1] and "|" in lines[t["header_line"]-1]
        assert len(t["headers"])==len(t["cells"])


def test_pdf_sources_and_page_ranges_are_valid():
    variants=rows("data/manifests/chinese_pdf_variants.jsonl")
    docs={d["document_id"]:d for d in rows("data/manifests/ingestion_chinese_md.jsonl")}
    readers={p:PdfReader(ROOT/p) for p in {v["path"] for v in variants}}
    assert len(variants)>=40 and len(readers)==2
    for v in variants:
        assert v["source_sha256"]==docs[v["document_id"]]["sha256"]
        assert v["source_family"]==docs[v["document_id"]]["source_family"]
        assert v["split"]==docs[v["document_id"]]["split"]
        assert 1 < v["page_start"] <= v["page_end"] <= len(readers[v["path"]].pages)
    for p,r in readers.items():
        assert "TiDB" in r.pages[0].extract_text()
        assert "中文" in r.pages[0].extract_text()
        assert all(page.extract_text().strip() for page in r.pages)
    for e in rows("data/annotations/tidb_zh/pdf_evidence_map.jsonl"):
        assert 1<=e["page_start"]<=e["page_end"]<=len(readers[e["pdf_path"]].pages)


def test_txt_variants_are_exact_same_source_not_new_knowledge():
    variants=rows("data/manifests/chinese_txt_variants.jsonl")
    assert len(variants)==12
    for v in variants:
        assert hashes(ROOT/v["path"])[0]==hashes(ROOT/v["derived_from"])[0]
        assert v["variant_policy"].startswith("replace_md")


def test_agent_and_lifecycle_contracts_not_marked_executed():
    cases=rows("evals/cases/agent_chinese_multimodal.jsonl")
    assert len(cases)==40 and {c["split"] for c in cases}=={"dev","test"}
    assert sum("read_original_image" in c["expected_actions"] for c in cases)==30
    for c in cases:
        assert c["execution_status"]=="not_run" and c["metrics"] is None and not c["gold_eligible"]
    lifecycle=rows("evals/cases/chinese_lifecycle.jsonl")
    assert len(lifecycle)==8
    for c in lifecycle:
        assert c["execution_status"]=="not_run"
        assert all((ROOT/p).is_file() for p in c["files"])


def test_gold_export_requires_real_review_fields():
    candidates=rows("data/annotations/tidb_zh/qa_candidates.jsonl")
    q=candidates[0]
    with pytest.raises(ValueError):
        validate_reviews([{"case_id":q["case_id"],"status":"approved","reviewer":"","notes":""}],candidates)
    with pytest.raises(ValueError):
        validate_reviews([{"case_id":"unknown","status":"approved"}],candidates)
    assert validate_reviews([{"case_id":q["case_id"],"status":"needs_changes"}],candidates)==[]


def test_review_html_links_stay_local_and_sources_are_escaped():
    for sp,count in (("dev",120),("test",60)):
        text=(ROOT/f"evals/results/chinese-review/{sp}_review.html").read_text(encoding="utf-8")
        assert text.count('<article id=')==count
        assert '<script' not in text.lower()
        assert '待人工审核' in text
