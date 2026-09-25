"""Offline tests for data contracts, source integrity and label isolation."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_datasets import hashes
from prepare_datasets import connected_components, source_group


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line]


def test_components_are_transitive():
    assert connected_components(["a", "b", "c", "d"], {"a": {1}, "b": {1, 2}, "c": {2}, "d": {3}}) == [["a", "b", "c"], ["d"]]


@pytest.mark.parametrize("filename,expected", [("book_page_002.png", "book"), ("notes_a17_3.jpg", "notes_a17"), ("page-uuid.png", "page-uuid")])
def test_source_filename_group(filename, expected):
    assert source_group(filename) == expected


def test_hashes(tmp_path):
    path = tmp_path / "sample"
    path.write_bytes(b"abc")
    assert hashes(path) == (hashlib.sha256(b"abc").hexdigest(), hashlib.sha1(b"blob 3\0abc").hexdigest())


def test_all_downloads_verified():
    lock = json.loads((ROOT / "data/manifests/sources.lock.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "data/manifests/download-report.json").read_text(encoding="utf-8"))
    expected = {f"data/raw/{s['name']}/{f['path']}": (f, s) for s in lock["sources"] for f in s["files"]}
    assert report["complete"]
    assert len(report["files"]) == len(expected) == 1672
    assert {f["path"] for f in report["files"]} == set(expected)
    for item in report["files"]:
        path = ROOT / item["path"]
        assert item["verified"] and path.stat().st_size == item["bytes"]
        sha, git = hashes(path)
        upstream = expected[item["path"]][0]
        assert sha == item["sha256"]
        if "lfs" in upstream:
            assert sha == upstream["lfs"]["oid"]
        else:
            assert git == upstream["oid"]


def test_complete_pdf_page_mapping():
    pages = rows("data/annotations/vidore/pages.jsonl")
    docs = rows("data/manifests/documents.jsonl")
    assert len(pages) == 1360
    for doc in docs:
        selected = [p for p in pages if p["document_id"] == doc["doc_id"]]
        assert {p["page_index"] for p in selected} == set(range(doc["physical_pages"]))
        for p in selected:
            assert p["page_number"] == p["page_index"] + 1
            assert hashes(ROOT / p["image_path"])[0] == p["sha256"]


def test_qa_evidence_references_exist():
    pages = {p["page_id"]: p for p in rows("data/annotations/vidore/pages.jsonl")}
    qa = rows("data/annotations/vidore/qa_all_languages.jsonl")
    assert len(qa) == len({q["case_id"] for q in qa}) == 1290
    assert len({q["group_id"] for q in qa}) == 215
    assert sum(len(q["evidence"]) for q in qa) == 6294
    for q in qa:
        assert q["reference_answer"] and q["evidence"]
        for e in q["evidence"]:
            assert e["page_id"] in pages
            assert e["score"] in (1, 2)
            assert e["page_number"] == pages[e["page_id"]]["page_number"]


def test_translation_and_evidence_split_isolation():
    dev = rows("data/splits/vidore/dev_all_languages.jsonl")
    test = rows("data/splits/vidore/test_all_languages.jsonl")
    assert len(dev) + len(test) == 1290
    assert not {q["group_id"] for q in dev} & {q["group_id"] for q in test}
    assert not {e["page_id"] for q in dev for e in q["evidence"]} & {e["page_id"] for q in test for e in q["evidence"]}
    assert len({q["group_id"] for q in dev}) == 182
    assert len({q["group_id"] for q in test}) == 33


def test_no_labels_in_ingestion_allowlists():
    allowed = rows("data/manifests/ingestion_public.jsonl") + rows("data/manifests/ingestion_engineering.jsonl")
    assert len(allowed) == 26
    for doc in allowed:
        path = ROOT / doc["path"]
        assert path.suffix in {".pdf", ".md", ".txt"}
        assert "/annotations/" not in doc["path"] and "/splits/" not in doc["path"]
        assert hashes(path)[0] == doc["sha256"]
    assert all("raw/vidore/pdfs/" in d["path"] for d in allowed if d["format"] == "pdf")


def test_full_omni_coverage_and_group_isolation():
    all_rows = rows("data/annotations/omnidocbench/pages.jsonl")
    dev = rows("data/splits/omnidocbench/dev.jsonl")
    test = rows("data/splits/omnidocbench/test.jsonl")
    assert len(all_rows) == len(dev) + len(test) == 1651
    assert not {r["group_id"] for r in dev} & {r["group_id"] for r in test}
    assert all((ROOT / r["image_path"]).is_file() and r["layout_dets"] for r in all_rows)
    inputs = rows("data/manifests/parsing_inputs.jsonl")
    assert len(inputs) == 1651 and all("layout_dets" not in r for r in inputs)


def test_fictional_labels_match_documents():
    docs = {d["document_id"]: d for d in rows("data/manifests/ingestion_engineering.jsonl")}
    qa = rows("data/annotations/engineering/qa.jsonl")
    assert len(docs) == 24 and len(qa) == 60
    assert sum(not q["answerable"] for q in qa) == 12
    for q in qa:
        text = "\n".join((ROOT / docs[e["document_id"]]["path"]).read_text(encoding="utf-8") for e in q["evidence"])
        assert all(fact in text for fact in q["required_facts"])
        assert not q["local_human_reviewed"]
    for doc in docs.values():
        text = (ROOT / doc["path"]).read_text(encoding="utf-8")
        assert "虚构工程测试资料" in text and "\ufffd" not in text
        if doc["format"] == "txt":
            assert "#" not in text


def test_agent_cases_are_prepared_not_claimed_passed():
    cases = rows("evals/cases/agent_workflows.jsonl")
    assert len(cases) == len({c["case_id"] for c in cases}) == 40
    assert all(c["status"] == "case_prepared_not_executed" for c in cases)
    assert all(c["turns"] and c["expected"] and c["budgets"]["tool_calls"] <= 10 for c in cases)


def test_engineering_topics_do_not_cross_split():
    dev = rows("data/splits/engineering/dev.jsonl")
    test = rows("data/splits/engineering/test.jsonl")
    assert (len(dev), len(test)) == (40, 20)
    assert not {q["group_id"] for q in dev} & {q["group_id"] for q in test}
