"""Normalize complete downloaded corpora without invoking models or parsers.

Evaluation labels and upstream OCR stay outside the ingestion allowlist. Original
PDF page indices are zero-based; the UI page number is explicitly one-based.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re

from PIL import Image
import pyarrow.parquet as pq
from pypdf import PdfReader

from download_datasets import ROOT, hashes, write_json


def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_rows(folder):
    for path in sorted(folder.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            yield from batch.to_pylist()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def stable_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def connected_components(groups, group_evidence):
    """Keep every question connected by an evidence unit on the same side."""
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    owners = {}
    for group in sorted(groups):
        for evidence in group_evidence[group]:
            if evidence in owners:
                parent[find(group)] = find(owners[evidence])
            else:
                owners[evidence] = group
    result = defaultdict(list)
    for group in sorted(groups):
        result[find(group)].append(group)
    return sorted(result.values(), key=lambda c: (-len(c), stable_id(c)))


def build_vidore():
    folder = ROOT / "data/raw/vidore"
    metadata = list(read_rows(folder / "documents_metadata"))
    documents, chapters = {}, {}
    for item in metadata:
        path = folder / "pdfs" / item["file_name"]
        reader = PdfReader(path)
        assert len(reader.pages) == item["page_number"]
        outline = sorted([(reader.get_destination_page_number(x), x.title) for x in reader.outline if not isinstance(x, list)])
        chapters[item["doc_id"]] = outline
        documents[item["doc_id"]] = {**item, "path": rel(path), "sha256": hashes(path)[0], "physical_pages": len(reader.pages)}
    jsonl(ROOT / "data/manifests/documents.jsonl", documents.values())
    page_rows, upstream_ocr = [], []
    for item in read_rows(folder / "corpus"):
        raw = item["image"]["bytes"]
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            extension = ".png" if image.format == "PNG" else ".jpg"
            image.verify()
        destination = ROOT / "data/derived/vidore/pages" / item["doc_id"] / f"page_{item['page_number_in_doc'] + 1:04d}{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        page_index = item["page_number_in_doc"]
        assert 0 <= page_index < documents[item["doc_id"]]["physical_pages"]
        title = "Front matter"
        for start, name in chapters[item["doc_id"]]:
            if start <= page_index:
                title = name
        page_rows.append({"page_id": f"vidore:{item['corpus_id']}", "corpus_id": item["corpus_id"],
                          "document_id": item["doc_id"], "pdf_path": documents[item["doc_id"]]["path"],
                          "page_index": page_index, "page_number": page_index + 1,
                          "chapter": title, "image_path": rel(destination), "width": width, "height": height,
                          "sha256": hashlib.sha256(raw).hexdigest()})
        upstream_ocr.append({"page_id": f"vidore:{item['corpus_id']}", "text": item["markdown"],
                             "provenance": "upstream_machine_ocr_not_gold", "ingest_allowed": False})
    assert len(page_rows) == len({p["page_id"] for p in page_rows})
    assert len({(p["document_id"], p["page_index"]) for p in page_rows}) == sum(x["physical_pages"] for x in documents.values())
    page_rows.sort(key=lambda p: (p["document_id"], p["page_index"]))
    page_by_id = {p["corpus_id"]: p for p in page_rows}
    jsonl(ROOT / "data/annotations/vidore/pages.jsonl", page_rows)
    jsonl(ROOT / "data/annotations/vidore/upstream_ocr_reference.jsonl", upstream_ocr)
    raw_queries = list(read_rows(folder / "queries"))
    raw_qrels = list(read_rows(folder / "qrels"))
    qrels_by_query = defaultdict(list)
    bbox_warnings = []
    for item in raw_qrels:
        page = page_by_id[item["corpus_id"]]
        boxes = []
        for box in item["bounding_boxes"]:
            # Do not silently clip bad gold annotations; retain and flag them.
            valid = 0 <= box["x1"] < box["x2"] <= page["width"] and 0 <= box["y1"] < box["y2"] <= page["height"]
            if not valid:
                bbox_warnings.append({"query_id": item["query_id"], "page_id": page["page_id"], "box": box})
            boxes.append({"original_pixels": box, "valid": valid, "normalized_xyxy": [box["x1"]/page["width"], box["y1"]/page["height"], box["x2"]/page["width"], box["y2"]/page["height"]] if valid else None})
        qrels_by_query[item["query_id"]].append({"page_id": page["page_id"], "document_id": page["document_id"],
                                                "page_number": page["page_number"], "score": item["score"],
                                                "content_types": item["content_type"], "boxes": boxes})
    # Translation grouping is verified against both raw answers and qrels,
    # rather than assuming the numerical IDs have a modulo relationship.
    grouped = defaultdict(list)
    for item in raw_queries:
        group = stable_id(item["raw_answers"])
        grouped[group].append(item)
    assert all(len(v) == 6 and len({q["language"] for q in v}) == 6 for v in grouped.values())
    group_pages, group_chapters = {}, {}
    for group, queries in grouped.items():
        signatures = {tuple(sorted((r["page_id"], r["score"]) for r in qrels_by_query[q["query_id"]])) for q in queries}
        assert len(signatures) == 1, "Translated queries have different relevance annotations"
        group_pages[group] = {r["page_id"] for r in qrels_by_query[queries[0]["query_id"]]}
        group_chapters[group] = {f"{page_by_id[int(p.split(':')[1])]['document_id']}:{page_by_id[int(p.split(':')[1])]['chapter']}" for p in group_pages[group]}
    components = connected_components(grouped, group_pages)
    # Largest evidence-connected component is development. All other components
    # remain locked, avoiding misleading random 60/40 splitting.
    dev_groups = set(components[0])
    split_by_group = {g: "dev" if g in dev_groups else "test" for g in grouped}
    cases = []
    for group, queries in grouped.items():
        for item in queries:
            cases.append({"case_id": f"vidore_q{item['query_id']}", "source": "vidore_official",
                          "source_query_id": item["query_id"], "group_id": group,
                          "split": split_by_group[group], "question": item["query"], "language": item["language"],
                          "question_types": item["query_types"], "content_types": item["content_type"],
                          "reference_answer": item["answer"], "raw_answers": item["raw_answers"],
                          "answer_provenance": "upstream_llm_merge_of_human_annotations",
                          "query_generator": item["query_generator"], "answerable": True,
                          "evidence": qrels_by_query[item["query_id"]], "local_human_reviewed": False})
    cases.sort(key=lambda q: q["source_query_id"])
    assert all(c["evidence"] for c in cases)
    jsonl(ROOT / "data/annotations/vidore/qa_all_languages.jsonl", cases)
    jsonl(ROOT / "data/annotations/vidore/qa_english.jsonl", (q for q in cases if q["language"] == "english"))
    split_counts = {}
    for split in ("dev", "test"):
        selected = [q for q in cases if q["split"] == split]
        english = [q for q in selected if q["language"] == "english"]
        jsonl(ROOT / f"data/splits/vidore/{split}_english.jsonl", english)
        jsonl(ROOT / f"data/splits/vidore/{split}_all_languages.jsonl", selected)
        split_counts[split] = len(english)
    dev_pages = set().union(*(group_pages[g] for g in dev_groups))
    test_pages = set().union(*(group_pages[g] for g in grouped if g not in dev_groups))
    assert not dev_pages & test_pages
    dev_chapters = set().union(*(group_chapters[g] for g in dev_groups))
    test_chapters = set().union(*(group_chapters[g] for g in grouped if g not in dev_groups))
    chapter_components = connected_components(grouped, group_chapters)
    write_json(ROOT / "data/splits/vidore/split_policy.json", {
        "policy": "translation_groups_and_evidence_page_connected_components_v1", "counts": split_counts,
        "evidence_page_overlap": 0, "chapter_overlap_count": len(dev_chapters & test_chapters),
        "chapter_overlap": sorted(dev_chapters & test_chapters),
        "chapter_connected_component_sizes": [len(c) for c in chapter_components],
        "note": "Page-disjoint, NOT chapter-disjoint or document-disjoint. Full official set is an external benchmark, not an untouched holdout after development use.",
        "evidence_component_sizes": [len(c) for c in components],
        "test_policy": "Do not tune on test answers; a local lock is procedural, not an access-control boundary."})
    write_json(ROOT / "data/annotations/vidore/bbox_warnings.json", bbox_warnings)
    return {"documents": len(documents), "pages": len(page_rows), "queries": len(cases),
            "unique_questions": len(grouped), "languages": dict(Counter(q["language"] for q in cases)),
            "qrels": len(raw_qrels), "splits_english": split_counts,
            "invalid_boxes": len(bbox_warnings), "chapter_overlap_count": len(dev_chapters & test_chapters)}, list(documents.values())


def source_group(name):
    # Typical upstream names encode PDF prefix followed by _page_003 or _17.
    stem = Path(name).stem
    return re.sub(r"(?:_page_\d+|_\d+)$", "", stem)


def build_omni():
    folder = ROOT / "data/raw/omnidocbench"
    raw = json.loads((folder / "OmniDocBench.json").read_text(encoding="utf-8"))
    rows, warnings, large_images = [], [], []
    for item in raw:
        info = item["page_info"]
        image = folder / "images" / info["image_path"]
        with Image.open(image) as opened:
            assert opened.size == (info["width"], info["height"]), info["image_path"]
            opened.verify()
        group = source_group(info["image_path"])
        if info["width"] * info["height"] > 50_000_000:
            large_images.append({"image_path": rel(image), "width": info["width"], "height": info["height"],
                                 "action": "Retain original. Future parser adapter must budget pixels and record any rescaling transform."})
        split = "dev" if int(stable_id(group), 16) % 10 < 6 else "test"
        for element in item["layout_dets"]:
            poly = element.get("poly", [])
            if any(x < 0 or x > info["width"] for x in poly[::2]) or any(y < 0 or y > info["height"] for y in poly[1::2]):
                warnings.append({"image": info["image_path"], "anno_id": element.get("anno_id"), "issue": "polygon_outside_page"})
        rows.append({"case_id": "omni_" + stable_id(info["image_path"]), "source": "omnidocbench_official",
                     "group_id": group, "split": split, "image_path": rel(image),
                     "source_page_number": info["page_no"], "page_attributes": info["page_attribute"],
                     "width": info["width"], "height": info["height"],
                     "sha256": hashes(image)[0], "layout_dets": item["layout_dets"], "extra": item.get("extra"),
                     "local_human_reviewed": False, "use": "parsing_evaluation_only"})
    assert len(rows) == len({r["case_id"] for r in rows})
    jsonl(ROOT / "data/annotations/omnidocbench/pages.jsonl", rows)
    jsonl(ROOT / "data/manifests/parsing_inputs.jsonl", ({"document_id": r["case_id"], "path": r["image_path"],
          "sha256": r["sha256"], "format": Path(r["image_path"]).suffix.lstrip("."),
          "use": "isolated_parsing_evaluation_not_qa_knowledge_base"} for r in rows))
    for split in ("dev", "test"):
        jsonl(ROOT / f"data/splits/omnidocbench/{split}.jsonl", (r for r in rows if r["split"] == split))
    write_json(ROOT / "data/annotations/omnidocbench/coordinate_warnings.json", warnings)
    write_json(ROOT / "data/annotations/omnidocbench/large_images.json", large_images)
    write_json(ROOT / "data/splits/omnidocbench/split_policy.json", {
        "method": "sha256_source_filename_group_60_40_v1",
        "limitation": "Filename grouping is heuristic. UUID page names may not reveal shared source documents; no strict document-disjoint generalization claim.",
        "upstream_split": "Preserve original subset attributes; local dev/test is not an official benchmark split."})
    return {"pages": len(rows), "languages": dict(Counter(r["page_attributes"]["language"] for r in rows)),
            "categories": dict(Counter(e["category_type"] for r in rows for e in r["layout_dets"])),
            "splits": dict(Counter(r["split"] for r in rows)), "coordinate_warnings": len(warnings),
            "coordinate_warning_pages": len({w['image'] for w in warnings}), "large_images": len(large_images)}


def main():
    download = json.loads((ROOT / "data/manifests/download-report.json").read_text(encoding="utf-8"))
    if not download["complete"]:
        raise SystemExit("Complete verified downloads are required before normalization")
    vidore, documents = build_vidore()
    omni = build_omni()
    jsonl(ROOT / "data/manifests/ingestion_public.jsonl", ({"document_id": d["doc_id"], "path": d["path"], "sha256": d["sha256"], "format": "pdf", "knowledge_base": "vidore_cs", "license": d["license"]} for d in documents))
    summary = {"status": "prepared_not_model_evaluated", "downloaded_files": len(download["files"]),
               "downloaded_bytes": sum(f["bytes"] for f in download["files"]),
               "vidore": vidore, "omnidocbench": omni, "api_calls": 0,
               "warnings_are_retained": True, "model_metrics": None}
    write_json(ROOT / "data/manifests/data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
