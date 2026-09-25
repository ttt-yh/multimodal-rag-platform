"""Index source evidence, assets and pre-question document-family splits.

This is data preparation, not the application's parser or retrieval index.
No downloaded scripts are executed and no reference answers enter ingestion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from PIL import Image

from download_chinese import BASE, REVISION
from download_datasets import ROOT, hashes, write_json
from prepare_datasets import jsonl, rel, stable_id

SOURCE = BASE / "source"
EXCLUDED_DIRS = {".github", ".agents", "resources", "scripts", "releases", "templates"}
IMAGE_RE = re.compile(r'!\[([^\]\r\n]*)\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+["\'][^)]*)?\s*\)|<img\b[^>]*?src=["\']([^"\']+)["\'][^>]*>', re.I)


def eligible(path):
    parts = path.relative_to(SOURCE).parts
    if any(p in EXCLUDED_DIRS or p.startswith(".") for p in parts[:-1]):
        return False
    return not (path.name.startswith(("TOC", "_")) or path.name in {"README.md", "CONTRIBUTING.md", "credits.md", "AGENTS.md", "CLAUDE.md"})


def title_of(text, fallback):
    match = re.search(r"^title:\s*(.+)$", text, re.M)
    if not match:
        match = re.search(r"^#\s+(.+)$", text, re.M)
    return match.group(1).strip().strip('"') if match else fallback


def clean(text):
    # Angle brackets in code/logs (e.g. `<unknown>`) are data, not HTML.
    protected = []
    def protect(match):
        protected.append(match.group(0))
        return f"CODEPLACEHOLDER{len(protected)-1}END"
    text = re.sub(r"(`+)([\s\S]*?)\1", protect, text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").strip()
    for i, original in enumerate(protected):
        text = text.replace(f"CODEPLACEHOLDER{i}END", original)
    return text


def sections(path, text):
    lines = text.splitlines()
    start, headings, fence, front = 0, [], None, bool(lines and lines[0] == "---")
    blocks = []
    for i, line in enumerate(lines):
        if front:
            if i and line == "---":
                front = False
                start = i + 1
            continue
        mark = re.match(r"^\s*(`{3,}|~{3,})", line)
        if mark:
            if fence is None:
                fence = mark.group(1)[0]
            elif mark.group(1)[0] == fence:
                fence = None
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line) if not fence else None
        if heading:
            if i > start:
                blocks.append((start, i, list(headings)))
            level, name = len(heading.group(1)), clean(heading.group(2))
            headings = [(n, t) for n, t in headings if n < level] + [(level, name)]
            start = i
    if start < len(lines):
        blocks.append((start, len(lines), headings))
    result = []
    for start, end, heads in blocks:
        content = "\n".join(lines[start:end]).strip()
        if not content:
            continue
        result.append({"evidence_id": "zh_sec_" + stable_id([path, start + 1, content]),
                       "source_path": path, "line_start": start + 1, "line_end": end,
                       "heading_path": [t for _, t in heads], "text": content,
                       "text_sha256": hashlib.sha256(content.encode()).hexdigest()})
    return result


def resolve_asset(source_path, target):
    target = target.strip("<>")
    url = urlsplit(target)
    if url.scheme or url.netloc:
        return None, "external_reference"
    path = (SOURCE / unquote(url.path).lstrip("/")) if url.path.startswith("/") else (SOURCE / source_path).parent / unquote(url.path)
    path = path.resolve()
    if not path.is_relative_to(SOURCE.resolve()):
        return None, "unsafe_path"
    return path, "available" if path.is_file() else "missing_local_asset"


def main():
    download = json.loads((ROOT / "data/manifests/chinese_download_report.json").read_text(encoding="utf-8"))
    assert download["complete"]
    docs, evidence, refs, exclusions = [], [], [], []
    external_file = ROOT / "data/manifests/chinese_external_assets.json"
    external = {x["url"]: x for x in json.loads(external_file.read_text(encoding="utf-8"))} if external_file.exists() else {}
    for path in sorted(SOURCE.rglob("*.md")):
        source_path = path.relative_to(SOURCE).as_posix()
        if not eligible(path):
            exclusions.append({"source_path": source_path, "reason": "navigation_release_or_repository_support_not_primary_product_corpus"})
            continue
        text = path.read_text(encoding="utf-8")
        doc_id = "tidb85_" + stable_id(source_path)
        docs.append({"document_id": doc_id, "source_path": source_path, "path": rel(path),
                     "title": title_of(text, path.stem), "sha256": hashes(path)[0], "format": "md",
                     "product_version": "8.5", "source_revision": REVISION, "language": "zh",
                     "knowledge_base": "tidb_zh_85", "source_family": doc_id,
                     "license": "CC-BY-SA-3.0", "source_url": f"https://github.com/pingcap/docs-cn/blob/{REVISION}/{source_path}"})
        for section in sections(source_path, text):
            section["document_id"] = doc_id
            evidence.append(section)
        for match in IMAGE_RE.finditer(text):
            alt, target = match.group(1) or "", match.group(2) or match.group(3)
            asset, status = resolve_asset(source_path, target)
            externally_cached = target in external
            if externally_cached:
                asset, status = ROOT / external[target]["path"], "available"
            record = {"document_id": doc_id, "source_path": source_path, "line": text[:match.start()].count("\n") + 1,
                      "alt": alt, "target": target, "status": status}
            if externally_cached:
                record["provenance"] = "publisher_url_snapshot_not_commit_versioned"
            if status == "available":
                record.update({"asset_path": rel(asset), "asset_sha256": hashes(asset)[0], "format": asset.suffix.lower().lstrip(".")})
                if asset.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                    try:
                        with Image.open(asset) as im:
                            record["width"], record["height"] = im.size
                            im.verify()
                        record["image_readable"] = True
                    except (OSError, ValueError, Image.DecompressionBombError) as exc:
                        record["status"] = "image_decode_warning"
                        record["image_readable"] = False
                        record["error"] = type(exc).__name__ + ": " + str(exc)
            refs.append(record)
    # Group identical documents and shared non-decorative images before any QA
    # generation. This is not transitive grouping of all hyperlinks (which would
    # connect most of a technical manual into a single component).
    parent = {d["document_id"]: d["document_id"] for d in docs}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owners = {}
    for doc in docs:
        key = "doc:" + doc["sha256"]
        if key in owners:
            parent[find(doc["document_id"])] = find(owners[key])
        else:
            owners[key] = doc["document_id"]
    for record in refs:
        if record["status"] != "available" or re.search(r"logo|icon|contribution-map", record["target"], re.I):
            continue
        key = "image:" + record["asset_sha256"]
        if key in owners:
            parent[find(record["document_id"])] = find(owners[key])
        else:
            owners[key] = record["document_id"]
    components = defaultdict(list)
    for doc in docs:
        components[find(doc["document_id"])].append(doc["document_id"])
    assignment = {}
    for members in components.values():
        group = "zh_group_" + stable_id(sorted(members))
        split = "dev" if int(stable_id(group), 16) % 3 != 0 else "test"
        for member in members:
            assignment[member] = (group, split)
    for doc in docs:
        doc["group_id"], doc["split"] = assignment[doc["document_id"]]
    for section in evidence:
        section["group_id"], section["split"] = assignment[section["document_id"]]
    jsonl(ROOT / "data/manifests/ingestion_chinese_md.jsonl", docs)
    jsonl(ROOT / "data/annotations/tidb_zh/sections.jsonl", evidence)
    jsonl(ROOT / "data/annotations/tidb_zh/assets.jsonl", refs)
    write_json(ROOT / "data/annotations/tidb_zh/exclusions.json", exclusions)
    write_json(ROOT / "data/annotations/tidb_zh/resource_warnings.json", [r for r in refs if r["status"] != "available"])
    jsonl(ROOT / "data/splits/tidb_zh/document_groups.jsonl", ({k: d[k] for k in ["document_id", "source_path", "group_id", "split", "sha256"]} for d in docs))
    write_json(ROOT / "data/splits/tidb_zh/policy.json", {
        "version": "document_content_and_shared_image_group_hash_v1", "created_before_qa": True,
        "method": "Group exact document duplicates and shared non-decorative local images; stable SHA256 hash modulo 3 gives nominal 2:1 dev:test.",
        "limitations": "Not semantic deduplication or topic-independent generalization; ordinary cross-links do not force union. Evidence pairs for cross-document questions must be in the same split.",
        "group_count": len(components), "largest_group_documents": max(map(len, components.values()))})
    summary = {"source_files": len(download["files"]), "raw_markdown_files": len(docs) + len(exclusions),
               "ingest_markdown_documents": len(docs), "excluded_support_documents": len(exclusions),
               "sections": len(evidence), "image_references": len(refs), "asset_status": dict(Counter(r["status"] for r in refs)),
               "unique_local_images": len({r["asset_sha256"] for r in refs if r["status"] == "available"}),
               "document_splits": dict(Counter(d["split"] for d in docs)),
               "revision": REVISION, "api_calls": 0}
    write_json(ROOT / "data/manifests/chinese_corpus_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
