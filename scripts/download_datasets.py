"""Download complete public datasets using pinned revisions and verified hashes.

No model credentials are used. Re-running resumes partial downloads and checks
existing files. The lock file is created once; it is not silently upgraded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data/manifests/sources.lock.json"
REPOS = {
    "vidore": "vidore/vidore_v3_computer_science",
    "omnidocbench": "opendatalab/OmniDocBench",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def hashes(path):
    sha256 = hashlib.sha256()
    git = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            git.update(block)
    return sha256.hexdigest(), git.hexdigest()


def get_json(url):
    response = requests.get(url, timeout=(20, 90))
    response.raise_for_status()
    return response


def make_lock():
    if LOCK.exists():
        return json.loads(LOCK.read_text(encoding="utf-8"))
    sources = []
    for name, repo in REPOS.items():
        metadata = get_json(f"https://huggingface.co/api/datasets/{repo}").json()
        revision = metadata["sha"]
        url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000"
        entries = []
        while url:
            response = get_json(url)
            entries.extend(item for item in response.json() if item["type"] == "file")
            url = response.links.get("next", {}).get("url")
        sources.append({"name": name, "repo": repo, "revision": revision,
                        "source_url": f"https://huggingface.co/datasets/{repo}",
                        "card_license": metadata.get("cardData", {}).get("license"),
                        "files": entries})
    lock = {"schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": "Complete pinned repositories, not sampled subsets", "sources": sources}
    write_json(LOCK, lock)
    return lock


def download_file(source, entry):
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe upstream path")
    dest = ROOT / "data/raw" / source["name"] / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = entry.get("lfs", {}).get("oid")

    def verify(path):
        if not path.exists() or path.stat().st_size != entry["size"]:
            return None
        sha256, git = hashes(path)
        matches = sha256 == expected_hash if expected_hash else git == entry["oid"]
        if not matches:
            return None
        return sha256

    sha256 = verify(dest)
    if sha256:
        return {"path": str(dest.relative_to(ROOT)).replace("\\", "/"), "bytes": entry["size"], "sha256": sha256, "verified": True}
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{quote(entry['path'], safe='/')}"
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= entry["size"]:
                if verify(partial):
                    partial.replace(dest)
                    break
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Invalid range response")
                with partial.open("ab" if append else "wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            sha256 = verify(partial)
            if not sha256:
                raise ValueError("Size or upstream hash mismatch")
            partial.replace(dest)
            break
        except (requests.RequestException, OSError, ValueError) as exc:
            if attempt == 4:
                return {"path": str(dest.relative_to(ROOT)), "verified": False, "error": type(exc).__name__ + ": " + str(exc)[:160]}
            time.sleep(min(2 ** attempt, 8))
    sha256 = verify(dest)
    return {"path": str(dest.relative_to(ROOT)).replace("\\", "/"), "bytes": entry["size"], "sha256": sha256, "verified": bool(sha256)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    lock = make_lock()
    total = sum(f["size"] for s in lock["sources"] for f in s["files"])
    print(json.dumps({"bytes": total, "sources": [{"name": s["name"], "revision": s["revision"], "files": len(s["files"]), "bytes": sum(f["size"] for f in s["files"])} for s in lock["sources"]]}, indent=2), flush=True)
    if args.plan_only:
        return
    if shutil.disk_usage(ROOT).free < total * 2 + 1024 ** 3:
        raise SystemExit("Insufficient free space for download and derived data")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_file, s, f) for s in lock["sources"] for f in s["files"]]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 50 == 0 or not result["verified"] or len(results) == len(futures):
                print(f"verified {sum(r['verified'] for r in results)}/{len(futures)}; completed {len(results)}", flush=True)
                write_json(ROOT / "data/manifests/download-report.json", {"complete": len(results) == len(futures) and all(r["verified"] for r in results), "files": sorted(results, key=lambda r: r["path"])})
    if not all(r["verified"] for r in results):
        raise SystemExit("Some downloads failed; rerun to resume. See download-report.json")


if __name__ == "__main__":
    main()
