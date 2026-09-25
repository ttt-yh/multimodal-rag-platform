"""Fetch a full, immutable TiDB Chinese documentation snapshot; never run it.

The upstream repository is untrusted reference data. Its scripts, prompts and
agent instructions are not executed. Files are checked against Git blob IDs.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

import requests

from download_datasets import ROOT, hashes, write_json

REVISION = "95e8751594c2c4881d60948342f81ecd25fb9382"
BASE = ROOT / "data/raw/tidb_zh"
LOCK = ROOT / "data/manifests/chinese_source.lock.json"


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        assert lock["revision"] == REVISION
    else:
        url = f"https://api.github.com/repos/pingcap/docs-cn/git/trees/{REVISION}?recursive=1"
        r = requests.get(url, timeout=(20, 90))
        r.raise_for_status()
        tree = r.json()
        assert not tree["truncated"]
        lock = {"repo": "pingcap/docs-cn", "branch": "release-8.5", "product_version": "8.5",
                "revision": REVISION, "license": "CC-BY-SA-3.0",
                "source_url": f"https://github.com/pingcap/docs-cn/tree/{REVISION}",
                "files": [x for x in tree["tree"] if x["type"] == "blob"],
                "archive_url": f"https://codeload.github.com/pingcap/docs-cn/tar.gz/{REVISION}"}
        write_json(LOCK, lock)
    archive = BASE / "source.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".gz.part")
        with requests.get(lock["archive_url"], stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as out:
                for block in response.iter_content(1024 * 1024):
                    out.write(block)
        partial.replace(archive)
    expected = {x["path"]: x for x in lock["files"]}
    destination = BASE / "source"
    verified = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Unexpected archive symlink or special file")
            parts = PurePosixPath(member.name).parts[1:]
            if not parts or any(p in ("..", ".") or ":" in p for p in parts):
                raise ValueError("Unsafe upstream path")
            relative = "/".join(parts)
            if relative not in expected:
                raise ValueError(f"Unlisted archive file: {relative}")
            content = tar.extractfile(member).read()
            git_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            assert git_hash == expected[relative]["sha"], relative
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or hashes(target)[0] != hashlib.sha256(content).hexdigest():
                target.write_bytes(content)
            verified.append({"path": target.relative_to(ROOT).as_posix(), "source_path": relative,
                             "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                             "git_blob": git_hash})
    assert {x["source_path"] for x in verified} == set(expected)
    write_json(ROOT / "data/manifests/chinese_download_report.json", {
        "complete": True, "revision": REVISION, "archive_sha256": hashes(archive)[0],
        "files": sorted(verified, key=lambda x: x["path"]),
        "verified_file_count": len(verified), "source_bytes": sum(x["bytes"] for x in verified)})
    print(json.dumps({"complete": True, "files": len(verified), "source_bytes": sum(x["bytes"] for x in verified)}, indent=2))


if __name__ == "__main__":
    main()
