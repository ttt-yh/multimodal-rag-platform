"""Cache only referenced publisher-hosted images; no arbitrary URL crawling."""
import hashlib
import json
from urllib.parse import urlsplit

from PIL import Image
import requests

from download_datasets import ROOT, hashes, write_json


def main():
    refs = [json.loads(x) for x in (ROOT / "data/annotations/tidb_zh/assets.jsonl").read_text(encoding="utf-8").splitlines()]
    dest = ROOT / "data/raw/tidb_zh/external_assets"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data/manifests/chinese_external_assets.json"
    known = {x["url"]: x for x in json.loads(manifest.read_text(encoding="utf-8"))} if manifest.exists() else {}
    for url in sorted({r["target"] for r in refs if urlsplit(r["target"]).hostname == "docs-download.pingcap.com"}):
        if url in known and (ROOT / known[url]["path"]).is_file():
            assert hashes(ROOT / known[url]["path"])[0] == known[url]["sha256"]
            continue
        response = requests.get(url, timeout=(20, 90), allow_redirects=False)
        response.raise_for_status()
        if response.status_code != 200 or len(response.content) > 20 * 1024 * 1024:
            raise ValueError("Unexpected publisher image response")
        path = dest / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".png")
        path.write_bytes(response.content)
        with Image.open(path) as im:
            width, height = im.size
            im.verify()
        known[url] = {"url": url, "path": path.relative_to(ROOT).as_posix(), "sha256": hashes(path)[0],
                      "bytes": len(response.content), "width": width, "height": height,
                      "source_revision": "external_url_not_git_versioned", "license": "see_parent_document_and_publisher_terms"}
        write_json(manifest, sorted(known.values(), key=lambda x: x["url"]))
    print(f"Cached {len(known)} referenced publisher images; decorative third-party badges are not fetched.")


if __name__ == "__main__":
    main()
