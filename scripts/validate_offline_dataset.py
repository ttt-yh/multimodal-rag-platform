"""Validate the offline gold set without calling any model service."""
from __future__ import annotations

import json
from pathlib import Path

from multimodal_rag.application.offline_evaluation import summarize_gold_dataset


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "evals" / "datasets" / "retrieval_gold_v1.jsonl"
    summary = summarize_gold_dataset(source)
    target = root / "evals" / "results" / "offline" / "retrieval_v1_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
