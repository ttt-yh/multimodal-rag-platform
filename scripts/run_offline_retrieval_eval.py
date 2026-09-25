"""Run reviewed retrieval evaluation over saved reports; no model calls are made."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_rag.application.offline_evaluation import load_gold_cases, score_retrieval


def main() -> int:
    parser = argparse.ArgumentParser(description="计算已审核评测集的 Recall/Precision/MRR")
    parser.add_argument("--gold", default="evals/datasets/retrieval_gold_v1.jsonl")
    parser.add_argument("--reports", default="evals/results/retrieval")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        cases = load_gold_cases(root / args.gold)
    except ValueError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    reports: list[dict] = []
    for path in sorted((root / args.reports).glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if isinstance(value, dict):
            reports.append(value)
    try:
        result = score_retrieval(cases, reports, k=args.k)
    except ValueError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    target = root / "evals" / "results" / "offline" / "retrieval_quality.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
