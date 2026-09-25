"""汇总已保存的阶段4A问答结果，不调用任何模型。"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "evals/results/stage4-qa"
    reports = []
    for path in sorted(report_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if item.get("case_id") and item.get("question"):
            reports.append(item)
    answerable = [item for item in reports if item.get("answerable") is True and "checks" in item]
    unanswerable = [item for item in reports if item.get("answerable") is not True and "checks" in item]
    summary = {
        "status": "completed" if reports and all("checks" in item for item in reports) else "partial_failure",
        "sample_count": len(reports), "success_count": sum("checks" in item for item in reports),
        "failure_count": sum("checks" not in item for item in reports),
        "answerable_count": len(answerable), "unanswerable_count": len(unanswerable),
        "citation_recall_average": round(sum(item["checks"]["citation_recall"] for item in answerable) / len(answerable), 4) if answerable else None,
        "unanswerable_refusal_rate": round(sum(bool(item["checks"]["refusal_check"]) for item in unanswerable) / len(unanswerable), 4) if unanswerable else None,
        "external_api_calls": sum(item.get("result", {}).get("external_calls", 0) for item in reports),
        "reports_dir": str(report_dir),
    }
    target = root / "evals/results/stage4/qa_eval_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
