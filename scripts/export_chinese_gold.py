"""Export explicitly reviewed decisions; candidates never become gold implicitly."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from download_datasets import ROOT, hashes, write_json
from prepare_datasets import jsonl
from build_chinese_cases import rows


def validate_reviews(reviews, candidates):
    by_id = {q["case_id"]:q for q in candidates}
    seen, result = set(), []
    for review in reviews:
        cid = review["case_id"]
        if cid in seen or cid not in by_id:
            raise ValueError(f"Unknown or repeated case ID: {cid}")
        seen.add(cid)
        if review["status"] not in {"approved", "rejected", "needs_changes"}:
            raise ValueError("Invalid review status")
        if review["status"] != "approved":
            continue
        if not review.get("reviewer", "").strip() or not review.get("notes", "").strip():
            raise ValueError("Named human reviewer and review notes required")
        datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        original = by_id[cid]
        allowed = {e["evidence_id"] for e in original["evidence"]}
        evidence_ids = set(review["evidence_ids"])
        if not evidence_ids or not evidence_ids <= allowed:
            raise ValueError("Review must refer to the candidate's frozen source evidence")
        if not review.get("question", "").strip() or not review.get("reference_answer", "").strip():
            raise ValueError("Question and reference answer required")
        if not isinstance(review.get("required_facts"), list) or any(not isinstance(f,str) for f in review["required_facts"]):
            raise ValueError("required_facts must be a list of strings")
        if original["answerable"] is True and not review["required_facts"]:
            raise ValueError("Answerable questions need explicit checked facts")
        result.append({**original, **{k:review[k] for k in ["question", "reference_answer", "required_facts"]},
                       "evidence":[e for e in original["evidence"] if e["evidence_id"] in evidence_ids],
                       "review_status":"human_approved", "gold_eligible":True, "answer_status":"human_reviewed",
                       "human_review":review})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--attest-human-review", action="store_true", help="Confirm these are real human decisions, not generated approvals")
    args=parser.parse_args()
    if not args.attest_human_review:
        parser.error("Explicit human-review attestation is required; automated drafts are not gold.")
    candidates=rows("data/annotations/tidb_zh/qa_candidates.jsonl")
    reviews=[json.loads(x) for x in args.reviews.read_text(encoding="utf-8").splitlines() if x.strip()]
    approved=validate_reviews(reviews,candidates)
    for split in ("dev","test"):
        jsonl(ROOT/f"data/splits/tidb_zh/{split}_gold.jsonl",(q for q in approved if q["split"]==split))
    write_json(ROOT/"data/manifests/chinese_gold_summary.json", {"approved":len(approved), "decisions_sha256":hashes(args.reviews)[0], "candidate_sha256":hashes(ROOT/"data/annotations/tidb_zh/qa_candidates.jsonl")[0]})
    print(f"Exported {len(approved)} explicitly reviewed records. No model evaluation performed.")


if __name__=="__main__":
    main()
