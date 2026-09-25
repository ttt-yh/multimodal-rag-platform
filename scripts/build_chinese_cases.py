"""Build 180 source-grounded candidate annotations, not a fabricated gold set.

No LLM is called. Extractive text/table answers retain exact evidence. Visual
questions have source context but NO invented visual answer; they require image
review. Human approval is a separate append-only review step, never implicit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import re

from download_datasets import ROOT, write_json
from prepare_datasets import jsonl, stable_id
from prepare_chinese import clean


def rows(path):
    return [json.loads(x) for x in (ROOT / path).read_text(encoding="utf-8").splitlines()]


PRIORITY = ["overview", "architecture", "storage", "computing", "scheduling", "hardware", "configuration", "troubleshoot", "monitor", "dashboard", "backup", "restore", "deployment", "maintain", "transaction"]


def rank(item):
    path = item["source_path"]
    return (0 if any(p in path for p in PRIORITY) else 1, 1 if path.startswith("ai/") else 0, stable_id(item.get("evidence_id", path)))


def evidence(section):
    return {k: section[k] for k in ["evidence_id", "document_id", "source_path", "line_start", "line_end", "heading_path", "text_sha256"]}


def content(section):
    return re.sub(r"^#{1,6} .+\n?", "", section["text"], count=1).strip()


def select(pool, count, used, max_per_doc=2):
    selected, per_doc = [], Counter()
    for candidate in sorted(pool, key=rank):
        identity = candidate.get("candidate_key", candidate.get("evidence_id", candidate["source_path"]))
        doc = candidate["document_id"]
        if identity in used or per_doc[doc] >= max_per_doc:
            continue
        selected.append(candidate)
        used.add(identity)
        per_doc[doc] += 1
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} valid candidates for requested {count}")


def main():
    docs = rows("data/manifests/ingestion_chinese_md.jsonl")
    by_doc = {d["document_id"]: d for d in docs}
    sections = rows("data/annotations/tidb_zh/sections.jsonl")
    assets = rows("data/annotations/tidb_zh/assets.jsonl")
    result, used = [], set()

    def add(kind, section, question, answer, required=None, extra=None):
        ev = section if isinstance(section, list) else [section]
        assert len({s["split"] for s in ev}) == 1
        item = {"case_id": "zh_" + kind + "_" + stable_id([question, [s["evidence_id"] for s in ev]]),
                "type": kind, "language": "zh", "question": question, "reference_answer": answer,
                "answerable": True, "required_facts": required or [], "evidence": [evidence(s) for s in ev],
                "group_ids": sorted({s["group_id"] for s in ev}), "split": ev[0]["split"],
                "scope": {"knowledge_base": "tidb_zh_85", "product_version": "8.5"},
                "reference_document_ids": sorted({s["document_id"] for s in ev}),
                "source": "project_generated_from_official_chinese_docs", "annotation_method": "deterministic_evidence_first_v1",
                "review_status": "pending_human_review", "gold_eligible": False,
                "answer_status": "extractive_draft_not_semantically_reviewed"}
        if extra:
            item.update(extra)
        result.append(item)

    quotas = {"concept_procedure": (27, 13), "configuration": (20, 10), "table": (20, 10),
              "visual": (20, 10), "multi_document": (20, 10), "unanswerable_clarify": (13, 7)}
    for split_index, split in enumerate(["dev", "test"]):
        ss = [s for s in sections if s["split"] == split]
        # Preserve warnings/units within a bounded complete section; never take a
        # truncated excerpt and pretend it is a complete reference answer.
        concepts = [s for s in ss if s["heading_path"] and 160 <= len(content(s)) <= 1000
                    and not re.search(r"```|!\[|<table|\|.*\|", content(s))
                    and "默认值" not in content(s) and len(s["heading_path"][-1]) < 70]
        for s in select(concepts, quotas["concept_procedure"][split_index], used):
            d = by_doc[s["document_id"]]
            add("concept_procedure", s, f"根据《{d['title']}》，关于“{s['heading_path'][-1]}”有哪些主要说明或操作要求？", clean(content(s)))
        configs = [s for s in ss if s["heading_path"] and 50 <= len(content(s)) <= 1800 and "默认值" in content(s)]
        for s in select(configs, quotas["configuration"][split_index], used, max_per_doc=8):
            default_lines = [clean(line).lstrip("-+* ") for line in content(s).splitlines() if "默认值" in line]
            add("configuration", s, f"TiDB 8.5 文档中“{' / '.join(s['heading_path'][-2:])}”的默认值是什么，文档还说明了哪些适用条件或限制？", clean(content(s)), default_lines)
        tables = []
        for s in ss:
            lines = s["text"].splitlines()
            for i in range(len(lines) - 2):
                if "|" not in lines[i] or not re.match(r"^\s*\|?\s*:?-{3,}", lines[i+1]):
                    continue
                headers = [clean(x) for x in lines[i].strip().strip("|").split("|")]
                if not 2 <= len(headers) <= 7:
                    continue
                for j in range(i+2, len(lines)):
                    if not lines[j].strip().startswith("|"):
                        break
                    cells = [clean(x) for x in lines[j].strip().strip("|").split("|")]
                    if len(cells) != len(headers) or not cells[0] or len(cells[0]) > 100 or any(not c for c in cells):
                        continue
                    # The first-column lookup must identify one unambiguous row.
                    if sum(1 for line in lines[i+2:] if line.strip().startswith("|") and clean(line.strip().strip("|").split("|")[0]) == cells[0]) != 1:
                        continue
                    tables.append({**s, "candidate_key": s["evidence_id"] + headers[0] + cells[0], "headers": headers, "cells": cells,
                                   "table_header_line": s["line_start"]+i, "table_row_line": s["line_start"]+j})
        for s in select(tables, quotas["table"][split_index], used):
            d = by_doc[s["document_id"]]
            pairs = [f"{h}：{v}" for h, v in zip(s["headers"][1:], s["cells"][1:])]
            add("table", s, f"《{d['title']}》的表格中，“{s['headers'][0]}”为“{s['cells'][0]}”这一行的其余各列分别是什么？", "；".join(pairs), s["cells"][1:],
                {"table_evidence": {"header_line": s["table_header_line"], "row_line": s["table_row_line"], "headers": s["headers"], "cells": s["cells"]}})
        visual_pool = []
        for a in assets:
            if a["status"] != "available" or a["format"] not in {"png", "jpg", "jpeg"} or by_doc[a["document_id"]]["split"] != split:
                continue
            if re.search(r"logo|icon|password|blank|badge", a["target"], re.I) or min(a.get("width", 0), a.get("height", 0)) < 250:
                continue
            s = next((s for s in ss if s["document_id"] == a["document_id"] and s["line_start"] <= a["line"] <= s["line_end"]), None)
            if s:
                visual_pool.append({**s, "candidate_key": "visual:" + a["asset_sha256"], "asset": a})
        for s in select(visual_pool, quotas["visual"][split_index], used, max_per_doc=1):
            a, d = s["asset"], by_doc[s["document_id"]]
            label = a["alt"] or a["target"].rsplit("/", 1)[-1]
            add("visual", s, f"请查看《{d['title']}》中“{label}”这幅图，说明图中主要对象及它们的关系或变化；看不清的细节不要推测。", None, extra={
                "visual_evidence": {"asset_path": a["asset_path"], "sha256": a["asset_sha256"], "width": a["width"], "height": a["height"], "region": "whole_original_image"},
                "answer_status": "requires_original_image_review", "reference_context_not_answer": clean(content(s)),
                "review_tasks": ["改写为具体关系或数值问题", "核验答案必须依赖图像而非仅附近正文", "从原图填写答案要点；不可读内容不出题"]})
        # Explicitly paired documents, chosen within the pre-existing partition.
        # This tests evidence integration; do not label it proven multi-hop reasoning.
        source_links = defaultdict(list)
        for s in ss:
            if not s["heading_path"] or not 100 <= len(content(s)) <= 1200:
                continue
            for target in re.findall(r"\]\(/([^\s)#]+\.md)(?:#[^)]*)?\)", s["text"]):
                source_links[s["document_id"]].append((s, target))
        pair_pool = []
        by_path = {d["source_path"]: d for d in docs}
        for pairs in source_links.values():
            for s, target in pairs:
                d2 = by_path.get(target)
                if not d2 or d2["split"] != split or d2["document_id"] == s["document_id"]:
                    continue
                second = next((e for e in ss if e["document_id"] == d2["document_id"] and e["heading_path"] and 100 <= len(content(e)) <= 1000), None)
                if second:
                    pair_pool.append({**s, "candidate_key": "pair:" + stable_id([s["evidence_id"], second["evidence_id"]]), "second": second})
        for s in select(pair_pool, quotas["multi_document"][split_index], used):
            second = s["second"]
            t1, t2 = by_doc[s["document_id"]]["title"], by_doc[second["document_id"]]["title"]
            add("multi_document", [s, second], f"请结合《{t1}》的“{s['heading_path'][-1]}”与《{t2}》的“{second['heading_path'][-1]}”，分别归纳两处说明，并标注各自来源。", f"《{t1}》：{clean(content(s))}\n《{t2}》：{clean(content(second))}", extra={"reasoning_label": "multi_document_evidence_integration_not_verified_multihop"})
        scope_pool = [s for s in ss if s["heading_path"] and 100 <= len(content(s)) <= 700]
        for index, s in enumerate(select(scope_pool, quotas["unanswerable_clarify"][split_index], used, max_per_doc=1)):
            d = by_doc[s["document_id"]]
            prompts = ["我公司昨天这套集群的实际 P99 延迟是多少？", "我当前线上集群的实际节点 IP 列表是什么？", "我公司下个月的采购成交价是多少？", "我刚才那次操作对应的实际任务编号是什么？", "我当前集群今天的实时 CPU 使用率是多少？"]
            if index % 2 == 0:
                add("unanswerable_clarify", s, f"仅根据《{d['title']}》，{prompts[index % len(prompts)]}", "静态文档不提供该用户的实时业务信息，无法据此确定。", extra={"answerable": False, "expected_action": "state_missing_runtime_information", "evidence_role": "scope_only_not_answer_evidence"})
            else:
                add("unanswerable_clarify", s, f"我在参考《{d['title']}》，现在想调整配置，但没确定参数名和优化目标，应该改成多少？", "需要先明确参数名称、适用版本、环境及目标，不能直接猜测一个数值。", extra={"answerable": None, "expected_action": "ask_clarification", "evidence_role": "scope_only_not_answer_evidence"})
    # Drafts were written after inspecting the original images, not inferred
    # from captions. This is AI visual review, NOT independent human approval.
    visual_drafts = json.loads((ROOT / "data_specs/chinese_visual_drafts.json").read_text(encoding="utf-8"))
    for q in result:
        if q["type"] != "visual":
            continue
        key = q["visual_evidence"]["asset_path"].split("/media/", 1)[-1]
        question, answer, facts = visual_drafts[key]
        document_title = by_doc[q["evidence"][0]["document_id"]]["title"]
        q.update(question=f"依据《{document_title}》中的原始截图或示意图，{question}",
                 reference_answer=answer, required_facts=facts,
                 answer_status="assistant_visual_reviewed_draft_pending_human",
                 annotation_method="original_image_inspection_by_assistant_v1",
                 review_tasks=["人工核对问题、数值、单位与原图一致", "确认图像定位足以支撑答案", "历史截图不视为当前版本界面"])
    assert len(result) == 180
    result.sort(key=lambda x: (x["split"], x["type"], x["case_id"]))
    jsonl(ROOT / "data/annotations/tidb_zh/qa_candidates.jsonl", result)
    for split in ["dev", "test"]:
        jsonl(ROOT / f"data/splits/tidb_zh/{split}_candidates.jsonl", (q for q in result if q["split"] == split))
    # Empty by design until explicit human review is imported. Consumers must
    # not silently use candidates as a gold-standard quality benchmark.
    write_json(ROOT / "data/annotations/tidb_zh/review_contract.json", {
        "schema": {"case_id": "existing candidate ID", "status": "approved|rejected|needs_changes",
                   "reviewer": "human identity", "reviewed_at": "ISO timestamp", "question": "reviewed question",
                   "reference_answer": "reviewed answer", "required_facts": ["reviewed facts"],
                   "evidence_ids": ["existing evidence IDs"], "notes": "what was checked"},
        "rule": "Export review decisions separately. Only validated approved records may become gold. A source-grounding rule check is not human approval."})
    summary = {"total_candidates": len(result), "splits": dict(Counter(q["split"] for q in result)),
               "types": dict(Counter(q["type"] for q in result)), "gold_approved": 0,
               "visual_answers_pending": sum(q["reference_answer"] is None for q in result),
               "no_answer": sum(q["answerable"] is False for q in result),
               "needs_clarification": sum(q["answerable"] is None for q in result),
               "api_calls": 0, "model_metrics": None}
    write_json(ROOT / "data/manifests/chinese_qa_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
