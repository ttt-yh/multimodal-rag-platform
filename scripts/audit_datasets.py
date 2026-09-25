"""Run offline acceptance tests and create a human-readable data report."""
from __future__ import annotations

import html
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil
import xml.etree.ElementTree as ET

from download_datasets import ROOT, hashes, write_json


def main():
    output = ROOT / "evals/results/data-readiness"
    output.mkdir(parents=True, exist_ok=True)
    run = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", f"--junitxml={output / 'tests.xml'}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (output / "tests.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    data = json.loads((ROOT / "data/manifests/data_summary.json").read_text(encoding="utf-8"))
    synthetic = json.loads((ROOT / "data/manifests/engineering_summary.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "data/splits/vidore/split_policy.json").read_text(encoding="utf-8"))
    chinese = {name: json.loads((ROOT / f"data/manifests/chinese_{name}_summary.json").read_text(encoding="utf-8")) for name in ["corpus", "qa", "pdf", "package"]}
    xml = ET.parse(output / "tests.xml")
    cases = xml.findall(".//testcase")
    failures = [c.attrib["name"] for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [c.attrib["name"] for c in cases if c.find("skipped") is not None]
    tests = {"total": len(cases), "passed": len(cases) - len(failures) - len(skipped), "failed": failures, "skipped": skipped, "exit_code": run.returncode}
    generated = []
    for subdir in ["data/derived", "data/annotations", "data/splits", "evals/cases", "output/pdf", "data_specs"]:
        for path in sorted((ROOT / subdir).rglob("*")):
            if path.is_file():
                generated.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": hashes(path)[0]})
    write_json(ROOT / "data/manifests/derived_files.json", generated)
    prior = ROOT / "data/manifests/bundle.json"
    if prior.exists() and not (ROOT / "data/manifests/bundle_v1.json").exists():
        old = json.loads(prior.read_text(encoding="utf-8"))
        if old.get("bundle_id", "").startswith("data-v1-"):
            shutil.copyfile(prior, ROOT / "data/manifests/bundle_v1.json")
    snapshot = {"schema_version": 2, "source_lock_sha256": hashes(ROOT / "data/manifests/sources.lock.json")[0],
                "chinese_source_lock_sha256": hashes(ROOT / "data/manifests/chinese_source.lock.json")[0],
                "chinese_ingestion_manifest_sha256": hashes(ROOT / "data/manifests/ingestion_chinese_md.jsonl")[0],
                "chinese_external_assets_sha256": hashes(ROOT / "data/manifests/chinese_external_assets.json")[0],
                "derived_manifest_sha256": hashes(ROOT / "data/manifests/derived_files.json")[0],
                "public_ingestion_manifest_sha256": hashes(ROOT / "data/manifests/ingestion_public.jsonl")[0],
                "engineering_ingestion_manifest_sha256": hashes(ROOT / "data/manifests/ingestion_engineering.jsonl")[0],
                "parsing_input_manifest_sha256": hashes(ROOT / "data/manifests/parsing_inputs.jsonl")[0],
                "data_environment_lock_sha256": hashes(ROOT / "requirements-data.lock.txt")[0],
                "preparation_scripts": {p.name: hashes(p)[0] for p in sorted((ROOT / "scripts").glob("*.py"))}}
    snapshot["bundle_id"] = "data-v2-" + hashes(ROOT / "data/manifests/derived_files.json")[0][:12]
    write_json(ROOT / "data/manifests/bundle.json", snapshot)
    report = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bundle": snapshot,
              "data": data, "synthetic": synthetic, "split": split, "chinese": chinese, "tests": tests,
              "status": "ready_with_documented_limitations" if not run.returncode else "failed",
              "model_metrics": None, "model_evaluation_run": False}
    write_json(output / "summary.json", report)
    warnings = [
        "中文 180 条为候选标注：文本/表格由原文抽取，30 条图像题经助手看图起草；均待独立人工审核，目前不是正式金标准。",
        "中文先按原文及共享图片分组，再选取 120 条开发、60 条测试候选；未做语义近重复检测，不宣称跨主题泛化。",
        "中文 MD 为默认主语料，PDF/TXT 是同源格式副本；同一来源不能重复入库。PDF 为项目重排版，不是官方原版扫描件。",
        "中文资源保留 1 张图片解码异常和 4 处未缓存的装饰性外链；8 张出版方外链图片已缓存但不属于 Git 固定提交，单独记录哈希。",
        f"ViDoRe 按翻译与证据页分组：开发 {data['vidore']['splits_english']['dev']} 题、测试 {data['vidore']['splits_english']['test']} 题。未标注页面不一定是不相关页面。",
        f"开发/测试证据页不重叠，但共享 {split['chapter_overlap_count']} 个章节；按章节连通分组会得到 {split['chapter_connected_component_sizes']}，不适合强行按 6:4 划分。不能宣称章节或文档独立泛化。",
        f"OmniDocBench 保留 {data['omnidocbench']['coordinate_warnings']} 条越界多边形警告，涉及 {data['omnidocbench']['coordinate_warning_pages']} 页；只对受影响区域的定位评分复核，不据此自动删除整页文字标注。",
        f"{data['omnidocbench']['large_images']} 张原图超过 5000 万像素，原图完整保留；未来处理需像素预算与坐标变换记录，不能无条件全尺寸解码。",
        "OmniDocBench 来源文件分组部分依赖文件名；UUID 页面不能确认原文档归属，因此不宣称严格文档级隔离。",
        "官方参考答案和区域标注原样保留；本轮未对全部问题进行人工语义复核。自建数据由确定性模板生成，不是生产数据或正式公开基准。",
        "这里只验收数据准备。Recall、回答正确率、Agent 成功率均尚未测量；80 条 Agent 流程契约及 8 条生命周期契约均未执行。新增 40 条 Agent 契约复用中文 QA，不是额外独立问题。",
    ]
    cards = [
        ("公开文件 / 哈希校验", f"{data['downloaded_files']} / {data['downloaded_files']}"),
        ("中文主知识库", f"{chinese['corpus']['ingest_markdown_documents']} 份 MD · {chinese['corpus']['unique_local_images']} 张图"),
        ("中文候选问答", "180 条 · 120 开发 / 60 测试 · 待人工审核"),
        ("中文 PDF 格式副本", f"{chinese['pdf']['source_documents']} 篇原文 · {sum(p['pages'] for p in chinese['pdf']['pdfs'])} 页"),
        ("英文独立基准", f"{data['vidore']['documents']} 本 · {data['vidore']['pages']} 页"),
        ("问答标注", "215 原始问题 · 1290 多语言条目"),
        ("解析评测", f"{data['omnidocbench']['pages']} 页"),
        ("中文模拟资料", "24 文档 · 60 问答"),
        ("Agent / 生命周期契约", "80 / 8 条 · 尚未执行"),
        ("数据自动化检查", f"{tests['passed']} / {tests['total']} 通过"),
        ("模型 API 调用", "0 次"),
    ]
    card_html = "".join(f"<article><small>{html.escape(k)}</small><strong>{html.escape(v)}</strong></article>" for k, v in cards)
    warning_html = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
    test_html = "".join(f"<tr><td>{html.escape(c.attrib['name'])}</td><td>{'失败' if c.attrib['name'] in failures else '跳过' if c.attrib['name'] in skipped else '通过'}</td><td>{html.escape(c.attrib.get('time', ''))} s</td></tr>" for c in cases)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>数据准备验收报告</title><style>
body{{margin:0;background:#f5f7f8;color:#182a35;font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1100px;margin:auto;padding:36px 24px}}
h1{{font-size:32px;margin-bottom:8px}}h2{{font-size:22px;margin-top:30px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}}article{{background:white;border:1px solid #dce4e8;padding:18px;border-radius:8px}}small{{display:block;color:#506776}}strong{{display:block;font-size:21px}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e0e6e9}}.notice{{border-left:4px solid #a86a15;background:#fff9ed;padding:16px}}a{{color:#096578}}code{{overflow-wrap:anywhere}}li{{margin:8px 0}}
</style><main><p>多模态 RAG 平台 / 数据阶段</p><h1>完整数据准备与校验报告</h1>
<p>数据版本：<code>{snapshot['bundle_id']}</code> · 本报告不包含模型效果成绩</p><div class="cards">{card_html}</div>
<h2>数据分工，分开使用</h2><table><tr><th>用途</th><th>数据</th><th>入口</th></tr>
<tr><td>中文技术知识主场景</td><td>TiDB 8.5 固定提交文档与图像</td><td>data/manifests/ingestion_chinese_md.jsonl</td></tr>
<tr><td>真实 PDF 入库、检索与回答</td><td>两本 OpenStax 教材及官方证据</td><td>data/manifests/ingestion_public.jsonl</td></tr>
<tr><td>独立解析质量评测</td><td>OmniDocBench 全量页面与元素标注</td><td>data/annotations/omnidocbench/pages.jsonl</td></tr>
<tr><td>工程回归与中文边界场景</td><td>虚构星桥文档、问答及工作流契约</td><td>data/manifests/ingestion_engineering.jsonl</td></tr></table>
<h2>已知限制与待复核项</h2><div class="notice"><ul>{warning_html}</ul></div>
<h2>验收检查</h2><table><tr><th>检查</th><th>结果</th><th>耗时</th></tr>{test_html}</table>
<p><a href="summary.json">机器可读汇总</a> · <a href="tests.log">测试日志</a> · <a href="../chinese-review/dev_review.html">中文开发集证据审核</a> · <a href="../../../docs/中文主语料补齐说明.md">中文补齐说明</a> · <a href="../../../docs/数据集准备与使用指南.md">数据使用指南</a></p>
<p>原始资料与预解析 OCR、评测答案严格分目录。后续入库只读取 manifest 白名单，不递归扫描整个 data 文件夹。原始文件不随代码仓库分发。</p></main></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps({"status": report["status"], "tests": tests, "bundle_id": snapshot["bundle_id"], "report": str(output / "report.html")}, ensure_ascii=False, indent=2))
    raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
