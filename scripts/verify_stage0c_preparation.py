"""0C离线样本验收：哈希、开发集隔离、页映射和六份文本来源回查。

只校验既定样本，不选择更容易通过的样例，不读取模型凭证或调用API。
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

from multimodal_rag.core.models import Document, DocumentVersion
from multimodal_rag.infrastructure.text_parser import parse_text
from verify_stage0a import run_tests

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(x) for x in (ROOT/path).read_text(encoding='utf-8').splitlines() if x.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config_path=ROOT/'evals/configs/stage0c_samples.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    output=ROOT/'evals/results/stage0c'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    checks={}
    hash_cache={}
    def valid_file(path,expected):
        resolved=(ROOT/path).resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.is_file(): return False
        if path not in hash_cache: hash_cache[path]=digest(resolved)
        return hash_cache[path]==expected
    samples=config['samples']
    checks['20_pages_18_files']=sum(x['page_count'] for x in samples)==20 and len(samples)==18
    checks['unique_sample_ids']=len({x['sample_id'] for x in samples})==18
    dev_omni={x['case_id'] for x in rows('data/splits/omnidocbench/dev.jsonl')}
    dev_zh={x['document_id'] for x in rows('data/manifests/ingestion_chinese_md.jsonl') if x['split']=='dev'}
    en_dev={(e['document_id'],e['page_number']) for x in rows('data/splits/vidore/dev_all_languages.jsonl') for e in x['evidence']}
    en_test={(e['document_id'],e['page_number']) for x in rows('data/splits/vidore/test_all_languages.jsonl') for e in x['evidence']}
    categories=Counter()
    for sample in samples:
        key=sample['sample_id']
        checks[key+'_upload_hash']=valid_file(sample['input_path'],sample['input_sha256'])
        checks[key+'_page_count']=len(sample['pages'])==sample['page_count']==len(sample['previews'])
        for i,page in enumerate(sample['pages']):
            categories[page['category']]+=1
            origin=(page['document_id'],page['source_page'])
            member=(page['document_id'] in dev_omni if page['source_kind']=='image' else
                    origin in en_dev and origin not in en_test if page['language']=='en' else page['document_id'] in dev_zh)
            checks[f'{key}_{i}_dev_source']=page['split']=='dev' and member and valid_file(page['source_path'],page['source_sha256'])
    checks['category_coverage']=sorted(categories.values())==[4]*5
    text_results=[]
    for entry in config['text_samples']:
        key=entry['document_id']+'_'+entry['format']
        checks[key+'_source']=entry['split']=='dev' and entry['document_id'] in dev_zh and valid_file(entry['path'],entry['sha256'])
        text=(ROOT/entry['path']).read_text(encoding='utf-8')
        doc=Document(document_id=entry['document_id'],title=entry['title'],knowledge_base='stage0c_dev',
            source_path=entry['path'],source_family=entry['source_family'],format=entry['format'],license=entry['license'])
        version=DocumentVersion(version_id='ver_'+entry['sha256'][:24],document_id=entry['document_id'],content_sha256=entry['sha256'],source_revision='stage0c-frozen')
        elements,warnings=parse_text(doc,version,text)
        lines=text.splitlines(keepends=True)
        checks[key+'_line_roundtrip']=bool(elements) and all(e.raw_text==''.join(lines[e.source.line_start-1:e.source.line_end]) for e in elements)
        data={'document':doc.model_dump(),'elements':[e.model_dump() for e in elements],'warnings':warnings}
        (output/(key+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        text_results.append({'id':key,'elements':len(elements),'source_roundtrip':checks[key+'_line_roundtrip']})
    checks['six_texts']=len(text_results)==6
    visual=json.loads((ROOT/'evals/configs/stage0c_visual_cases.json').read_text(encoding='utf-8'))
    checks['eight_visual_questions']=len(visual['cases'])==8 and len({c['sample_id'] for c in visual['cases']})==3
    checks['visual_inputs_exist']=all(c['sample_id'] in {s['sample_id'] for s in samples} for c in visual['cases'])
    tests=run_tests(Path(sys.executable),'backend/tests','app-tests',output,isolated_temp=True)
    passed=all(checks.values()) and tests['status']=='passed'
    summary={'status':'offline_checks_passed' if passed else 'failed','checks':checks,'checks_passed':sum(checks.values()),
        'checks_total':len(checks),'application_tests':tests,'text_results':text_results,'categories':dict(categories),
        'external_api_calls':0,'parser_quality_metrics':None,'visual_quality_metrics':None,
        'sample_manifest_sha256':digest(config_path),'visual_cases_sha256':digest(ROOT/'evals/configs/stage0c_visual_cases.json'),
        'reference_status':'assistant_reviewed_human_pending','unique_visual_images':3,
        'limitations':['未执行新的解析/视觉请求，不能宣称OCR或视觉准确率通过。','PDF框坐标暂不猜测量纲，归一化器保留原始bbox并明确标为页级定位。','20页为开发验证样本，非全量语料成绩；8道视觉题共用3张图，非8张独立图片。']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    latest=output.parent
    # Once live validation exists, an offline rerun must not replace its public
    # entry point with "live pending" or imply that no real calls have happened.
    has_live = (latest/'live/ledger.json').exists()
    latest_summary = latest/('offline-summary.json' if has_live else 'summary.json')
    latest_summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=''.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>' for k,v in checks.items())
    page=f'''<!doctype html><meta charset="utf-8"><title>0C离线准备验收</title><style>body{{font:16px/1.7 system-ui;max-width:1050px;margin:40px auto}}td{{border:1px solid #ddd;padding:6px}}</style><h1>0C：{summary['status']}</h1><p>样本检查 {sum(checks.values())}/{len(checks)}；应用测试 {tests['passed']}/{tests['total']}。外部API调用0次。</p><p><a href="preparation/report.html">固定样本原页预览</a> · <a href="{output.name}/summary.json">本轮JSON报告</a> · <a href="{output.name}/app-tests.xml">JUnit</a></p><p>已固定20页、6份文本、8道视觉题（3张图）；尚无真实解析质量分数。</p><table>{cards}</table>'''
    report_path = latest/('offline-report.html' if has_live else 'report.html')
    report_path.write_text(page,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['status','checks_passed','checks_total','external_api_calls']}|{'application_tests':tests['passed'],'report':str(report_path)},ensure_ascii=False))
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
