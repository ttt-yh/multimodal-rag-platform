"""Offline, escaped side-by-side report. Never calls APIs or rewrites model answers."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from run_stage0c_live import ROOT, OUT, read, write, digest


class TableRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = []
        elif tag in {'td','th'}: self.cell = ''

    def handle_data(self, data):
        if self.cell is not None: self.cell += data

    def handle_endtag(self, tag):
        if tag in {'td','th'} and self.row is not None:
            self.row.append(''.join((self.cell or '').split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def table_checks(result):
    """Ten pre-call visual reference values, with explicit table/row association."""
    tables = []
    for e in result['elements']:
        if e['element']['kind'] == 'table':
            parser = TableRows()
            parser.feed(e['element']['raw_text'])
            tables.append({r[0]:r[1:] for r in parser.rows if r})
    expected = [
        (0,'sshpass',0,'1.06及以上'),(0,'numa',0,'2.0.12及以上'),(0,'tar',0,'任意'),
        (1,'TiKV',0,'8核+'),(1,'TiKV',1,'32GB+'),(1,'TiKV',4,'3'),
        (1,'TiDB',0,'8核+'),(1,'TiDB',1,'16GB+'),
        (2,'TiDB',0,'16核+'),(2,'TiDB',1,'48GB+')]
    checks = []
    for table, row, col, value in expected:
        try: actual = tables[table][row][col]
        except (IndexError,KeyError): actual = None
        checks.append({'sample_id':'s0c_06','table_index':table,'row':row,'value_index':col,
                       'expected':value,'actual':actual,'passed':actual==value})
    return checks


def main():
    config = read(ROOT/'evals/configs/stage0c_samples.json')
    cases = read(ROOT/'evals/configs/stage0c_visual_cases.json')
    review = read(ROOT/'evals/configs/stage0c_live_review.json')
    ledger = read(OUT/'ledger.json')
    assert digest((ROOT/'evals/configs/stage0c_samples.json').read_bytes()) == ledger['fingerprint']['samples']
    assert digest((ROOT/'evals/configs/stage0c_visual_cases.json').read_bytes()) == ledger['fingerprint']['questions']
    results = {s['sample_id']:read(OUT/(s['sample_id']+'.json')) for s in config['samples']}
    warning_counts = Counter(w['code'] for r in results.values() for w in r['warnings'])
    mapped, source_total, mapped_pages = 0, 0, 0
    for s in config['samples']:
        result = results[s['sample_id']]
        assert digest((OUT/(s['sample_id']+'.zip')).read_bytes()) == result['artifact']['archive_sha256']
        seen = set()
        for row in result['elements']:
            origin = s['pages'][row['provider_page_idx']]
            e = row['element']; loc = e['source']
            valid = (e['document_id']==origin['document_id'] and e['version_id']=='ver_'+origin['source_sha256'][:24]
                     and loc['source_path']==origin['source_path'] and loc['kind']==origin['source_kind'])
            valid &= (loc['page_number']==origin['source_page'] if origin['source_kind']=='pdf'
                      else loc['image_ref']==origin['source_path'] and loc['image_width']==origin['width'] and loc['image_height']==origin['height'])
            mapped += bool(valid); source_total += 1; seen.add(row['provider_page_idx'])
        mapped_pages += len(seen)
    vision = []
    for c in cases['cases']:
        cid=c['case_id']; answer=ledger['vision'][cid]; verdict=review['vision'][cid]
        assert digest(answer['text'].encode()) == verdict['answer_sha256'], 'review must match exact answer'
        vision.append({'case_id':cid,'question':c['question'],'answer':answer['text'],
                       'expected_points':c['expected_points'],'usage':answer['usage'],**verdict})
    prompt = sum(v['usage']['prompt_tokens'] for v in vision)
    completion = sum(v['usage']['completion_tokens'] for v in vision)
    table_values = table_checks(results['s0c_06'])
    junit = ET.parse(OUT.parent/'control-tests.xml').findall('.//testcase')
    tests = {'total':len(junit),'passed':sum(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in junit)}
    source_files = sorted((ROOT/'backend/src').rglob('*.py')) + sorted((ROOT/'scripts').glob('*.py'))
    code_hash = digest(b''.join(str(p.relative_to(ROOT)).encode()+p.read_bytes() for p in source_files))
    summary = {
        'status':'live_validation_completed_with_findings','generated_at':datetime.now(timezone.utc).isoformat(),
        'source_fingerprint_sha256':code_hash,'frozen_input_fingerprint':ledger['fingerprint'],
        'parser':{'files_completed':len(results),'files_total':18,'pages_with_elements':mapped_pages,'pages_total':20,
            'source_mapping_valid':mapped,'source_mapping_total':source_total,
            'missing_image_resources':sum(r['artifact']['missing_image_references'] for r in results.values()),
            'tables':sum(e['element']['kind']=='table' for r in results.values() for e in r['elements']),
            'warning_counts':dict(warning_counts),'table_spotcheck_passed':sum(c['passed'] for c in table_values),
            'table_spotcheck_total':len(table_values),'table_spotcheck':table_values,'cer':None},
        'vision':{'completed':len(vision),'core_points_passed':sum(v['key_points_pass'] for v in vision),
            'fully_grounded_passed':sum(v['grounded_pass'] for v in vision),'unique_images':3,
            'reviewer':review['reviewer'],'human_reviewed':False,'cases':vision},
        'application_tests':tests,'http_requests_used':ledger['used'],'reserved_limits':ledger['limits'],
        'usage':{'prompt_tokens':prompt,'completion_tokens':completion,
            'vision_list_price_estimate_cny':round((prompt+completion*10)/1_000_000,6),
            'pricing_source':'https://help.aliyun.com/zh/model-studio/model-pricing',
            'actual_bill_verified':False,'parser_bill_verified':False},
        'findings':review['parser_findings'],
        'limits':['20页和8题仅为开发验证，8题共享3图，不能代表全语料或正式测试集。',
                  '参考与复核均由助手完成，待用户审核；不是独立人工金标准。',
                  '页级/原图级回映射通过不等于OCR、表格内容和阅读顺序准确。',
                  '21个空正文元素及图表数值缺失仍保留警告；不补造文字、不自动执行提取命令。',
                  '未进行检索、回答全链路或Agent评测，未启动全量入库。']}
    write(OUT/'summary.json',summary)
    write(OUT.parent/'summary.json',summary)
    esc = html.escape
    def pre(value): return '<pre>'+esc(str(value))+'</pre>'
    cards = []
    for s in config['samples']:
        sid=s['sample_id']; result=results[sid]
        imgs=''.join(f'<a href="../{esc(p.split("stage0c/")[1])}"><img loading="lazy" src="../{esc(p.split("stage0c/")[1])}" alt="{sid} original page"></a>' for p in s['previews'])
        texts=''.join('<h4>'+esc(f"{i}: {e['provider_type']} | original {e['element']['source'].get('page_number') or 'image'} | index={e['index_eligible']}")+'</h4>'+pre(e['element']['raw_text']) for i,e in enumerate(result['elements']))
        cards.append(f'<details id="{sid}"><summary>{sid} · {esc(s["category"])} · {len(result["elements"])}元素 · {len(result["warnings"])}警告</summary><p><a href="{sid}.json">归一化JSON</a> · <a href="{sid}.zip">原始结果包</a></p>{pre(result["warnings"])}<div class="pair"><div>{imgs}</div><div>{texts}</div></div></details>')
    vcards=[]
    for v,c in zip(vision,cases['cases']):
        s=next(s for s in config['samples'] if s['sample_id']==c['sample_id'])
        pic='../'+s['previews'][c['preview_index']].split('stage0c/')[1]
        vcards.append(f'<details><summary>{v["case_id"]} · 核心要点通过 · 完整回答{"通过" if v["grounded_pass"] else "存在问题"}</summary><div class="pair"><div><img loading="lazy" src="{pic}" alt="原页"></div><div><h4>{esc(v["question"])}</h4><p>预置参考：{esc("；".join(v["expected_points"]))}</p>{pre(v["answer"])}<p class="note">复核：{esc(v["note"])}</p></div></div></details>')
    findings=''.join(f'<li><b>{f["sample_id"]}</b>：{esc(f["finding"])}<br>处理：{esc(f["action"])}</li>' for f in summary['findings'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0C真实解析与视觉验证</title>
<style>body{{font:16px/1.65 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#172b38;background:#f7f9fa}}h1,h2{{color:#164b56}}.metrics{{display:flex;gap:12px;flex-wrap:wrap}}.metrics p,details,.note{{background:white;border:1px solid #d8e0e5;border-radius:8px;padding:14px}}.metrics b{{font-size:24px;display:block}}summary{{cursor:pointer;font-weight:600}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}img{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:#edf2f4;padding:12px}}a{{color:#086987}}@media(max-width:750px){{.pair{{grid-template-columns:1fr}}}}</style>
<h1>0C：真实解析与视觉验证</h1><p>已完成本批调用，发现问题已保留。结果为开发样本验证，不是全量准确率。</p>
<div class="metrics"><p><b>{len(results)} / {len(config['samples'])}</b>PDF解析完成 · {mapped_pages}页</p><p><b>{mapped} / {source_total}</b>元素来源映射</p><p><b>{summary['parser']['table_spotcheck_passed']} / {len(table_values)}</b>单页表格关键值抽查</p><p><b>{summary['vision']['core_points_passed']} / {len(vision)} → {summary['vision']['fully_grounded_passed']} / {len(vision)}</b>核心要点 → 完整回答证据一致性</p></div>
<p class="note">模型：MinerU vlm + qwen3-vl-plus；视觉8题共用3张图。助手复核，用户未审核。应用测试 {tests['passed']}/{tests['total']}。</p>
<h2>调用与预算</h2><p>解析HTTP {ledger['used']['parser']}/60（提交1、上传18、批次查询1、下载18）；视觉 {ledger['used']['vision']}/8，无重试。本批解析20页。视觉输入 {prompt} Token，输出 {completion} Token；按北京地域公示单价估算 ¥{summary['usage']['vision_list_price_estimate_cny']:.6f}，低于¥1预留。该数值不是账单，免费额度、折扣和解析服务实际账单未核验。</p>
<p><a href="https://help.aliyun.com/zh/model-studio/model-pricing">模型价格来源</a> · <a href="https://mineru.net/doc/docs/index_en/">解析API契约</a> · <a href="summary.json">完整指标JSON</a> · <a href="ledger.json">脱敏调用台账</a> · <a href="../control-tests.xml">测试JUnit</a></p>
<h2>发现的问题与处理边界</h2><ul>{findings}</ul><p>归一化器已修复chart标题遗漏，保留图像资源；header不再一律过滤，避免丢失截图控件；空元素不进入候选索引。21个空正文警告、2个图表需视觉读取警告仍然保留。</p>
<h2>原页与解析结果（18份）</h2>{''.join(cards)}<h2>视觉回答与逐条复核（8题）</h2>{''.join(vcards)}
<h2>下一步</h2><p>0D阶段验收：将解析缺失与视觉补写纳入质量门禁，确认需复核状态和页级引用边界，再进入多模态入库。不要重跑本批视觉题来挑选更好成绩。</p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    (OUT.parent/'report.html').write_text(page.replace('href="../','href="').replace('src="../','src="').replace('href="s0c_','href="live/s0c_').replace('href="ledger.json"','href="live/ledger.json"'),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['status','application_tests','http_requests_used','usage']},ensure_ascii=False))


if __name__=='__main__': main()
