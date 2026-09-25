"""0D stage gate: current offline tests + read-only DB + historical API evidence.

No model/parser calls, database migrations, bulk ingestion, or old-project access.
Each run freezes its own contracts, input/evidence hashes and results.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from multimodal_rag.api.app import create_app
from multimodal_rag.core.models import Document, DocumentVersion, Element, IngestionJob
from multimodal_rag.core.quality import assess_parser_result, POLICY_VERSION
from multimodal_rag.infrastructure.database import connection, MIGRATIONS
from multimodal_rag.infrastructure.parser_normalization import normalize_archive
from multimodal_rag.infrastructure.settings import Settings, load_settings
from verify_stage0a import ROOT, run_tests, live_smoke, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def database_check(settings):
    try:
        with connection(settings, read_only=True) as conn:
            identity=conn.execute("SELECT current_database(),current_user,current_setting('server_version')").fetchone()
            tables={r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='mrag'").fetchall()}
            migrations=conn.execute('SELECT name,sha256 FROM mrag.schema_migrations ORDER BY name').fetchall()
        expected={p.name:sha(p) for p in MIGRATIONS.glob('*.sql')}
        checks={'tables_present':{'documents','document_versions','elements','ingestion_jobs','schema_migrations'}<=tables,
                'migration_hashes_match':dict(migrations)==expected}
        return {'status':'passed' if all(checks.values()) else 'failed','checks':checks,
                'database':identity[0],'role':identity[1],'server_version':identity[2],'read_only':True}
    except Exception as exc:
        return {'status':'failed','error_code':getattr(exc,'code',type(exc).__name__),'read_only':True}


def evidence_check(config, settings):
    paths=[ROOT/'evals/configs/stage0c_samples.json',ROOT/'evals/configs/stage0c_visual_cases.json',
           ROOT/'evals/configs/stage0c_live_review.json',ROOT/'evals/results/stage0c/live/ledger.json',
           ROOT/'evals/results/stage0b-live/summary.json']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    samples,questions,review,ledger,stage0b=[read(p) for p in paths]
    checks={'frozen_samples':sha(paths[0])==config['samples_manifest_sha256']==ledger['fingerprint']['samples'],
            'frozen_questions':sha(paths[1])==config['visual_cases_sha256']==ledger['fingerprint']['questions'],
            'saved_model_identity':ledger['fingerprint']['parser_model']==settings.parser_model and
                ledger['fingerprint']['vision_model']==settings.vision_model,
            'parser_budget':ledger['used']['parser']<=60,'vision_budget':ledger['used']['vision']<=8,
            'policy_version':POLICY_VERSION==config['quality_policy'],
            'historical_stage0b':stage0b['status']=='verified_existing_records' and all(stage0b['checks'].values())}
    # Recheck original call evidence instead of accepting the aggregate status alone.
    for name,expected in stage0b['source_sha256'].items():
        p=(ROOT/name).resolve()
        checks['historical_hash:'+name]=p.is_relative_to(ROOT) and p.is_file() and sha(p)==expected
        if checks['historical_hash:'+name]: hashes[name]=expected
    for service in stage0b['services']:
        if service['service']!='parser':
            checks['historical_model:'+service['service']]=service['model']==getattr(settings,service['service']+'_model')
        if service['service']=='embedding':
            original=read(ROOT/service['source_report'])
            checks['embedding_dimensions']=original['dimensions']==settings.embedding_dimensions
    quality={}; warning_counts=Counter(); source_count=0; known_origins={}
    for s in samples['samples']:
        sid=s['sample_id']; blob=ROOT/'evals/results/stage0c/live'/(sid+'.zip')
        checks[sid+'_input_hash']=sha(ROOT/s['input_path'])==s['input_sha256']
        for origin in s['pages']:
            name=origin['source_path']
            if name not in known_origins: known_origins[name]=sha(ROOT/name)
            checks[sid+'_source_'+str(origin['source_page'])]=known_origins[name]==origin['source_sha256'] and origin['split']=='dev'
        result=normalize_archive(blob.read_bytes(),s)  # Offline regeneration of saved raw provider response.
        hashes[blob.relative_to(ROOT).as_posix()]=sha(blob)
        saved=read(blob.with_suffix('.json'))
        checks[sid+'_normalized_reproducible']=result==saved
        checks[sid+'_all_pages_mapped']={r['provider_page_idx'] for r in result['elements']}==set(range(s['page_count']))
        checks[sid+'_raw_matches_ledger']=sha(blob)==ledger['samples'][sid]['artifact']['archive_sha256']
        source_count+=len(result['elements'])
        warning_counts.update(w['code'] for w in result['warnings'])
        quality[sid]=assess_parser_result(result,tuple(config['known_content_issues'].get(sid,[])))
    checks['source_contracts_valid']=all(q['status']!='blocked' for q in quality.values())
    checks['known_issues_not_approved']=all(quality[s]['status']=='needs_review' for s in config['known_content_issues'])
    by_id={s['sample_id']:s for s in samples['samples']}
    for case in questions['cases']:
        cid=case['case_id']; response=ledger['vision'][cid]
        image=ROOT/by_id[case['sample_id']]['previews'][case['preview_index']]
        checks[cid+'_image_and_answer_hash']=(sha(image)==response['image_sha256']==ledger['fingerprint']['images'][cid]
            and hashlib.sha256(response['text'].encode()).hexdigest()==review['vision'][cid]['answer_sha256'])
    counts=Counter(r['service'] for r in ledger['records'])
    checks['budget_matches_request_records']=dict(counts)==ledger['used']
    return {'status':'passed' if all(checks.values()) else 'failed','checks':checks,
            'source_hashes':hashes,'quality_assessments':quality,'quality_counts':dict(Counter(q['status'] for q in quality.values())),
            'warning_counts':dict(warning_counts),'mapped_elements':source_count,
            'historical_calls':ledger['used'],'new_external_api_calls':0,
            'historical_call_time_range':{'created_at':ledger['created_at'],'updated_at':ledger['updated_at']},
            'human_reviewed':False}


def main():
    config=read(ROOT/'evals/configs/stage0d.json')
    assert config['new_external_api_calls_allowed'] is False
    settings=load_settings()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base=ROOT/'evals/results/stage0d'; output=base/run_id
    output.mkdir(parents=True)
    app_tests=run_tests(Path(sys.executable),'backend/tests','app-tests',output,isolated_temp=True)
    data_python=ROOT/('.venv-data/Scripts/python.exe' if os.name=='nt' else '.venv-data/bin/python')
    data_tests=run_tests(data_python,'tests','data-tests',output,isolated_temp=True)
    smoke=live_smoke(output,config['demo_document_id'])
    db=database_check(settings)
    try: evidence=evidence_check(config,settings)
    except Exception as exc:
        evidence={'status':'failed','error_code':getattr(exc,'code',type(exc).__name__)}
    # Schema and OpenAPI snapshots contain no user configuration or credentials.
    contracts={'models':{cls.__name__:cls.model_json_schema() for cls in [Document,DocumentVersion,Element,IngestionJob]},
               'openapi':create_app(Settings(_env_file=None)).openapi(),
               'quality_policy':POLICY_VERSION,'stage1_constraints':config['stage1_release_constraints']}
    write_json(output/'contracts.json',contracts)
    write_json(output/'config.json',config)
    tracked=sorted(set((ROOT/'backend').rglob('*.py'))|set((ROOT/'backend').rglob('*.sql'))|
                   set((ROOT/'scripts').glob('*.py'))|set((ROOT/'evals/configs').glob('*.json'))|
                   {ROOT/p for p in ['uv.lock','pyproject.toml','requirements-data.lock.txt','.env.example']})
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in tracked}
    passed=all(r['status']=='passed' for r in [app_tests,data_tests,smoke,db,evidence])
    result={'phase':'0D','run_id':run_id,'status':'passed_for_stage1_development_with_constraints' if passed else 'failed',
        'app_tests':app_tests,'data_tests':data_tests,'http_smoke':smoke,'database':db,'evidence':evidence,
        'source_files':hashes,'source_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        'contracts_sha256':sha(output/'contracts.json'),
        'installed_versions':{k:importlib.metadata.version(k) for k in ['fastapi','pydantic','httpx2','psycopg','pytest']},
        'python':sys.version,'new_external_api_calls':0,'database_writes':0,
        'not_implemented':config['not_implemented'],'constraints':config['stage1_release_constraints'],
        'model_quality_approved':False,'bulk_ingestion_authorized':False}
    write_json(output/'summary.json',result); write_json(base/'summary.json',result)
    esc=html.escape
    rows=''.join(f'<tr><td>{esc(name)}</td><td>{r["status"]}</td><td>{esc(str(detail))}</td></tr>' for name,r,detail in [
        ('应用回归',app_tests,f'{app_tests["passed"]}/{app_tests["total"]}'),
        ('数据回归',data_tests,f'{data_tests["passed"]}/{data_tests["total"]}'),
        ('本机HTTP实启实测',smoke,sum(smoke.get('checks',{}).values())),
        ('项目库只读核验',db,db.get('server_version',db.get('error_code'))),
        ('历史证据与离线复现',evidence,f'{sum(evidence.get("checks",{}).values())}/{len(evidence.get("checks",{}))}')])
    qrows=''.join(f'<tr><td>{sid}</td><td>{q["status"]}</td><td>{esc(", ".join(q["blocking_reasons"]+q["review_reasons"]))}</td></tr>' for sid,q in evidence.get('quality_assessments',{}).items())
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0D阶段验收</title>
<style>body{{font:16px/1.7 system-ui;max-width:1080px;margin:32px auto;padding:0 20px;color:#243342}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:9px;text-align:left;overflow-wrap:anywhere}}.note{{padding:16px;background:#fff5dc}}a{{color:#126781}}</style>
<h1>0D：阶段0验收</h1><p>{esc(result['status'])} · {run_id}</p>
<p class="note">只允许进入阶段1开发，不表示完整RAG、混合检索或模型质量已通过。本轮外部模型/解析调用0次，数据库写入0次。历史0C指标不改写为本轮新成绩。</p>
<table><tr><th>检查</th><th>状态</th><th>结果</th></tr>{rows}</table>
<p><a href="{run_id}/summary.json">完整快照</a> · <a href="{run_id}/contracts.json">接口/结构契约</a> · <a href="{run_id}/app-tests.xml">应用JUnit</a> · <a href="{run_id}/data-tests.xml">数据JUnit</a> · <a href="{run_id}/preview.json">本机预览输出</a> · <a href="../stage0c/report.html">0C原页对照</a></p>
<h2>解析质量规则回放</h2><p>candidate=未发现已知结构问题，非内容正确或已审核；needs_review=需复核；blocked=来源/资源等硬错误。规则已实现并测试，尚未接入未来worker或版本激活流程。</p>
<table><tr><th>样本</th><th>状态</th><th>原因</th></tr>{qrows}</table>
<h2>阶段1约束</h2><ul>{''.join('<li>'+esc(x)+'</li>' for x in result['constraints'])}</ul>
<h2>尚未实现</h2><p>{esc(', '.join(result['not_implemented']))}</p></html>'''
    (base/'report.html').write_text(page,encoding='utf-8')
    (output/'report.html').write_text(page.replace(f'href="{run_id}/','href="').replace('href="../stage0c/','href="../../stage0c/'),encoding='utf-8')
    print(json.dumps({'status':result['status'],'app_tests':app_tests['passed'],'data_tests':data_tests['passed'],
        'http_smoke':smoke['status'],'database':db['status'],'evidence':evidence['status'],
        'quality_counts':evidence.get('quality_counts'),'new_external_api_calls':0,'report':str(base/'report.html')},ensure_ascii=False))
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
