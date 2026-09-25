"""Explicit, resumable 0C experiment. No automatic POST/PUT or failed VLM retries.

The append-only operation decisions and pre-reserved counts survive interruptions.
Signed transfer URLs and credentials never enter the ledger. An interrupted upload
requires investigation, not resubmission. Re-running poll only queries the saved batch.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.model_adapters import VisionAdapter
from multimodal_rag.infrastructure.parser_batch import ParserBatchAdapter
from multimodal_rag.infrastructure.parser_normalization import normalize_archive
from multimodal_rag.infrastructure.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evals/results/stage0c/live'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    """Same-directory atomic replacement: never leave half of a budget ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


class Ledger:
    def __init__(self, path, fingerprint):
        self.path = path
        self.data = read(path) if path.exists() else {
            'schema_version': 1, 'fingerprint': fingerprint,
            'limits': {'parser_http':60, 'vision_http':8, 'pages':20, 'output_tokens':512,
                       'reserved_cny':1, 'not_an_account_billing_cap':True},
            'used': {'parser':0,'vision':0}, 'operations':{}, 'samples':{}, 'vision':{},
            'records':[], 'created_at':datetime.now(timezone.utc).isoformat()}
        if self.data['fingerprint'] != fingerprint:
            raise AppError('frozen_inputs_changed', '固定样本或模型配置发生变化，不能混入当前批次', 409)
        self.save()

    def save(self):
        self.data['updated_at'] = datetime.now(timezone.utc).isoformat()
        write(self.path, self.data)

    def begin(self, name):
        if name in self.data['operations']:
            raise AppError('operation_already_attempted', '已尝试的非幂等操作不自动重发', 409)
        self.data['operations'][name] = 'attempting'
        self.save()


class DurableBudget(CallBudget):
    def __init__(self, ledger, service):
        self.ledger, self.service = ledger, service
        super().__init__(60 if service == 'parser' else 8, ledger.data['used'][service])

    def consume(self):
        super().consume()
        self.ledger.data['used'][self.service] = self.used
        self.ledger.save()  # Reserve before network. A crash may overcount, never undercount.


def validate_inputs(config, cases):
    samples = config['samples']
    if len(samples) != 18 or sum(s['page_count'] for s in samples) != 20 or len(cases['cases']) != 8:
        raise AppError('invalid_fixed_scope', '仅允许预先固定的18文件20页和8道题', 422)
    for s in samples:
        path = (ROOT/s['input_path']).resolve()
        if not path.is_relative_to(ROOT) or digest(path.read_bytes()) != s['input_sha256']:
            raise AppError('input_hash_mismatch', '上传文件与固定清单不一致', 422)
        if len(s['pages']) != s['page_count'] or any(p['split'] != 'dev' for p in s['pages']):
            raise AppError('invalid_source_split', '只允许开发样本', 422)
    images = {}
    by_id = {s['sample_id']:s for s in samples}
    for c in cases['cases']:
        path = (ROOT/by_id[c['sample_id']]['previews'][c['preview_index']]).resolve()
        if not path.is_relative_to(ROOT):
            raise AppError('invalid_image_path', '图片路径越界', 422)
        images[c['case_id']] = digest(path.read_bytes())
    return images


def submit(ledger, gateway, samples):
    ledger.begin('submit_batch')
    adapter = ParserBatchAdapter(gateway)
    batch, urls = adapter.submit(samples)
    ledger.data['batch_id'] = batch
    ledger.data['operations']['submit_batch'] = 'submitted'
    ledger.save()
    print('batch_id saved; uploading fixed PDFs', flush=True)
    for s in samples:
        sid = s['sample_id']
        ledger.begin('upload_'+sid)
        adapter.upload(urls[sid], (ROOT/s['input_path']).read_bytes())
        ledger.data['samples'][sid] = {'state':'uploaded'}
        ledger.data['operations']['upload_'+sid] = 'uploaded'
        ledger.save()
        print(sid+' uploaded', flush=True)


def poll(ledger, gateway, samples):
    if 'batch_id' not in ledger.data:
        raise AppError('no_saved_batch', '没有可续查的批次；禁止猜测或重新提交', 409)
    unfinished = [s for s in samples if ledger.data['samples'].get(s['sample_id'],{}).get('state') not in {'normalized','failed'}]
    if not unfinished:
        print('All parser results already saved', flush=True)
        return
    # Keep one download slot for every pending file, even when the provider is slow.
    if gateway.budget.used + 1 + len(unfinished) > 60:
        raise AppError('poll_budget_reserved', '保留结果下载预算，停止轮询', 429)
    states = ParserBatchAdapter(gateway).poll(ledger.data['batch_id'], {s['sample_id'] for s in samples})
    for s in unfinished:
        sid = s['sample_id']
        row = states[sid]
        ledger.data['samples'][sid] = {'state':row['state']}
        ledger.save()
        if row['state'] == 'done':
            path = OUT/(sid+'.zip')
            if not path.exists():
                content = gateway.request('GET','',transfer_url=row['url'],binary=True,
                                          max_bytes=gateway.settings.parser_max_zip_bytes)
                # Keep the exact bounded response for offline diagnosis, including malformed archives.
                path.write_bytes(content)
            normalize_one(ledger, s)
        print(sid+' '+ledger.data['samples'][sid]['state'], flush=True)


def normalize_one(ledger, sample):
    sid = sample['sample_id']
    path = OUT/(sid+'.zip')
    if not path.exists():
        return
    try:
        result = normalize_archive(path.read_bytes(), sample)
        old_path = OUT/(sid+'.json')
        if old_path.exists():
            old = read(old_path)
            if old['normalizer_version'] != result['normalizer_version']:
                # Preserve the first live output before offline adapter corrections.
                history = OUT/'normalization-history'/(sid+'.'+old['normalizer_version']+'.json')
                if not history.exists(): write(history,old)
        write(OUT/(sid+'.json'), result)
        ledger.data['samples'][sid] = {'state':'normalized','elements':len(result['elements']),
            'warnings':result['warnings'],'artifact':result['artifact']}
    except AppError as exc:
        ledger.data['samples'][sid] = {'state':'normalization_failed','error_code':exc.code}
    ledger.save()


def vision(ledger, gateway, samples, cases):
    by_id = {s['sample_id']:s for s in samples}
    for c in cases['cases']:
        cid = c['case_id']
        if 'vision_'+cid in ledger.data['operations']:
            print(cid+' already attempted; skipped', flush=True)
            continue
        ledger.begin('vision_'+cid)
        path = ROOT/by_id[c['sample_id']]['previews'][c['preview_index']]
        try:
            # Never pass expected_points, labels, or extracted reference text to the VLM.
            answer = VisionAdapter(gateway).describe(c['question'], path.read_bytes(), 'image/png')
            ledger.data['vision'][cid] = answer | {'image_sha256':digest(path.read_bytes()),
                'question':c['question'],'review_status':'pending','human_reviewed':False}
            ledger.data['operations']['vision_'+cid] = 'completed'
        except AppError as exc:
            ledger.data['vision'][cid] = {'error_code':exc.code, 'review_status':'not_scored'}
            ledger.data['operations']['vision_'+cid] = 'failed_no_retry'
        ledger.save()
        print(cid+' '+ledger.data['operations']['vision_'+cid], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['submit','poll','vision','normalize','status'])
    parser.add_argument('--confirm-live',action='store_true')
    args = parser.parse_args()
    if args.step in {'submit','poll','vision'} and not args.confirm_live:
        parser.error('network steps require --confirm-live')
    OUT.mkdir(parents=True,exist_ok=True)
    lock = OUT/'runner.lock'
    try:
        lock_fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        print('runner_lock_exists: verify no running process before manually clearing the lock')
        return 2
    gateway = None
    ledger = None
    try:
        os.write(lock_fd, str(os.getpid()).encode())
        config_path = ROOT/'evals/configs/stage0c_samples.json'
        cases_path = ROOT/'evals/configs/stage0c_visual_cases.json'
        config, cases = read(config_path), read(cases_path)
        images = validate_inputs(config, cases)
        settings = load_settings().model_copy(update={'mode':'api','api_enabled':True,
            'api_max_output_tokens':512,'api_get_retries':0,'api_timeout_seconds':120})
        if settings.parser_model != 'vlm' or settings.vision_model != 'qwen3-vl-plus':
            raise AppError('model_changed','本批模型与已确认预算不符',409)
        fingerprint = {'samples':digest(config_path.read_bytes()),'questions':digest(cases_path.read_bytes()),
                       'images':images,'parser_model':settings.parser_model,'vision_model':settings.vision_model,
                       'parser_endpoint_sha256':digest(settings.service_base_url('parser').encode()),
                       'vision_endpoint_sha256':digest(settings.service_base_url('vision').encode()),
                       'max_tokens':512,'thinking':False}
        ledger = Ledger(OUT/'ledger.json',fingerprint)
        if args.step == 'status':
            print(json.dumps({'used':ledger.data['used'],'samples':{k:v['state'] for k,v in ledger.data['samples'].items()},
                              'vision_saved':len(ledger.data['vision'])}))
        elif args.step == 'normalize':
            for s in config['samples']:
                normalize_one(ledger,s)
        else:
            service = 'vision' if args.step == 'vision' else 'parser'
            gateway = HttpGateway(settings,service,budget=DurableBudget(ledger,service))
            if args.step == 'submit': submit(ledger,gateway,config['samples'])
            elif args.step == 'poll': poll(ledger,gateway,config['samples'])
            else: vision(ledger,gateway,config['samples'],cases)
        return 0
    except Exception as exc:
        # No raw provider errors, signed URLs, settings or secrets on stdout.
        code = exc.code if isinstance(exc,AppError) else type(exc).__name__
        if ledger:
            ledger.data.setdefault('errors',[]).append({'step':args.step,'code':code})
            ledger.save()
        print('Stopped safely: '+code,flush=True)
        return 1
    finally:
        if gateway:
            if ledger:
                ledger.data['records'].extend(gateway.records)
                ledger.save()
            gateway.close()
        os.close(lock_fd)
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
