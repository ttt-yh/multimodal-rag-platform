import importlib.util
import json
from pathlib import Path

import httpx2 as httpx
import pytest

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import CallBudget, HttpGateway
from multimodal_rag.infrastructure.parser_batch import ParserBatchAdapter
from multimodal_rag.infrastructure.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('stage0c_live', ROOT/'scripts/run_stage0c_live.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def gateway(handler):
    settings = Settings(_env_file=None, mode='mock', parser_api_key='test-only')
    return HttpGateway(settings,'parser',budget=CallBudget(60),transport=httpx.MockTransport(handler))


def test_durable_budget_reopens_and_stops(tmp_path):
    ledger = runner.Ledger(tmp_path/'ledger.json',{'frozen':True})
    budget = runner.DurableBudget(ledger,'vision')
    budget.consume()
    reopened = runner.Ledger(ledger.path,{'frozen':True})
    assert reopened.data['used']['vision'] == 1
    budget = runner.DurableBudget(reopened,'vision')
    for _ in range(7): budget.consume()
    with pytest.raises(AppError,match='预算'):
        budget.consume()
    assert runner.read(ledger.path)['used']['vision'] == 8


def test_attempt_is_not_repeated_after_interrupt(tmp_path):
    ledger = runner.Ledger(tmp_path/'ledger.json',{})
    ledger.begin('submit_batch')
    reopened = runner.Ledger(ledger.path,{})
    with pytest.raises(AppError): reopened.begin('submit_batch')


def test_changed_inputs_rejected(tmp_path):
    runner.Ledger(tmp_path/'ledger.json',{'sha':'one'})
    with pytest.raises(AppError): runner.Ledger(tmp_path/'ledger.json',{'sha':'two'})


def test_batch_single_request_preserves_order_and_page_ranges():
    def handler(request):
        body = json.loads(request.content)
        assert body['files'] == [
            {'name':'s0c_01.pdf','data_id':'s0c_01','page_ranges':'1-2'},
            {'name':'s0c_02.pdf','data_id':'s0c_02','page_ranges':'1-1'}]
        return httpx.Response(200,json={'code':0,'data':{'batch_id':'batch-1',
            'file_urls':['https://mineru.oss-cn-shanghai.aliyuncs.com/a',
                         'https://mineru.oss-cn-shanghai.aliyuncs.com/b']}})
    g = gateway(handler)
    batch, urls = ParserBatchAdapter(g).submit([{'sample_id':'s0c_01','page_count':2},
                                               {'sample_id':'s0c_02','page_count':1}])
    assert batch == 'batch-1' and urls['s0c_02'].endswith('/b') and g.budget.used == 1
    g.close()


@pytest.mark.parametrize('samples',[
    [{'sample_id':f's_{i}','page_count':2} for i in range(11)],
    [{'sample_id':'same','page_count':1}]*2,
    [{'sample_id':'bad','page_count':0}],
])
def test_bad_batch_never_sends(samples):
    g = gateway(lambda r: pytest.fail('should not send'))
    with pytest.raises(AppError): ParserBatchAdapter(g).submit(samples)
    assert g.budget.used == 0
    g.close()


def test_upload_does_not_forward_auth():
    def handler(request):
        assert 'authorization' not in request.headers
        assert request.content.startswith(b'%PDF-')
        return httpx.Response(200,content=b'')
    g = gateway(handler)
    ParserBatchAdapter(g).upload('https://mineru.oss-cn-shanghai.aliyuncs.com/a', b'%PDF-'+b'x'*2_100_000)
    assert g.budget.used == 1
    g.close()


@pytest.mark.parametrize('rows',[
    [{'data_id':'a','state':'done','full_zip_url':'http://127.0.0.1/private'}],
    [{'data_id':'a','state':'pending'},{'data_id':'a','state':'pending'}],
    [{'data_id':'foreign','state':'pending'}],
])
def test_poll_rejects_unsafe_or_mismatched_rows(rows):
    g = gateway(lambda r: httpx.Response(200,json={'code':0,'data':{'extract_result':rows}}))
    with pytest.raises(AppError): ParserBatchAdapter(g).poll('batch1',{'a'})
    g.close()


def test_batch_poll_once_for_all_files():
    rows = [{'data_id':f's_{i}','state':'pending'} for i in range(18)]
    g = gateway(lambda r: httpx.Response(200,json={'code':0,'data':{'extract_result':rows}}))
    assert len(ParserBatchAdapter(g).poll('batch1',{r['data_id'] for r in rows})) == 18
    assert g.budget.used == 1
    g.close()
