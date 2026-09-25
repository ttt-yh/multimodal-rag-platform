"""0C bounded batch API. Signed URLs are runtime-only values, never report fields."""
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.parser_adapter import checked_task_id
from multimodal_rag.infrastructure.model_adapters import invalid, successful


class ParserBatchAdapter:
    def __init__(self, gateway):
        if gateway.service != 'parser':
            raise ValueError('parser gateway required')
        self.gateway = gateway

    def submit(self, samples):
        if not samples or len(samples) > 18 or sum(s['page_count'] for s in samples) > 20:
            raise AppError('sample_budget_exceeded', '本批最多18文件、20页', 422)
        ids = [checked_task_id(s['sample_id']) for s in samples]
        if len(ids) != len(set(ids)) or any(type(s['page_count']) is not int or not 1 <= s['page_count'] <= 2 for s in samples):
            raise AppError('invalid_samples', '样本编号或页数不合法', 422)
        data = self.gateway.request('POST', '/file-urls/batch', payload={
            'files': [{'name': sid+'.pdf', 'data_id': sid, 'page_ranges': f"1-{s['page_count']}"}
                      for sid, s in zip(ids, samples)],
            'model_version': self.gateway.settings.parser_model,
            'enable_table': True, 'enable_formula': True, 'language': 'ch'})
        try:
            batch_id = checked_task_id(data['data']['batch_id'])
            urls = data['data']['file_urls']
            if not isinstance(urls, list) or len(urls) != len(ids):
                invalid(self.gateway)
            for url in urls:
                self.gateway._transfer_url(url)
        except (KeyError, TypeError):
            invalid(self.gateway)
        successful(self.gateway, data)
        return batch_id, dict(zip(ids, urls))

    def upload(self, url, content):
        if not content.startswith(b'%PDF-') or len(content) > 20_971_520:
            raise AppError('invalid_parser_sample', '本批只接受不超过20MB的PDF', 422)
        self.gateway.request('PUT', '', transfer_url=url, content=content, binary=True)

    def poll(self, batch_id, expected_ids):
        data = self.gateway.request('GET', '/extract-results/batch/'+checked_task_id(batch_id))
        try:
            rows = data['data']['extract_result']
            if not isinstance(rows, list):
                invalid(self.gateway)
            results = {}
            for row in rows:
                sid, state = row['data_id'], row['state']
                if sid not in expected_ids or sid in results or state not in {'waiting-file','pending','running','converting','done','failed'}:
                    invalid(self.gateway)
                results[sid] = {'state': state}
                if state == 'done':
                    self.gateway._transfer_url(row['full_zip_url'])
                    results[sid]['url'] = row['full_zip_url']
            if set(results) != set(expected_ids):
                invalid(self.gateway)
        except (KeyError, TypeError):
            invalid(self.gateway)
        successful(self.gateway, data)
        return results
