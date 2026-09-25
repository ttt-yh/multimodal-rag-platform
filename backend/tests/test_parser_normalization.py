from io import BytesIO
import json
from zipfile import ZipFile

import pytest

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.parser_normalization import normalize_archive


def archive(rows, assets=None):
    out=BytesIO()
    with ZipFile(out,'w') as z:
        z.writestr('full.md','fixture body')
        z.writestr('sample_content_list.json',json.dumps(rows))
        for key,value in (assets or {}).items(): z.writestr(key,value)
    return out.getvalue()


def sample(image=False):
    return {'sample_id':'s0c_01','page_count':2,'pages':[
        {'document_id':'doc','source_path':'original.pdf' if not image else 'original.png','source_sha256':'a'*64,
         'source_page':p,'split':'dev','source_kind':'image' if image else 'pdf','width':600,'height':800} for p in [473,474]]}


def test_local_pages_map_to_original_not_fragment():
    blob=archive([{'type':'text','text':'one','page_idx':0,'bbox':[0,0,500,600]}, {'type':'text','text':'two','page_idx':1}])
    result=normalize_archive(blob,sample())
    assert [x['element']['source']['page_number'] for x in result['elements']] == [473,474]
    assert result['elements'][0]['element']['source']['bbox'] is None
    assert result['elements'][0]['provider_bbox'] == [0,0,500,600]
    assert result['status'] == 'mapped'
    assert result == normalize_archive(blob,sample())


def test_image_origin_keeps_image_coordinates_not_fake_pdf_page():
    result=normalize_archive(archive([{'type':'text','text':'one','page_idx':0}]),sample(True))
    location=result['elements'][0]['element']['source']
    assert location['kind']=='image' and location['page_number'] is None and location['image_width']==600
    assert result['warnings'][0]['code']=='pages_without_elements'


@pytest.mark.parametrize('index',[-1,2,True,'0'])
def test_bad_pages_rejected(index):
    with pytest.raises(AppError): normalize_archive(archive([{'type':'text','text':'x','page_idx':index}]),sample())


@pytest.mark.parametrize('path',['../outside.png','https://site/a.png','C:\\tmp.png','/outside.png'])
def test_unsafe_image_refs_rejected(path):
    with pytest.raises(AppError): normalize_archive(archive([{'type':'image','img_path':path,'page_idx':0}]),sample())


def test_html_table_preserved_as_data():
    table='<table><tr><td>32 GB+</td></tr></table>'
    result=normalize_archive(archive([{'type':'table','table_body':table,'table_caption':['memory'],'page_idx':0}]),sample())
    assert table in result['elements'][0]['element']['raw_text']
    assert result['elements'][0]['element']['kind']=='table'


def test_missing_asset_reported_not_silently_claimed_complete():
    result=normalize_archive(archive([{'type':'image','img_path':'images/a.png','page_idx':0}]),sample())
    assert result['status']=='mapped_with_warnings'
    assert result['artifact']['missing_image_references']==1


def test_test_split_cannot_enter_dev_mapping():
    item=sample(); item['pages'][0]['split']='test'
    with pytest.raises(AppError): normalize_archive(archive([{'type':'text','text':'x','page_idx':0}]),item)


def test_unknown_type_and_empty_output_marked():
    result=normalize_archive(archive([{'type':'future-type','page_idx':0}]),sample())
    assert {'unknown_type_preserved_as_text','empty_parsed_element'} <= {w['code'] for w in result['warnings']}


def test_live_chart_caption_is_not_lost_or_fabricated():
    result=normalize_archive(archive([{'type':'chart','chart_caption':['Revenue'],
        'chart_footnote':['Source'],'content':'','img_path':'images/c.jpg','page_idx':0}],
        {'images/c.jpg':b'image'}),sample())
    row=result['elements'][0]
    assert row['element']['kind']=='image'
    assert row['element']['raw_text']=='Revenue\n\nSource'
    assert row['index_eligible']
    assert any(w['code']=='chart_values_require_visual_reading' for w in result['warnings'])


def test_decorations_preserved_but_not_indexed():
    result=normalize_archive(archive([{'type':t,'text':text,'page_idx':0}
        for t,text in [('header','Settings'),('footer','PingCAP docs-cn | revision'),
                       ('page_number','16'),('page_footnote','source')]]),sample())
    assert [r['index_eligible'] for r in result['elements']]==[True,False,False,True]
    assert not any(w['code']=='unknown_type_preserved_as_text' for w in result['warnings'])


def test_empty_provider_text_remains_a_quality_warning():
    result=normalize_archive(archive([{'type':'text','text':'','page_idx':0}]),sample())
    assert not result['elements'][0]['index_eligible']
    assert any(w['code']=='empty_parsed_element' for w in result['warnings'])
