from copy import deepcopy

import pytest

from multimodal_rag.core.quality import assess_parser_result


def result():
    return {'elements':[{'element':{'element_id':'el_one','document_id':'doc','version_id':'ver',
        'kind':'text','order':0,'raw_text':'body','source':{
            'source_path':'source.pdf','kind':'pdf','precision':'page','page_number':2}},
        'index_eligible':True}], 'warnings':[], 'artifact':{'missing_image_references':0}}


def test_candidate_is_not_approval():
    r=result(); before=deepcopy(r)
    assessment=assess_parser_result(r)
    assert assessment['status']=='candidate'
    assert not assessment['auto_activate']
    assert r==before


@pytest.mark.parametrize('code',['empty_parsed_element','chart_values_require_visual_reading',
                                'unknown_type_preserved_as_text','new_provider_warning'])
def test_warnings_require_review(code):
    r=result(); r['warnings']=[{'code':code}]
    assert assess_parser_result(r)['status']=='needs_review'


@pytest.mark.parametrize('code',['pages_without_elements','missing_image_resource'])
def test_missing_evidence_blocks(code):
    r=result(); r['warnings']=[{'code':code}]
    assert assess_parser_result(r)['status']=='blocked'


def test_invalid_source_blocks():
    r=result(); r['elements'][0]['element']['source']['page_number']=None
    assert assess_parser_result(r)['status']=='blocked'


def test_empty_input_and_duplicate_ids_block():
    assert assess_parser_result({'elements':[]})['status']=='blocked'
    r=result(); r['elements']*=2
    assert 'duplicate_element_id' in assess_parser_result(r)['blocking_reasons']


def test_known_content_issue_not_cleared_by_good_structure():
    r=assess_parser_result(result(),('command_transcription_unverified',))
    assert r['status']=='needs_review' and not r['auto_activate']


def test_empty_eligible_text_cannot_be_indexed():
    r=result(); r['elements'][0]['element']['raw_text']=''
    assert 'empty_element_marked_indexable' in assess_parser_result(r)['blocking_reasons']


def test_nonempty_header_not_silently_excluded():
    r=result(); r['elements'][0]['provider_type']='header'
    assert assess_parser_result(r)['index_candidate_elements']==1


@pytest.mark.parametrize('field,value',[('artifact',None),('warnings',None),('warnings',['bad'])])
def test_incomplete_quality_envelope_blocks(field,value):
    r=result(); r[field]=value
    assert assess_parser_result(r)['status']=='blocked'


def test_image_without_resource_reference_blocks():
    r=result(); r['elements'][0]['element']['kind']='image'
    assert 'missing_image_reference' in assess_parser_result(r)['blocking_reasons']
