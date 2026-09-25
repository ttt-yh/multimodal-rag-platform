"""版本化入库使用的解析质量判定规则；纯函数，不联网、不写库、不激活版本。

该规则已接入文本与 PDF 入库 worker。candidate 只表示未发现已知结构问题，
绝不等价于 OCR 内容全部正确，也不会绕过后续人工审核与发布门禁。
"""
from pydantic import ValidationError

from multimodal_rag.core.models import Element

POLICY_VERSION = 'parser-quality-v1'
BLOCKING = {'missing_image_resource', 'pages_without_elements'}
REVIEW = {'empty_parsed_element', 'chart_values_require_visual_reading',
          'unknown_type_preserved_as_text'}


def assess_parser_result(result: dict, known_issues: tuple[str, ...] = ()) -> dict:
    """不自动审核、不吞掉未知警告；人工发现的问题可作为输入保留。"""
    blocked, review = set(), set(known_issues)
    rows = result.get('elements')
    if not isinstance(rows, list) or not rows:
        blocked.add('no_elements')
        rows = []
    ids = set()
    eligible = 0
    for row in rows:
        try:
            element = Element.model_validate(row['element'])
            if element.element_id in ids:
                blocked.add('duplicate_element_id')
            ids.add(element.element_id)
            if not element.source.source_path.strip():
                blocked.add('missing_source_path')
            if element.kind == 'image' and not element.image_ref:
                blocked.add('missing_image_reference')
            if row.get('index_eligible') is True:
                eligible += 1
                if not element.raw_text.strip() and not element.image_ref:
                    blocked.add('empty_element_marked_indexable')
            elif row.get('index_eligible') is not False:
                blocked.add('missing_index_eligibility')
        except (ValidationError, KeyError, TypeError):
            blocked.add('invalid_element_contract')
    if rows and eligible == 0:
        blocked.add('no_indexable_elements')
    artifact = result.get('artifact', {})
    if not isinstance(artifact, dict) or type(artifact.get('missing_image_references')) is not int:
        blocked.add('invalid_artifact_contract')
        artifact = {}
    if artifact.get('missing_image_references', 0):
        blocked.add('missing_image_resource')
    warnings = result.get('warnings')
    if not isinstance(warnings, list) or any(not isinstance(w, dict) for w in warnings):
        blocked.add('invalid_warning_contract')
        warnings = []
    for warning in warnings:
        code = warning.get('code', 'unknown_warning')
        if code in BLOCKING:
            blocked.add(code)
        else:
            review.add(code if code in REVIEW else 'unrecognized_warning:'+str(code))
    status = 'blocked' if blocked else 'needs_review' if review else 'candidate'
    return {'policy_version': POLICY_VERSION, 'status': status,
            'blocking_reasons': sorted(blocked), 'review_reasons': sorted(review),
            'index_candidate_elements': eligible, 'auto_activate': False,
            'meaning': 'structure_screening_only_not_content_accuracy',
            'worker_integration': 'versioned_ingestion_worker'}
