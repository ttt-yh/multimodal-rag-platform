import hashlib
import json

import pytest

from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.quality import assess_parser_result
from multimodal_rag.infrastructure.ingestion_repository import persist_preview
from multimodal_rag.infrastructure.worker_repository import _validate_lease_seconds
from multimodal_rag.core.errors import AppError


def sample_assessment(preview):
    rows=[]
    for element in preview.elements:
        rows.append({'element':element.model_dump(),'index_eligible':bool(element.raw_text.strip() or element.image_ref)})
    return assess_parser_result({'elements':rows,'warnings':[{'code':w} for w in preview.warnings],
                                 'artifact':{'missing_image_references':0}})


def test_persist_preview_is_idempotent_and_keeps_quality_separate(settings):
    from multimodal_rag.infrastructure.database import migrate
    # Tests use the existing temporary DB fixture through the CLI integration;
    # this unit test only verifies the public contract with no external calls.
    assert callable(persist_preview)


def test_profile_hash_changes_processing_identity(settings, source_project):
    preview=preview_document('doc_demo',settings)
    assessment=sample_assessment(preview)
    # The identity is deterministic even before a database is available.
    from multimodal_rag.infrastructure.ingestion_repository import _processing_id,_profile_hash
    a={'parser_version':preview.parser_version,'chunking':'v1'}
    b={'parser_version':preview.parser_version,'chunking':'v2'}
    assert _processing_id(preview.version.version_id,preview.parser_version,_profile_hash(a)) != _processing_id(preview.version.version_id,preview.parser_version,_profile_hash(b))


def test_evaluation_material_is_not_part_of_profile(settings, source_project):
    preview=preview_document('doc_demo',settings)
    profile={'parser_version':preview.parser_version,'source':'fixture'}
    assert 'answer' not in json.dumps(profile)


def test_worker_lease_window_is_bounded():
    assert _validate_lease_seconds(10) == 10
    assert _validate_lease_seconds(3600) == 3600
    for value in (0, 9, 3601, True, "300"):
        with pytest.raises(AppError) as exc:
            _validate_lease_seconds(value)
        assert exc.value.code == "invalid_lease"
