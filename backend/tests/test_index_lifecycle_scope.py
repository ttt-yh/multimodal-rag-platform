from multimodal_rag.infrastructure.index_lifecycle import _processing_scope


def test_processing_scope_prefers_full_manifest_scope():
    assert _processing_scope({"processing_version_ids": ["proc_b", "proc_a"]}, "proc_anchor") == {
        "proc_a", "proc_b"
    }


def test_processing_scope_supports_legacy_single_version_manifest():
    assert _processing_scope({}, "proc_anchor") == {"proc_anchor"}
