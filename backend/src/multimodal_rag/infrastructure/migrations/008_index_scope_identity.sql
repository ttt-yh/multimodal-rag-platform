-- An index version is identified by index_version_id, which hashes the full
-- processing_version_ids scope plus the embedding/BM25 configuration.
-- The former uniqueness rule only considered the compatibility anchor and
-- incorrectly rejected a later cumulative index that reused the same anchor.
ALTER TABLE mrag.index_versions
  DROP CONSTRAINT IF EXISTS index_versions_processing_version_id_embedding_model_embedd_key;

