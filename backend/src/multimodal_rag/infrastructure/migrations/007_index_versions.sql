-- Dense and keyword indexes are published together through this metadata row.
CREATE TABLE mrag.index_versions (
  index_version_id text PRIMARY KEY,
  processing_version_id text NOT NULL REFERENCES mrag.processing_versions(processing_version_id),
  embedding_model text NOT NULL,
  embedding_dimensions integer NOT NULL CHECK (embedding_dimensions > 0),
  bm25_version text NOT NULL,
  chunk_count integer NOT NULL CHECK (chunk_count >= 0),
  vector_count integer NOT NULL CHECK (vector_count >= 0),
  keyword_count integer NOT NULL CHECK (keyword_count >= 0),
  artifact_path text NOT NULL,
  status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','active','retired','failed')),
  manifest jsonb NOT NULL CHECK (jsonb_typeof(manifest)='object'),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (processing_version_id, embedding_model, embedding_dimensions, bm25_version)
);

CREATE INDEX index_versions_processing ON mrag.index_versions(processing_version_id, status, created_at DESC);
