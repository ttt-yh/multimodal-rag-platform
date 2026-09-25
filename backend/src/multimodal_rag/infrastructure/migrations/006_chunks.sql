CREATE TABLE mrag.chunks (
  chunk_id text PRIMARY KEY,
  processing_version_id text NOT NULL REFERENCES mrag.processing_versions(processing_version_id),
  document_id text NOT NULL,
  version_id text NOT NULL,
  ordinal integer NOT NULL CHECK (ordinal >= 0),
  text text NOT NULL,
  heading_path jsonb NOT NULL CHECK (jsonb_typeof(heading_path)='array'),
  element_ids jsonb NOT NULL CHECK (jsonb_typeof(element_ids)='array'),
  source_locations jsonb NOT NULL CHECK (jsonb_typeof(source_locations)='array'),
  length_unit text NOT NULL CHECK (length_unit='characters'),
  estimated_length integer NOT NULL CHECK (estimated_length >= 0),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (processing_version_id, ordinal),
  FOREIGN KEY (document_id,version_id) REFERENCES mrag.document_versions(document_id,version_id)
);

CREATE INDEX chunks_processing_version ON mrag.chunks(processing_version_id, ordinal);
