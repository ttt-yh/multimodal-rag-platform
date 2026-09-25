-- Processing is separate from immutable source content.
-- A parser/chunking/profile change creates a new processing version.
ALTER TABLE mrag.elements
  ADD COLUMN processing_version_id text;

CREATE TABLE mrag.processing_versions (
  processing_version_id text PRIMARY KEY,
  document_id text NOT NULL,
  version_id text NOT NULL,
  parser_version text NOT NULL,
  profile_sha256 text NOT NULL CHECK (profile_sha256 ~ '^[0-9a-f]{64}$'),
  processing_profile jsonb NOT NULL CHECK (jsonb_typeof(processing_profile)='object'),
  quality_status text NOT NULL CHECK (quality_status IN ('candidate','needs_review','blocked','approved')),
  release_status text NOT NULL DEFAULT 'draft' CHECK (release_status IN ('draft','active','retired')),
  element_count integer NOT NULL DEFAULT 0 CHECK (element_count >= 0),
  warning_count integer NOT NULL DEFAULT 0 CHECK (warning_count >= 0),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (document_id,version_id) REFERENCES mrag.document_versions(document_id,version_id),
  UNIQUE (document_id,version_id,parser_version,profile_sha256)
);

ALTER TABLE mrag.elements
  ADD CONSTRAINT elements_processing_version_fk
  FOREIGN KEY (processing_version_id) REFERENCES mrag.processing_versions(processing_version_id);

CREATE UNIQUE INDEX elements_processing_ordinal
  ON mrag.elements(processing_version_id, ordinal)
  WHERE processing_version_id IS NOT NULL;

ALTER TABLE mrag.ingestion_jobs
  ADD COLUMN processing_version_id text,
  ADD COLUMN quality_status text CHECK (quality_status IS NULL OR quality_status IN ('candidate','needs_review','blocked','approved'));

ALTER TABLE mrag.ingestion_jobs
  ADD CONSTRAINT ingestion_jobs_processing_version_fk
  FOREIGN KEY (processing_version_id) REFERENCES mrag.processing_versions(processing_version_id);

CREATE INDEX processing_quality_status ON mrag.processing_versions(quality_status, release_status, created_at);
