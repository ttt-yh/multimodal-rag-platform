-- Durable provenance for asynchronous PDF parsing.  Signed provider URLs and
-- credentials are intentionally never persisted.
CREATE TABLE mrag.parser_artifacts (
  processing_version_id text PRIMARY KEY REFERENCES mrag.processing_versions(processing_version_id),
  job_id text NOT NULL UNIQUE REFERENCES mrag.ingestion_jobs(job_id),
  provider_batch_id text NOT NULL,
  normalizer_version text NOT NULL,
  archive_sha256 text NOT NULL CHECK (archive_sha256 ~ '^[0-9a-f]{64}$'),
  artifact_path text NOT NULL,
  image_count integer NOT NULL CHECK (image_count >= 0),
  created_at timestamptz NOT NULL DEFAULT now()
);
