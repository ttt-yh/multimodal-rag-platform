-- Human review is an auditable event; it does not activate a knowledge version.
CREATE TABLE mrag.quality_reviews (
  review_id text PRIMARY KEY,
  processing_version_id text NOT NULL REFERENCES mrag.processing_versions(processing_version_id),
  reviewer text NOT NULL,
  decision text NOT NULL CHECK (decision IN ('approved','rejected')),
  notes text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX quality_reviews_processing_version
  ON mrag.quality_reviews(processing_version_id, created_at DESC);
