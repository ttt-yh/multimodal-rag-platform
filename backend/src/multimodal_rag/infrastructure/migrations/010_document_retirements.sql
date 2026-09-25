-- Document retirement is an auditable business event; source and derived data remain recoverable.
CREATE TABLE mrag.document_retirements (
  retirement_id text PRIMARY KEY,
  document_id text NOT NULL REFERENCES mrag.documents(document_id),
  reviewer text NOT NULL,
  notes text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX document_retirements_document ON mrag.document_retirements(document_id, created_at DESC);
