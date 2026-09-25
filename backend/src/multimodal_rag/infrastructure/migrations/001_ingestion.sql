-- Schema mrag and schema_migrations are managed by the migration runner.
CREATE TABLE mrag.documents (
  document_id text PRIMARY KEY,
  knowledge_base text NOT NULL,
  title text NOT NULL,
  source_path text NOT NULL,
  source_family text NOT NULL,
  format text NOT NULL CHECK (format IN ('md','txt','pdf','image')),
  license text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE mrag.document_versions (
  version_id text PRIMARY KEY,
  document_id text NOT NULL REFERENCES mrag.documents(document_id),
  content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
  source_revision text NOT NULL,
  status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','active','retired')),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (document_id, version_id),
  UNIQUE (document_id, content_sha256)
);
CREATE UNIQUE INDEX one_active_document_version ON mrag.document_versions(document_id) WHERE status='active';
CREATE TABLE mrag.elements (
  element_id text PRIMARY KEY,
  document_id text NOT NULL,
  version_id text NOT NULL,
  parser_version text NOT NULL,
  kind text NOT NULL CHECK (kind IN ('heading','text','table','image','code')),
  ordinal integer NOT NULL CHECK (ordinal >= 0),
  raw_text text NOT NULL,
  heading_path jsonb NOT NULL CHECK (jsonb_typeof(heading_path)='array'),
  source jsonb NOT NULL CHECK (jsonb_typeof(source)='object'),
  image_ref text,
  generated_description text,
  FOREIGN KEY (document_id,version_id) REFERENCES mrag.document_versions(document_id,version_id),
  UNIQUE (version_id,parser_version,ordinal)
);
CREATE TABLE mrag.ingestion_jobs (
  job_id text PRIMARY KEY,
  document_id text NOT NULL,
  version_id text NOT NULL,
  operation text NOT NULL CHECK (operation IN ('preview','ingest')),
  status text NOT NULL CHECK (status IN ('pending','running','succeeded','failed')),
  stage text NOT NULL,
  idempotency_key text NOT NULL UNIQUE,
  processing_profile jsonb NOT NULL DEFAULT '{}'::jsonb,
  external_batch_id text,
  error_code text,
  attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (document_id,version_id) REFERENCES mrag.document_versions(document_id,version_id)
);
CREATE INDEX ingestion_job_status ON mrag.ingestion_jobs(status,created_at);
