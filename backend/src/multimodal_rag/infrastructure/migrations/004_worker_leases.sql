-- A worker lease makes a database job recoverable after a process crash.
ALTER TABLE mrag.ingestion_jobs
  ADD COLUMN lease_owner text,
  ADD COLUMN lease_expires_at timestamptz,
  ADD COLUMN heartbeat_at timestamptz,
  ADD COLUMN completed_at timestamptz;

CREATE INDEX ingestion_jobs_claimable
  ON mrag.ingestion_jobs(status, lease_expires_at, created_at)
  WHERE status IN ('pending', 'running');
