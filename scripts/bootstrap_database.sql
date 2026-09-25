-- Run explicitly with psql as administrator, connected to postgres.
-- Dedicated database/role only. No password literal and no changes to old projects.
\set ON_ERROR_STOP on
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='multimodal_rag_app') THEN
    CREATE ROLE multimodal_rag_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
  ELSIF EXISTS (SELECT FROM pg_roles WHERE rolname='multimodal_rag_app'
               AND (rolsuper OR rolcreatedb OR rolcreaterole OR rolreplication)) THEN
    RAISE EXCEPTION 'Existing project role has unexpected privileges; inspect manually.';
  END IF;
END $$;
SELECT 'CREATE DATABASE multimodal_rag OWNER multimodal_rag_app ENCODING ''UTF8'' TEMPLATE template0'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname='multimodal_rag')
\gexec
SELECT (pg_get_userbyid(datdba) = 'multimodal_rag_app') AS correct_owner
FROM pg_database WHERE datname='multimodal_rag'
\gset
\if :correct_owner
  \connect multimodal_rag
  REVOKE CONNECT, TEMPORARY ON DATABASE multimodal_rag FROM PUBLIC;
  GRANT CONNECT, TEMPORARY ON DATABASE multimodal_rag TO multimodal_rag_app;
  REVOKE CREATE ON SCHEMA public FROM PUBLIC;
  \echo 'Dedicated database ready. Set role password interactively: \password multimodal_rag_app'
\else
  DO $$ BEGIN RAISE EXCEPTION 'Existing database owner mismatch; refusing to modify it.'; END $$;
\endif
