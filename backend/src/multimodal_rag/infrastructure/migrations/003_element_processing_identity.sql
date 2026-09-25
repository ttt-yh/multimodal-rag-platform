-- Element identity and eligibility belong to a processing run, not only to the
-- immutable source/parser pair. Preserve legacy rows with a NULL processing id.
ALTER TABLE mrag.elements
  DROP CONSTRAINT IF EXISTS elements_version_id_parser_version_ordinal_key;

-- 002 created a temporary identity rule. Replace it with the final rule below.
DROP INDEX IF EXISTS mrag.elements_processing_ordinal;

ALTER TABLE mrag.elements
  ADD COLUMN index_eligible boolean,
  ADD COLUMN excluded_reason text;

CREATE UNIQUE INDEX elements_processing_version_ordinal
  ON mrag.elements(processing_version_id, ordinal)
  WHERE processing_version_id IS NOT NULL;
