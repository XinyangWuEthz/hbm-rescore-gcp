-- Which result fields moved between the committed results and one re-scoring run, and in
-- how many runs. Group keys and list indices are collapsed so one field is one row.
--   bq query --use_legacy_sql=false --parameter=run_id::<RUN_ID> "$(cat bq/drift_summary.sql)"
SELECT
  kind,
  REGEXP_REPLACE(REGEXP_REPLACE(path, r'^/groups/[^/]+', '/groups/*'), r'\[\d+\]', '[]') AS field,
  COUNT(DISTINCT run) AS runs,
  COUNT(*) AS diffs
FROM `hbm_rescore.drift`
WHERE run_id = @run_id AND kind != 'manual_review'
GROUP BY kind, field
ORDER BY runs DESC, diffs DESC
LIMIT 50
