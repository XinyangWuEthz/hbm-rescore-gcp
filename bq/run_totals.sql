-- One line per re-scoring run: how many runs loaded, how many usable, and how much drift.
--   bq query --use_legacy_sql=false --parameter=run_id::<RUN_ID> "$(sed '/^--/d' bq/run_totals.sql)"
SELECT
  COUNT(*) AS runs,
  COUNTIF(usable) AS usable,
  SUM(drift_changed) AS changed,
  SUM(drift_added) AS added,
  SUM(drift_removed) AS removed,
  ROUND(SUM(seconds)) AS task_seconds
FROM `hbm_rescore.runs`
WHERE run_id = @run_id
