"""Entry point of the `collect` Cloud Run job: summarise one re-scoring run into BigQuery.

Runs after every `rescore` task of a RUN_ID has finished. It refuses to load anything while
a selected run lacks provenance.json, compares each re-scored result with the one the
research repository committed (rescore.drift), writes two newline-delimited JSON files next
to the results, and loads them into BigQuery with load jobs. A RUN_ID that is already in
BigQuery is refused unless REPLACE=1, which deletes its rows first.

Without BQ_DATASET the job stops after writing the two files, which is how it runs locally.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from rescore.drift import KINDS, Diff, compare
from rescore.manifest import load_manifest
from rescore.task import log_event, select_runs

PROVENANCE_FIELDS = ("run_id", "run", "task_index", "task_attempt", "execution", "hf_repo",
                     "hf_revision", "research_commit", "image_ref", "started_at",
                     "finished_at", "seconds")
SUMMARY_FIELDS = ("n_total", "n_decided", "refusal_rate", "parse_failure_rate",
                  "population_acceptance_rate", "attr_free_acceptance_rate",
                  "reference_agreement", "attr_free_reference_agreement",
                  "attribute_mention_rate", "mean_feedback_similarity", "n_tests", "usable")
TABLES = ("runs", "drift")
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "bq"


class IncompleteRun(RuntimeError):
    def __init__(self, missing: list[str]):
        super().__init__(f"{len(missing)} runs have no provenance.json yet: "
                         f"{', '.join(missing[:10])}")
        self.missing = missing


class AlreadyLoaded(RuntimeError):
    pass


def gather(run_root: Path, expected: list[str]) -> dict[str, Path]:
    missing = [run for run in expected if not (run_root / run / "provenance.json").is_file()]
    if missing:
        raise IncompleteRun(missing)
    return {run: run_root / run for run in expected}


def _clean(value):
    """BigQuery's JSON loader rejects NaN and infinity; they mean "no value" here."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_row(result: dict, provenance: dict, diffs: list[Diff]) -> dict:
    meta = result.get("meta", {})
    summary = result.get("summary", {})
    family = (meta.get("mitigation") or {}).get("family")
    counts = Counter(d.kind for d in diffs)
    return {
        **{k: provenance.get(k) for k in PROVENANCE_FIELDS},
        "model": meta.get("model"),
        "lang": meta.get("lang"),
        "family": "baseline" if family in (None, "", "none") else family,
        "prompt_strategy": meta.get("prompt_strategy"),
        **{k: _clean(summary.get(k)) for k in SUMMARY_FIELDS},
        **{f"drift_{kind}": counts.get(kind, 0) for kind in KINDS},
    }


def drift_rows(run_id: str, run: str, diffs: list[Diff]) -> list[dict]:
    return [{"run_id": run_id, "run": run, "path": d.path, "kind": d.kind,
             "committed": json.dumps(d.committed, default=str),
             "rescored": json.dumps(d.rescored, default=str)} for d in diffs]


def load_schema(path: Path) -> list:
    from google.cloud import bigquery

    return [bigquery.SchemaField.from_api_repr(f)
            for f in json.loads(Path(path).read_text(encoding="utf-8"))]


def make_client():
    from google.cloud import bigquery

    return bigquery.Client()


def existing_rows(client, table: str, run_id: str) -> int:
    from google.cloud import bigquery

    config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)])
    rows = client.query(f"SELECT COUNT(*) AS n FROM `{table}` WHERE run_id = @run_id",
                        job_config=config).result()
    return next(iter(rows)).n


def delete_rows(client, table: str, run_id: str) -> None:
    from google.cloud import bigquery

    config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)])
    client.query(f"DELETE FROM `{table}` WHERE run_id = @run_id", job_config=config).result()


def load_table(client, uri: str, table: str, schema_path: Path) -> None:
    from google.cloud import bigquery

    config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        schema=load_schema(schema_path),
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    client.load_table_from_uri(uri, table, job_config=config).result()


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                            for r in rows), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    try:
        run_id = os.environ["RUN_ID"]
        output_dir = Path(os.environ["OUTPUT_DIR"])
        research_dir = Path(os.environ.get("RESEARCH_DIR", "/app/research"))
        manifest = load_manifest(Path(os.environ.get("MANIFEST", "/app/manifest/runs.json")))
        subset_env = os.environ.get("RUN_SUBSET", "").strip()
        subset = [s.strip() for s in subset_env.split(",") if s.strip()] or None
        dataset = os.environ.get("BQ_DATASET", "").strip()
        replace = os.environ.get("REPLACE", "0") == "1"

        expected = [r["run"] for r in select_runs(manifest, subset)]
        run_root = output_dir / "runs" / run_id
        dirs = gather(run_root, expected)

        runs, drift = [], []
        for run, d in dirs.items():
            result = _read(d / "result.json")
            committed = _read(research_dir / "eval" / "results" / f"{run}.json")
            diffs = compare(committed, result)
            runs.append(run_row(result, _read(d / "provenance.json"), diffs))
            drift.extend(drift_rows(run_id, run, diffs))
        tables = run_root / "_tables"
        _write_jsonl(runs, tables / "runs.jsonl")
        _write_jsonl(drift, tables / "drift.jsonl")
        log_event("INFO", "tables written", run_id=run_id, runs=len(runs),
                  drift_rows=len(drift))

        if not dataset:
            return 0
        bucket = os.environ["GCS_BUCKET"]
        client = make_client()
        names = {t: f"{dataset}.{t}" for t in TABLES}
        present = {t: existing_rows(client, names[t], run_id) for t in TABLES}
        if any(present.values()):
            if not replace:
                raise AlreadyLoaded(f"RUN_ID {run_id} is already in BigQuery ({present}); "
                                    f"set REPLACE=1 to delete and reload it")
            for t in TABLES:
                delete_rows(client, names[t], run_id)
        for t in TABLES:
            load_table(client, f"gs://{bucket}/runs/{run_id}/_tables/{t}.jsonl", names[t],
                       SCHEMA_DIR / f"{t}.schema.json")
        log_event("INFO", "loaded into BigQuery", run_id=run_id, dataset=dataset,
                  replaced=bool(any(present.values())))
        return 0
    except (KeyError, ValueError, FileNotFoundError, IncompleteRun, AlreadyLoaded) as exc:
        log_event("ERROR", str(exc), error=type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
