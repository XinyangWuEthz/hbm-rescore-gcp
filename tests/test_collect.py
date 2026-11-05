import json
from pathlib import Path

import pytest
from google.cloud import bigquery

from rescore import collect as C
from rescore.drift import Diff

ROOT = Path(__file__).resolve().parents[1]


def result_of(run: str, **summary) -> dict:
    base = {"n_total": 100, "n_decided": 98, "refusal_rate": 0.0, "parse_failure_rate": 0.02,
            "population_acceptance_rate": 0.3, "attr_free_acceptance_rate": 0.31,
            "reference_agreement": 0.8, "attr_free_reference_agreement": 0.81,
            "attribute_mention_rate": 0.0, "mean_feedback_similarity": 0.7, "n_tests": 12,
            "usable": True}
    base.update(summary)
    return {"run_name": run,
            "meta": {"model": "Qwen/Qwen3.5-4B", "lang": "en", "mitigation": {"family": "none"},
                     "prompt_strategy": "baseline"},
            "summary": base, "groups": {}, "manual_review": []}


def provenance_of(run: str, run_id: str = "rid") -> dict:
    return {"run_id": run_id, "run": run, "task_index": 0, "task_attempt": 0,
            "execution": "rescore-x", "hf_repo": "org/ds", "hf_revision": "rev",
            "research_commit": "abc", "image_ref": "img@sha",
            "started_at": "2026-11-05T10:00:00+00:00",
            "finished_at": "2026-11-05T10:00:05+00:00", "seconds": 5.0}


def publish_run(run_root: Path, run: str, result: dict, with_provenance: bool = True) -> None:
    d = run_root / run
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(result))
    if with_provenance:
        (d / "provenance.json").write_text(json.dumps(provenance_of(run, run_root.name)))


def research_with(root: Path, committed: dict[str, dict]) -> Path:
    results = root / "eval" / "results"
    results.mkdir(parents=True)
    for run, record in committed.items():
        (results / f"{run}.json").write_text(json.dumps(record))
    return root


class FakeJob:
    def __init__(self, rows=()):
        self._rows = rows

    def result(self):
        return self._rows


class FakeRow:
    def __init__(self, n):
        self.n = n


class FakeClient:
    def __init__(self, existing: int = 0):
        self.existing = existing
        self.calls: list[tuple] = []

    def query(self, sql, job_config=None):
        params = {p.name: p.value for p in job_config.query_parameters}
        if sql.lstrip().upper().startswith("DELETE"):
            self.calls.append(("delete", sql, params))
            return FakeJob()
        self.calls.append(("count", sql, params))
        return FakeJob([FakeRow(self.existing)])

    def load_table_from_uri(self, uri, table, job_config=None):
        self.calls.append(("load", uri, table, job_config.source_format,
                           job_config.write_disposition))
        return FakeJob()


# --- gathering ---------------------------------------------------------------------------

def test_gather_ignores_dir_without_provenance(tmp_path):
    publish_run(tmp_path, "a", result_of("a"))
    publish_run(tmp_path, "b", result_of("b"), with_provenance=False)
    with pytest.raises(C.IncompleteRun) as err:
        C.gather(tmp_path, ["a", "b"])
    assert err.value.missing == ["b"]


def test_gather_returns_dirs_when_complete(tmp_path):
    publish_run(tmp_path, "a", result_of("a"))
    assert C.gather(tmp_path, ["a"]) == {"a": tmp_path / "a"}


# --- rows --------------------------------------------------------------------------------

def test_run_row_fields():
    diffs = [Diff("/x", "changed", 1, 2), Diff("/y", "float_noise", 0.1, 0.1),
             Diff("/manual_review[0]/a", "manual_review", "p", "q")]
    row = C.run_row(result_of("a"), provenance_of("a"), diffs)
    assert row["family"] == "baseline"
    assert (row["model"], row["lang"], row["prompt_strategy"]) == ("Qwen/Qwen3.5-4B", "en",
                                                                    "baseline")
    assert row["parse_failure_rate"] == 0.02 and row["usable"] is True
    assert (row["drift_changed"], row["drift_float_noise"], row["drift_manual_review"],
            row["drift_added"], row["drift_removed"]) == (1, 1, 1, 0, 0)
    assert row["run_id"] == "rid" and row["image_ref"] == "img@sha"


def test_mitigation_family_kept():
    result = result_of("a")
    result["meta"]["mitigation"] = {"family": "scrub", "mode": "lexical"}
    assert C.run_row(result, provenance_of("a"), [])["family"] == "scrub"


def test_nan_summary_value_becomes_null():
    row = C.run_row(result_of("a", mean_feedback_similarity=float("nan")), provenance_of("a"),
                    [])
    assert row["mean_feedback_similarity"] is None
    json.dumps(row, allow_nan=False)


def test_drift_rows_serialise_values():
    rows = C.drift_rows("rid", "a", [Diff("/s/usable", "added", None, True)])
    assert rows == [{"run_id": "rid", "run": "a", "path": "/s/usable", "kind": "added",
                     "committed": "null", "rescored": "true"}]


def test_row_keys_match_schema():
    runs_schema = [f.name for f in C.load_schema(ROOT / "bq" / "runs.schema.json")]
    drift_schema = [f.name for f in C.load_schema(ROOT / "bq" / "drift.schema.json")]
    assert list(C.run_row(result_of("a"), provenance_of("a"), [])) == runs_schema
    assert list(C.drift_rows("r", "a", [Diff("/x", "changed", 1, 2)])[0]) == drift_schema


def test_schema_types_are_bigquery_types():
    for name in ("runs", "drift"):
        for field in C.load_schema(ROOT / "bq" / f"{name}.schema.json"):
            assert isinstance(field, bigquery.SchemaField)
            assert field.field_type in {"STRING", "INTEGER", "FLOAT", "BOOLEAN", "TIMESTAMP"}


# --- the whole job -----------------------------------------------------------------------

def collect_env(monkeypatch, tmp_path, runs: list[str], **extra) -> Path:
    manifest = {"hf_repo": "org/ds", "hf_revision": "rev", "research_commit": "abc",
                "runs": [{"index": i, "run": r, "config_path": f"c/{r}.yaml",
                          "parquet_sha256": "0" * 64, "parquet_bytes": 1}
                         for i, r in enumerate(runs)]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    env = {"RUN_ID": "rid", "OUTPUT_DIR": str(tmp_path / "bucket"),
           "RESEARCH_DIR": str(tmp_path / "research"),
           "MANIFEST": str(tmp_path / "manifest.json"),
           "BQ_DATASET": "hbm_rescore", "GCS_BUCKET": "my-bucket"}
    env.update(extra)
    for name in ("RUN_SUBSET", "REPLACE", "BQ_DATASET", "GCS_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        if v is not None:
            monkeypatch.setenv(k, v)
    return tmp_path / "bucket" / "runs" / "rid"


@pytest.fixture
def two_runs(tmp_path):
    run_root = tmp_path / "bucket" / "runs" / "rid"
    for run in ("a", "b"):
        publish_run(run_root, run, result_of(run))
    stale = result_of("b", attribute_mention_rate=0.5)
    del stale["summary"]["usable"]
    research_with(tmp_path / "research", {"a": result_of("a"), "b": stale})
    return run_root


def use_client(monkeypatch, client: FakeClient) -> FakeClient:
    monkeypatch.setattr(C, "make_client", lambda: client)
    return client


def test_main_writes_tables_and_loads_both(tmp_path, monkeypatch, two_runs):
    collect_env(monkeypatch, tmp_path, ["a", "b"])
    client = use_client(monkeypatch, FakeClient(existing=0))

    assert C.main() == 0

    runs = [json.loads(x) for x in (two_runs / "_tables" / "runs.jsonl").read_text().splitlines()]
    drift = [json.loads(x) for x in (two_runs / "_tables" / "drift.jsonl").read_text().splitlines()]
    assert [r["run"] for r in runs] == ["a", "b"]
    assert [(r["drift_changed"], r["drift_added"]) for r in runs] == [(0, 0), (1, 1)]
    assert {(d["run"], d["path"], d["kind"]) for d in drift} == {
        ("b", "/summary/attribute_mention_rate", "changed"), ("b", "/summary/usable", "added")}
    loads = [c for c in client.calls if c[0] == "load"]
    assert loads == [
        ("load", "gs://my-bucket/runs/rid/_tables/runs.jsonl", "hbm_rescore.runs",
         "NEWLINE_DELIMITED_JSON", "WRITE_APPEND"),
        ("load", "gs://my-bucket/runs/rid/_tables/drift.jsonl", "hbm_rescore.drift",
         "NEWLINE_DELIMITED_JSON", "WRITE_APPEND")]


def test_rerun_same_run_id_refused(tmp_path, monkeypatch, two_runs, capsys):
    collect_env(monkeypatch, tmp_path, ["a", "b"])
    client = use_client(monkeypatch, FakeClient(existing=2))

    assert C.main() == 1

    assert not [c for c in client.calls if c[0] in ("load", "delete")]
    err = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert err["severity"] == "ERROR" and err["error"] == "AlreadyLoaded"


def test_replace_deletes_before_load(tmp_path, monkeypatch, two_runs):
    collect_env(monkeypatch, tmp_path, ["a", "b"], REPLACE="1")
    client = use_client(monkeypatch, FakeClient(existing=2))

    assert C.main() == 0

    kinds = [c[0] for c in client.calls if c[0] != "count"]
    assert kinds == ["delete", "delete", "load", "load"]
    deletes = [c for c in client.calls if c[0] == "delete"]
    assert all(c[2] == {"run_id": "rid"} for c in deletes)


def test_main_without_dataset_writes_tables_only(tmp_path, monkeypatch, two_runs):
    collect_env(monkeypatch, tmp_path, ["a", "b"], BQ_DATASET=None, GCS_BUCKET=None)
    monkeypatch.setattr(C, "make_client", lambda: pytest.fail("no client in local mode"))

    assert C.main() == 0
    assert (two_runs / "_tables" / "runs.jsonl").exists()


def test_incomplete_run_fails_before_writing(tmp_path, monkeypatch):
    run_root = collect_env(monkeypatch, tmp_path, ["a", "b"])
    publish_run(run_root, "a", result_of("a"))
    research_with(tmp_path / "research", {"a": result_of("a"), "b": result_of("b")})
    use_client(monkeypatch, FakeClient())

    assert C.main() == 1
    assert not (run_root / "_tables").exists()
