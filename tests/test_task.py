import hashlib
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rescore import task as T


def runs_of(n: int) -> list[dict]:
    return [{"index": i, "run": f"r{i}", "config_path": f"configs/r{i}.yaml",
             "parquet_sha256": "0" * 64, "parquet_bytes": 1} for i in range(n)]


def manifest_of(*names: str) -> dict:
    runs = [{"index": i, "run": n, "config_path": f"configs/{n}.yaml",
             "parquet_sha256": "0" * 64, "parquet_bytes": 1} for i, n in enumerate(names)]
    return {"hf_repo": "org/ds", "hf_revision": "rev", "research_commit": "abc", "runs": runs}


def write_parquet(path: Path, similarity) -> bytes:
    frame = pd.DataFrame({"pair_id": ["a", "b"]})
    if similarity is not None:
        frame["feedback_similarity"] = similarity
    frame.to_parquet(path, index=False)
    return path.read_bytes()


def fake_research(root: Path, body: str) -> Path:
    """A research checkout whose scripts/run_audit.py is the given Python body."""
    (root / "scripts").mkdir(parents=True)
    (root / "reports" / "reference_metrics").mkdir(parents=True)
    (root / "eval" / "results").mkdir(parents=True)
    (root / "scripts" / "run_audit.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return root


# A run_audit.py stand-in that behaves like the real one on success: it reads the parquet
# from $HBM_OUTPUT_ROOT/outputs/raw, writes <results-dir>/<run>.json and a reference row.
FAKE_OK = """
    import json, os, sys
    from pathlib import Path
    args = sys.argv[1:]
    results = Path(args[args.index("--results-dir") + 1])
    config = args[args.index("--config") + 1]
    run = Path(config).stem
    raw = Path(os.environ["HBM_OUTPUT_ROOT"]) / "outputs" / "raw" / f"{run}.parquet"
    assert raw.exists(), raw
    assert "--score-only" in args
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{run}.json").write_text(json.dumps({"run_name": run, "config": config}))
    row = Path("reports") / "reference_metrics" / f"{run}.json"
    row.write_text(json.dumps({"run": run, "fresh": True}))
"""


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Replaces hf_hub_download with a local store: {"runs/<run>.parquet": bytes, ...}."""
    store: dict[str, bytes] = {}

    def download(repo_id, filename, repo_type, revision, local_dir):
        assert repo_type == "dataset"
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(store[filename])
        return str(target)

    monkeypatch.setattr(T, "hf_hub_download", download)
    return store


# --- selecting the run -------------------------------------------------------------------

def test_count_mismatch_names_both_numbers():
    with pytest.raises(T.TaskCountMismatch, match=r"10.*64|64.*10"):
        T.entry_for_task(runs_of(64), index=0, count=10)


def test_entry_for_task_returns_the_indexed_run():
    assert T.entry_for_task(runs_of(3), index=2, count=3)["run"] == "r2"


def test_subset_keeps_manifest_order():
    assert [r["run"] for r in T.select_runs(manifest_of("a", "b", "c"), ["c", "a"])] == ["a", "c"]


def test_no_subset_returns_all():
    assert len(T.select_runs(manifest_of("a", "b"), None)) == 2


def test_unknown_subset_name_raises():
    with pytest.raises(ValueError, match="zzz"):
        T.select_runs(manifest_of("a"), ["a", "zzz"])


# --- inputs ------------------------------------------------------------------------------

def test_fetch_inputs_places_files_where_run_audit_reads_them(tmp_path, fake_hub):
    data = write_parquet(tmp_path / "src.parquet", [0.5, 0.6])
    fake_hub["runs/Qwen3.5-x.parquet"] = data
    fake_hub["runs/Qwen3.5-x.meta.json"] = b'{"run_name": "Qwen3.5-x"}'
    entry = {"run": "Qwen3.5-x", "parquet_sha256": hashlib.sha256(data).hexdigest()}
    raw_dir = tmp_path / "out" / "outputs" / "raw"

    path = T.fetch_inputs(entry, "org/ds", "rev", raw_dir)

    assert path == raw_dir / "Qwen3.5-x.parquet"
    assert path.read_bytes() == data
    assert (raw_dir / "Qwen3.5-x.meta.json").exists()


def test_checksum_mismatch_raises(tmp_path, fake_hub):
    fake_hub["runs/a.parquet"] = b"tampered"
    fake_hub["runs/a.meta.json"] = b"{}"
    with pytest.raises(T.ChecksumMismatch, match="a"):
        T.fetch_inputs({"run": "a", "parquet_sha256": "0" * 64}, "org/ds", "rev", tmp_path)


def test_missing_similarity_cache_raises(tmp_path):
    p = tmp_path / "x.parquet"
    write_parquet(p, [np.nan, np.nan])
    with pytest.raises(T.MissingSimilarityCache):
        T.assert_similarity_cached(p)


def test_missing_similarity_column_raises(tmp_path):
    p = tmp_path / "x.parquet"
    write_parquet(p, None)
    with pytest.raises(T.MissingSimilarityCache):
        T.assert_similarity_cached(p)


def test_cached_similarity_passes(tmp_path):
    p = tmp_path / "x.parquet"
    write_parquet(p, [0.4, np.nan])
    T.assert_similarity_cached(p)


# --- scoring -----------------------------------------------------------------------------

def test_rescore_runs_score_only_against_output_root(tmp_path):
    research = fake_research(tmp_path / "research", FAKE_OK)
    out_root = tmp_path / "out"
    (out_root / "outputs" / "raw").mkdir(parents=True)
    write_parquet(out_root / "outputs" / "raw" / "a.parquet", [0.1, 0.2])

    result, returncode = T.rescore({"run": "a", "config_path": "configs/a.yaml"}, research,
                                   out_root, tmp_path / "results")

    assert (result, returncode) == (tmp_path / "results" / "a.json", 0)
    assert json.loads(result.read_text())["config"] == "configs/a.yaml"


def test_rescore_failure_carries_returncode(tmp_path):
    research = fake_research(tmp_path / "research",
                             "import sys\nsys.stderr.write('boom')\nsys.exit(1)\n")
    with pytest.raises(T.RescoreFailed) as err:
        T.rescore({"run": "a", "config_path": "c.yaml"}, research, tmp_path, tmp_path / "res")
    assert err.value.returncode == 1
    assert "boom" in err.value.stderr_tail


def test_stale_reference_row_not_published(tmp_path):
    # The image ships the research repo's committed rows. A run whose scoring does not
    # write a fresh row must not publish the committed one as if it were new.
    research = fake_research(tmp_path / "research", """
        import json, sys
        from pathlib import Path
        args = sys.argv[1:]
        results = Path(args[args.index("--results-dir") + 1])
        results.mkdir(parents=True, exist_ok=True)
        (results / "a.json").write_text("{}")
    """)
    (research / "reports" / "reference_metrics" / "a.json").write_text('{"stale": true}')

    result, _ = T.rescore({"run": "a", "config_path": "c.yaml"}, research, tmp_path / "o",
                          tmp_path / "res")
    T.publish("a", result, research, tmp_path / "dest", {"run": "a"})

    assert not (tmp_path / "dest" / "reference_metrics.json").exists()


def scoring_that_writes(parse_failure_rate: float, exit_code: int) -> str:
    return f"""
        import json, sys
        from pathlib import Path
        args = sys.argv[1:]
        results = Path(args[args.index("--results-dir") + 1])
        results.mkdir(parents=True, exist_ok=True)
        summary = {{"parse_failure_rate": {parse_failure_rate}}}
        (results / "a.json").write_text(json.dumps({{"summary": summary}}))
        sys.exit({exit_code})
    """


def test_unusable_run_is_returned_with_its_exit_code(tmp_path):
    # run_audit.py writes the record and then exits 1 when over half the responses did not
    # parse. That record is the study's result for the run, so it is published, not retried.
    research = fake_research(tmp_path / "research", scoring_that_writes(0.9999, 1))
    result, returncode = T.rescore({"run": "a", "config_path": "c.yaml"}, research,
                                   tmp_path / "o", tmp_path / "res")
    assert returncode == 1
    assert json.loads(result.read_text())["summary"]["parse_failure_rate"] == 0.9999


def test_failure_after_a_usable_record_still_raises(tmp_path):
    research = fake_research(tmp_path / "research", scoring_that_writes(0.1, 1))
    with pytest.raises(T.RescoreFailed):
        T.rescore({"run": "a", "config_path": "c.yaml"}, research, tmp_path / "o",
                  tmp_path / "res")


def test_results_dir_seeded_with_committed_records_except_this_run(tmp_path):
    # reference_metrics pairs a mitigated run with its baseline by reading --results-dir.
    # This run's own committed record must not be there, or a failed scoring could hand it
    # back as the fresh one.
    research = fake_research(tmp_path / "research", """
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        results = Path(args[args.index("--results-dir") + 1])
        assert (results / "base.json").read_text() == '{"run_name": "base"}'
        assert not (results / "a.json").exists()
        (results / "a.json").write_text("{}")
    """)
    (research / "eval" / "results" / "base.json").write_text('{"run_name": "base"}')
    (research / "eval" / "results" / "a.json").write_text('{"committed": true}')

    result, returncode = T.rescore({"run": "a", "config_path": "c.yaml"}, research,
                                   tmp_path / "o", tmp_path / "res")

    assert returncode == 0 and result.read_text() == "{}"


# --- publishing --------------------------------------------------------------------------

def test_publish_clears_old_provenance_before_writing(tmp_path, monkeypatch):
    # Re-executing a RUN_ID writes over an earlier attempt. If this attempt dies before its
    # own provenance.json, the old one must not make the half-written run look complete.
    research = fake_research(tmp_path / "research", "")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "provenance.json").write_text('{"image_ref": "old"}')
    (dest / "reference_metrics.json").write_text('{"old": true}')
    result = tmp_path / "a.json"
    result.write_text("{}")
    real = T._write_bytes

    def dies_at_provenance(path, data):
        if path.name == "provenance.json":
            raise OSError("killed")
        real(path, data)

    monkeypatch.setattr(T, "_write_bytes", dies_at_provenance)
    with pytest.raises(OSError):
        T.publish("a", result, research, dest, {"run": "a"})

    assert not (dest / "provenance.json").exists()
    assert not (dest / "reference_metrics.json").exists()


def test_publish_writes_provenance_last(tmp_path, monkeypatch):
    research = fake_research(tmp_path / "research", "")
    (research / "reports" / "reference_metrics" / "a.json").write_text("{}")
    result = tmp_path / "a.json"
    result.write_text('{"x": 1}')
    order = []
    real = T._write_bytes
    monkeypatch.setattr(T, "_write_bytes", lambda p, d: (order.append(p.name), real(p, d)))

    T.publish("a", result, research, tmp_path / "dest", {"run": "a"})

    assert order == ["result.json", "reference_metrics.json", "provenance.json"]


def test_publish_copies_result_bytes_unchanged(tmp_path):
    research = fake_research(tmp_path / "research", "")
    result = tmp_path / "a.json"
    result.write_bytes(b'{"b": 2,  "a": 1}\n')
    T.publish("a", result, research, tmp_path / "dest", {"run": "a"})
    assert (tmp_path / "dest" / "result.json").read_bytes() == b'{"b": 2,  "a": 1}\n'


def test_log_event_prints_one_json_line(capsys):
    T.log_event("ERROR", "failed", run="a", task_index=3)
    line = capsys.readouterr().out
    assert line.count("\n") == 1
    assert json.loads(line) == {"severity": "ERROR", "message": "failed", "run": "a",
                                "task_index": 3}


# --- the whole task ----------------------------------------------------------------------

def task_env(monkeypatch, tmp_path, manifest: dict, **extra) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    env = {"RUN_ID": "test-1", "OUTPUT_DIR": str(tmp_path / "bucket"),
           "RESEARCH_DIR": str(tmp_path / "research"), "MANIFEST": str(path),
           "CLOUD_RUN_TASK_INDEX": "0", "CLOUD_RUN_TASK_COUNT": "1", "IMAGE_REF": "img@sha"}
    env.update(extra)
    for name in ("RUN_SUBSET", "CLOUD_RUN_TASK_ATTEMPT", "CLOUD_RUN_EXECUTION"):
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return tmp_path / "bucket" / "runs" / "test-1"


def test_main_publishes_layout_and_provenance(tmp_path, monkeypatch, fake_hub):
    fake_research(tmp_path / "research", FAKE_OK)
    data = write_parquet(tmp_path / "src.parquet", [0.3, 0.4])
    fake_hub["runs/a.parquet"] = data
    fake_hub["runs/a.meta.json"] = b"{}"
    manifest = manifest_of("a")
    manifest["runs"][0]["parquet_sha256"] = hashlib.sha256(data).hexdigest()
    run_root = task_env(monkeypatch, tmp_path, manifest)

    assert T.main() == 0

    dest = run_root / "a"
    assert sorted(p.name for p in dest.iterdir()) == ["provenance.json",
                                                      "reference_metrics.json", "result.json"]
    prov = json.loads((dest / "provenance.json").read_text())
    assert {k: prov[k] for k in ("run_id", "run", "task_index", "task_attempt", "execution",
                                 "hf_repo", "hf_revision", "research_commit", "image_ref",
                                 "run_audit_returncode")} == {
        "run_id": "test-1", "run": "a", "task_index": 0, "task_attempt": 0,
        "execution": "local", "hf_repo": "org/ds", "hf_revision": "rev",
        "research_commit": "abc", "image_ref": "img@sha", "run_audit_returncode": 0}
    assert {"started_at", "finished_at", "seconds"} <= prov.keys()


def test_main_count_mismatch_fails_without_output(tmp_path, monkeypatch, capsys):
    run_root = task_env(monkeypatch, tmp_path, manifest_of("a", "b"),
                        CLOUD_RUN_TASK_COUNT="1")
    assert T.main() == 1
    assert not run_root.exists()
    err = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert err["severity"] == "ERROR" and "2" in err["message"] and "1" in err["message"]


def test_main_subset_selects_by_name(tmp_path, monkeypatch, fake_hub):
    fake_research(tmp_path / "research", FAKE_OK)
    data = write_parquet(tmp_path / "src.parquet", [0.3, 0.4])
    fake_hub["runs/b.parquet"] = data
    fake_hub["runs/b.meta.json"] = b"{}"
    manifest = manifest_of("a", "b")
    manifest["runs"][1]["parquet_sha256"] = hashlib.sha256(data).hexdigest()
    run_root = task_env(monkeypatch, tmp_path, manifest, RUN_SUBSET="b")

    assert T.main() == 0
    assert [p.name for p in run_root.iterdir()] == ["b"]
