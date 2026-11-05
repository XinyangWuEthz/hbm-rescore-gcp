"""Entry point of the `rescore` Cloud Run job: one task re-scores one published run.

Cloud Run starts CLOUD_RUN_TASK_COUNT copies of this container and gives each its
CLOUD_RUN_TASK_INDEX. The task looks its run up in the pinned manifest, downloads that run's
cached generations from Hugging Face at the pinned revision, checks the sha256, runs the
research repository's `run_audit.py --score-only`, and copies the outputs to
$OUTPUT_DIR/runs/<RUN_ID>/<run>/. In the cloud OUTPUT_DIR is a Cloud Storage bucket mounted
as a volume, so this code only ever writes to a local path.

provenance.json is written last. Its presence is what marks a run as complete, so a task
that died half-way through publishing is never mistaken for a finished one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

from rescore.manifest import load_manifest

#: Where run_audit.py reads cached generations, relative to HBM_OUTPUT_ROOT. Every config in
#: the manifest resolves `raw_dir` to this (checked 2026-11-05 for all 64).
RAW_SUBDIR = Path("outputs") / "raw"
STDERR_TAIL_CHARS = 4000
#: run_audit.py writes the record and then exits 1 above this share of unparsable responses
#: (`MAX_PARSE_FAILURE`, src/hiring_bias_mitigation/eval/audit.py at 7e17753). Such a run is
#: the study's result, marked unusable, not a failed task.
PARSE_FAILURE_ABORT = 0.5


class TaskCountMismatch(ValueError):
    pass


class ChecksumMismatch(ValueError):
    pass


class MissingSimilarityCache(ValueError):
    pass


class RescoreFailed(RuntimeError):
    def __init__(self, returncode: int, stderr_tail: str):
        super().__init__(f"run_audit.py exited {returncode}: {stderr_tail[-500:]}")
        self.returncode = returncode
        self.stderr_tail = stderr_tail


def log_event(severity: str, message: str, **fields) -> None:
    """Prints one JSON line; Cloud Logging turns it into a structured entry."""
    print(json.dumps({"severity": severity, "message": message, **fields},
                     ensure_ascii=False, default=str), flush=True)


def select_runs(manifest: dict, subset: list[str] | None) -> list[dict]:
    runs = manifest["runs"]
    if subset is None:
        return list(runs)
    known = {r["run"] for r in runs}
    unknown = [name for name in subset if name not in known]
    if unknown:
        raise ValueError(f"RUN_SUBSET names runs not in the manifest: {', '.join(unknown)}")
    wanted = set(subset)
    return [r for r in runs if r["run"] in wanted]


def entry_for_task(runs: list[dict], index: int, count: int) -> dict:
    if count != len(runs):
        raise TaskCountMismatch(
            f"the job started {count} tasks but {len(runs)} runs are selected; set --tasks "
            f"to {len(runs)} or RUN_SUBSET to {count} run names"
        )
    return runs[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_inputs(entry: dict, hf_repo: str, revision: str, raw_dir: Path) -> Path:
    """Downloads <run>.parquet and <run>.meta.json into raw_dir and verifies the parquet."""
    run = entry["run"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=raw_dir.parent) as staging:
        for suffix in (".parquet", ".meta.json"):
            local = hf_hub_download(hf_repo, f"runs/{run}{suffix}", repo_type="dataset",
                                    revision=revision, local_dir=staging)
            shutil.move(local, raw_dir / f"{run}{suffix}")
    parquet = raw_dir / f"{run}.parquet"
    actual = _sha256(parquet)
    if actual != entry["parquet_sha256"]:
        raise ChecksumMismatch(
            f"{run}.parquet sha256 {actual} does not match the manifest "
            f"{entry['parquet_sha256']}"
        )
    return parquet


def assert_similarity_cached(parquet: Path) -> None:
    """Fails unless the parquet carries feedback_similarity, so no encoder is ever needed."""
    import pyarrow.parquet as pq

    if "feedback_similarity" not in pq.read_schema(parquet).names:
        raise MissingSimilarityCache(f"{parquet.name} has no feedback_similarity column")
    column = pq.read_table(parquet, columns=["feedback_similarity"]).column(0)
    if column.null_count == len(column) or not _has_number(column):
        raise MissingSimilarityCache(f"{parquet.name} has no cached feedback_similarity")


def _has_number(column) -> bool:
    import pyarrow.compute as pc

    return bool(pc.any(pc.invert(pc.is_nan(column.cast("double")))).as_py())


def _seed_results(research_dir: Path, results_dir: Path, run: str) -> None:
    """Copies the committed records except this run's into results_dir.

    reference_metrics finds a mitigated run's baseline by reading --results-dir. This run's
    own committed record stays out, so the file scoring leaves behind is always fresh.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    for record in sorted((research_dir / "eval" / "results").glob("*.json")):
        if record.name != f"{run}.json":
            shutil.copyfile(record, results_dir / record.name)


def _unusable(result: Path) -> bool:
    if not result.is_file():
        return False
    rate = json.loads(result.read_text(encoding="utf-8")).get("summary", {}).get(
        "parse_failure_rate")
    return isinstance(rate, int | float) and rate > PARSE_FAILURE_ABORT


def rescore(entry: dict, research_dir: Path, output_root: Path,
            results_dir: Path) -> tuple[Path, int]:
    """Returns the fresh record and run_audit.py's exit code (0, or 1 for an unusable run)."""
    run = entry["run"]
    # The image ships the research repository's committed rows. Remove this run's, so the
    # row published afterwards can only be one that this scoring wrote.
    (research_dir / "reports" / "reference_metrics" / f"{run}.json").unlink(missing_ok=True)
    _seed_results(research_dir, results_dir, run)
    cmd = [sys.executable, "scripts/run_audit.py", "--config", entry["config_path"],
           "--score-only", "--results-dir", str(results_dir)]
    env = {**os.environ, "HBM_OUTPUT_ROOT": str(output_root)}
    proc = subprocess.run(cmd, cwd=research_dir, env=env, capture_output=True, text=True,
                          check=False)
    result = results_dir / f"{run}.json"
    if proc.returncode != 0 and not (proc.returncode == 1 and _unusable(result)):
        raise RescoreFailed(proc.returncode, proc.stderr[-STDERR_TAIL_CHARS:])
    return result, proc.returncode


def _write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def publish(run: str, result_json: Path, research_dir: Path, dest_dir: Path,
            provenance: dict) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    # An earlier attempt for this RUN_ID may have left these. Until this attempt writes its
    # own provenance.json the run must read as incomplete, and an old reference row must
    # never sit next to a new result.
    (dest_dir / "provenance.json").unlink(missing_ok=True)
    (dest_dir / "reference_metrics.json").unlink(missing_ok=True)
    _write_bytes(dest_dir / "result.json", result_json.read_bytes())
    row = research_dir / "reports" / "reference_metrics" / f"{run}.json"
    if row.is_file():
        _write_bytes(dest_dir / "reference_metrics.json", row.read_bytes())
    _write_bytes(dest_dir / "provenance.json",
                 (json.dumps(provenance, indent=2) + "\n").encode("utf-8"))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main() -> int:
    task_index = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
    context = {"task_index": task_index,
               "task_attempt": int(os.environ.get("CLOUD_RUN_TASK_ATTEMPT", "0")),
               "execution": os.environ.get("CLOUD_RUN_EXECUTION", "local")}
    try:
        run_id = os.environ["RUN_ID"]
        output_dir = Path(os.environ["OUTPUT_DIR"])
        research_dir = Path(os.environ.get("RESEARCH_DIR", "/app/research"))
        manifest = load_manifest(Path(os.environ.get("MANIFEST", "/app/manifest/runs.json")))
        subset_env = os.environ.get("RUN_SUBSET", "").strip()
        subset = [s.strip() for s in subset_env.split(",") if s.strip()] or None
        runs = select_runs(manifest, subset)
        entry = entry_for_task(runs, task_index,
                               int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1")))
        run = entry["run"]
        context["run"] = run
        log_event("INFO", "re-scoring", run_id=run_id, **context)

        started, t0 = _now(), time.monotonic()
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            parquet = fetch_inputs(entry, manifest["hf_repo"], manifest["hf_revision"],
                                   work / RAW_SUBDIR)
            assert_similarity_cached(parquet)
            result, returncode = rescore(entry, research_dir, work, work / "results")
            if returncode:
                log_event("WARNING", "scored, but run_audit.py marks the run unusable",
                          run_id=run_id, run_audit_returncode=returncode, **context)
            provenance = {"run_id": run_id, "run": run, **context,
                          "hf_repo": manifest["hf_repo"],
                          "hf_revision": manifest["hf_revision"],
                          "research_commit": manifest["research_commit"],
                          "image_ref": os.environ.get("IMAGE_REF", "unknown"),
                          "started_at": started, "finished_at": _now(),
                          "seconds": round(time.monotonic() - t0, 3),
                          "run_audit_returncode": returncode}
            publish(run, result, research_dir, output_dir / "runs" / run_id / run, provenance)
        log_event("INFO", "published", run_id=run_id, seconds=provenance["seconds"], **context)
        return 0
    except (KeyError, ValueError, RescoreFailed) as exc:
        log_event("ERROR", str(exc), error=type(exc).__name__, **context)
        return 1


if __name__ == "__main__":
    sys.exit(main())
