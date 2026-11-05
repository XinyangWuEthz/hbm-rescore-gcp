"""Builds the pinned list of runs that the Cloud Run job re-scores.

One Cloud Run task handles one run, chosen by its position in this list, so the list has to
be fixed before the job starts: the same index must mean the same run on every attempt. It
pins the Hugging Face dataset revision the inputs come from and the research-code commit
that scores them.

    python -m rescore.manifest --hf-repo <repo> --revision <sha> \
        --research-commit <sha> --out manifest/runs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PARQUET_PREFIX = "runs/"
PARQUET_SUFFIX = ".parquet"


def build_manifest(
    files: list[dict],
    metas: dict[str, dict],
    hf_repo: str,
    hf_revision: str,
    research_commit: str,
) -> dict:
    """Returns the manifest for the parquet files in `files`, sorted by run name.

    `files` items are {"path": "runs/<run>.parquet", "size": int, "sha256": str}; other
    files under runs/ are ignored. `metas` maps a run name to its meta.json contents.
    """
    entries = []
    for item in files:
        path = item["path"]
        if not (path.startswith(PARQUET_PREFIX) and path.endswith(PARQUET_SUFFIX)):
            continue
        run = path[len(PARQUET_PREFIX):-len(PARQUET_SUFFIX)]
        meta = metas.get(run)
        if meta is None:
            raise ValueError(f"run {run!r} has a parquet but no meta.json")
        config_path = meta.get("config_path")
        if not config_path:
            raise ValueError(f"meta.json of run {run!r} has no config_path")
        entries.append({"run": run, "config_path": config_path,
                        "parquet_sha256": item["sha256"], "parquet_bytes": item["size"]})

    entries.sort(key=lambda e: e["run"])
    runs = [{"index": i, **e} for i, e in enumerate(entries)]
    return {"hf_repo": hf_repo, "hf_revision": hf_revision,
            "research_commit": research_commit, "runs": runs}


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _list_hub(hf_repo: str, revision: str) -> tuple[list[dict], dict[str, dict]]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files, metas = [], {}
    for entry in api.list_repo_tree(hf_repo, path_in_repo="runs", repo_type="dataset",
                                    revision=revision):
        path = entry.path
        if path.endswith(PARQUET_SUFFIX):
            if entry.lfs is None:
                raise ValueError(f"{path} is not stored in LFS; no sha256 to pin")
            files.append({"path": path, "size": entry.size, "sha256": entry.lfs.sha256})
        elif path.endswith(".meta.json"):
            run = path[len(PARQUET_PREFIX):-len(".meta.json")]
            local = hf_hub_download(hf_repo, path, repo_type="dataset", revision=revision)
            metas[run] = json.loads(Path(local).read_text(encoding="utf-8"))
    return files, metas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--research-commit", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    files, metas = _list_hub(args.hf_repo, args.revision)
    manifest = build_manifest(files, metas, args.hf_repo, args.revision, args.research_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(manifest['runs'])} runs")


if __name__ == "__main__":
    main()
