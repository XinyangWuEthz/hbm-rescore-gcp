import pytest

from rescore.manifest import build_manifest


def f(path: str, size: int = 10, sha256: str = "0" * 64) -> dict:
    return {"path": path, "size": size, "sha256": sha256}


def build(files, metas):
    return build_manifest(files=files, metas=metas, hf_repo="r", hf_revision="rev",
                          research_commit="c")


def test_runs_sorted_and_indexed():
    m = build([f("runs/b.parquet"), f("runs/a.parquet")],
              {"a": {"config_path": "configs/a.yaml"}, "b": {"config_path": "configs/b.yaml"}})
    assert [r["run"] for r in m["runs"]] == ["a", "b"]
    assert [r["index"] for r in m["runs"]] == [0, 1]


def test_entry_fields_and_pins():
    m = build([f("runs/a.parquet", size=123, sha256="ab" * 32)],
              {"a": {"config_path": "configs/a.yaml"}})
    assert (m["hf_repo"], m["hf_revision"], m["research_commit"]) == ("r", "rev", "c")
    assert m["runs"] == [{"index": 0, "run": "a", "config_path": "configs/a.yaml",
                          "parquet_sha256": "ab" * 32, "parquet_bytes": 123}]


def test_non_parquet_files_ignored():
    m = build([f("runs/a.parquet"), f("runs/a.meta.json")], {"a": {"config_path": "c.yaml"}})
    assert [r["run"] for r in m["runs"]] == ["a"]


def test_missing_meta_raises():
    with pytest.raises(ValueError, match="b"):
        build([f("runs/a.parquet"), f("runs/b.parquet")], {"a": {"config_path": "c.yaml"}})


def test_missing_config_path_raises():
    with pytest.raises(ValueError, match="config_path"):
        build([f("runs/a.parquet")], {"a": {"model": "m"}})
