"""Guards on the GitHub Actions workflows that deploy to and run in the GCP project."""

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
NAMES = ("ci.yml", "deploy.yml", "run.yml")
SHA_PIN = re.compile(r"^[\w.-]+/[\w.-]+(/[\w.-]+)*@[0-9a-f]{40}$")


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def steps(workflow: dict):
    for job in workflow["jobs"].values():
        yield from job.get("steps", [])


@pytest.mark.parametrize("name", NAMES)
def test_every_action_is_pinned_to_a_commit_sha(name):
    uses = [s["uses"] for s in steps(load(name)) if "uses" in s]
    assert all(SHA_PIN.match(u) for u in uses), uses


@pytest.mark.parametrize("name", NAMES)
def test_inputs_never_interpolated_into_run_scripts(name):
    # A workflow_dispatch input pasted into `run:` is shell injection; pass it through env.
    for step in steps(load(name)):
        script = step.get("run", "")
        assert "inputs." not in script and "github.event" not in script, step.get("name")


def test_ci_cannot_request_an_oidc_token():
    workflow = load("ci.yml")
    assert workflow["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in workflow["jobs"].values())


@pytest.mark.parametrize("name", ["deploy.yml", "run.yml"])
def test_cloud_workflows_authenticate_with_wif_only(name):
    workflow = load(name)
    assert workflow["permissions"]["id-token"] == "write"
    auth = [s for s in steps(workflow)
            if s.get("uses", "").startswith("google-github-actions/auth@")]
    assert len(auth) == 1
    assert set(auth[0]["with"]) == {"workload_identity_provider", "service_account", "project_id"}


def test_deploy_runs_ci_first():
    jobs = load("deploy.yml")["jobs"]
    assert jobs["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert jobs["deploy"]["needs"] == "ci"


SQL_ARG = re.compile(r"""\"\$\(sed '/\^--/d' (bq/[\w.]+\.sql)\)\"""")


def test_bq_never_sees_a_leading_sql_comment():
    # bq parses flags with absl, which reads an argument starting with "--" as a flag even
    # after positional arguments, so the comment header of each .sql file must be stripped.
    text = (WORKFLOWS / "run.yml").read_text(encoding="utf-8")
    files = SQL_ARG.findall(text)
    assert len(files) == text.count("bq query") == 2
    root = WORKFLOWS.parents[1]
    for name in files:
        stripped = "\n".join(line for line in (root / name).read_text().splitlines()
                             if not line.startswith("--"))
        assert stripped.lstrip().startswith("SELECT"), name
