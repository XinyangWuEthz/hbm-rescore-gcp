# One image for both Cloud Run jobs. `rescore` runs the default entrypoint; `collect`
# overrides the command with `python -m rescore.collect`.
#
#   docker build --platform linux/amd64 \
#     --build-arg RESEARCH_COMMIT=7e177532f2240d6a10cd76ed300e8c379123297f -t hbm-rescore:dev .
#
# Cloud Run only runs linux/amd64 images, so build for that platform even on Apple Silicon.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30

# --- research code at one pinned commit ---------------------------------------------------
FROM ${PYTHON_IMAGE} AS research
ARG RESEARCH_REPO=https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at
ARG RESEARCH_COMMIT
RUN test -n "$RESEARCH_COMMIT" || { echo "RESEARCH_COMMIT build arg is required" >&2; exit 1; }
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# A shallow fetch of exactly one commit: the image never contains a moving branch.
RUN git init -q repo \
 && git -C repo fetch -q --depth 1 "$RESEARCH_REPO" "$RESEARCH_COMMIT" \
 && git -C repo checkout -q FETCH_HEAD \
 && test "$(git -C repo rev-parse HEAD)" = "$RESEARCH_COMMIT"
# Only what scoring reads. This is the set the 2026-11-05 local re-score ran with.
RUN mkdir /src/research \
 && cd repo && cp -R src scripts configs data eval reports pyproject.toml /src/research/

# --- runtime ------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}
ARG RESEARCH_COMMIT
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    RESEARCH_DIR=/app/research \
    MANIFEST=/app/manifest/runs.json
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=research /src/research /app/research
COPY manifest /app/manifest
COPY rescore /app/rescore
# The manifest names the commit its runs are scored with; refuse an image that disagrees.
RUN python -c "import json, sys; m = json.load(open('/app/manifest/runs.json')); \
sys.exit(0 if m['research_commit'] == sys.argv[1] else \
'manifest research_commit ' + m['research_commit'] + ' != build arg ' + sys.argv[1])" \
    "$RESEARCH_COMMIT"
ENTRYPOINT ["python", "-m", "rescore.task"]
