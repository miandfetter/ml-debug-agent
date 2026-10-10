# Evaluation image: runs a diagnoser over a benchmark. It never trains, so it leaves out
# PyTorch (the "train" dependency group) and the dev tools.
#
#   docker build --provenance=false -t ml-debug-agent .
#   docker run --rm ml-debug-agent --system baseline --per-bug 1
#
# --provenance=false makes a single plain image instead of an index plus a build
# attestation, which keeps ECR's cleanup rule and vulnerability scanning simple.
#
# Two stages: "builder" installs everything with uv, and the final image copies in only
# the finished virtual environment, so uv and the build steps don't ship.

# ---- Stage 1: build the virtual environment ----------------------------------------------
FROM python:3.12-slim AS builder

# uv, copied from its official image at the same version used locally.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: they change rarely, so a code-only change
# rebuilds in seconds instead of reinstalling everything. The cache mount keeps uv's
# download cache on the build machine instead of baking it into the image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group train --no-install-project

# Then the project itself, installed into the environment (not as a link to src/),
# so the final stage needs only .venv.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group train --no-editable

# ---- Stage 2: the image that actually runs ------------------------------------------------
FROM python:3.12-slim

# Run as an unprivileged user rather than root. Only the results folder needs to be
# writable; the environment stays owned by root and readable by everyone.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
RUN mkdir results && chown app results

COPY --from=builder /app/.venv /app/.venv

USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

# The wrapper runs run_benchmark, plus the optional S3 download/upload around it (see
# cloud_run.py). With no S3 settings it behaves exactly like run_benchmark.
ENTRYPOINT ["python", "-m", "ml_debug_agent.eval.cloud_run"]
CMD ["--system", "baseline"]
