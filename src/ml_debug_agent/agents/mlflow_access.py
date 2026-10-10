"""The only module that reads MLflow. Every agent tool goes through `get_run_view`.

Benchmark runs contain the answer in several places: the run name (e.g. "overfit-s2"),
the run description, and the "eval.true_bug" tag. None of those are ever returned by
`get_run_view`. Only `get_true_bug` reads the answer, and only the eval harness may call it.

Check what the agent sees for a run:
    uv run python -m ml_debug_agent.agents.mlflow_access            # first benchmark run
    uv run python -m ml_debug_agent.agents.mlflow_access <run_id>
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import fields
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from ml_debug_agent.training.train import HIDDEN_FIELDS, Bug, TrainConfig

DEFAULT_BENCHMARK_EXPERIMENT = "benchmark-v1"
SPLIT_ARTIFACT = "data/split_indices.json"

# Allowlist, not a blocklist: anything not named here is never exposed.
ALLOWED_PARAMS = ({f.name for f in fields(TrainConfig)} - HIDDEN_FIELDS) | {"n_train_examples"}
EPOCH_METRICS = (
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "grad_norm_mean",
    "grad_norm_max",
    "lr",
)
FINAL_METRICS = ("test_loss", "test_acc", "diverged_at_epoch")


class RunView(BaseModel):
    """Everything the agent is allowed to know about one training run."""

    run_id: str
    params: dict[str, Any]
    history: dict[str, list[float]]  # metric name -> one value per epoch, in order
    final_metrics: dict[str, float]
    training_status: str  # "completed" or "diverged" (what a job scheduler would report)


def _parse(value: str) -> Any:
    """MLflow stores params as strings; turn them back into Python values."""
    if value == "None":
        return None
    if value in ("True", "False"):
        return value == "True"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def benchmark_experiment() -> str:
    """The benchmark to use when none is named: BENCHMARK_EXPERIMENT from .env, or v1."""
    return os.environ.get("BENCHMARK_EXPERIMENT", DEFAULT_BENCHMARK_EXPERIMENT)


def list_benchmark_runs(experiment: str | None = None) -> list[str]:
    """Run IDs of finished runs in an experiment. Returns IDs only, never names."""
    experiment = experiment or benchmark_experiment()
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(f"No MLflow experiment named {experiment!r}. Run the generator first.")
    runs = mlflow.search_runs([exp.experiment_id], output_format="list")
    finished = [r for r in runs if r.info.status == "FINISHED"]
    # Sorted by name only to get a stable order; the names themselves are not returned.
    return [r.info.run_id for r in sorted(finished, key=lambda r: r.info.run_name or "")]


def get_run_view(run_id: str) -> RunView:
    """The agent's view of a run: allowed params, per-epoch metrics, final metrics."""
    client = MlflowClient()
    run = client.get_run(run_id)

    params = {k: _parse(v) for k, v in run.data.params.items() if k in ALLOWED_PARAMS}

    history: dict[str, list[float]] = {}
    for name in EPOCH_METRICS:
        points = sorted(client.get_metric_history(run_id, name), key=lambda m: m.step)
        if points:
            history[name] = [m.value for m in points]

    final_metrics = {k: v for k, v in run.data.metrics.items() if k in FINAL_METRICS}

    return RunView(
        run_id=run_id,
        params=params,
        history=history,
        final_metrics=final_metrics,
        training_status=run.data.tags.get("status", "unknown"),
    )


def get_split_indices(run_id: str) -> dict[str, list[int]]:
    """The train/val split recorded for a run, used by the overlap check."""
    return mlflow.artifacts.load_dict(f"runs:/{run_id}/{SPLIT_ARTIFACT}")


def get_true_bug(run_id: str) -> Bug:
    """EVAL HARNESS ONLY. Returns the ground-truth bug. Never call this from agent code."""
    return Bug(MlflowClient().get_run(run_id).data.tags["eval.true_bug"])


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else list_benchmark_runs()[0]
    print(json.dumps(get_run_view(target).model_dump(), indent=2))