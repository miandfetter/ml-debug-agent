"""Tests for where new MLflow experiments store their artifacts.

A local folder stands in for the S3 bucket, so these need no AWS access.

    uv run pytest tests/test_train.py
"""

from __future__ import annotations

import mlflow

from ml_debug_agent.training.train import use_experiment


def test_new_experiment_uses_the_artifact_root(runs, tmp_path, monkeypatch):
    root = tmp_path / "artifact-root"
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", f"{root}/")  # trailing slash is tolerated
    use_experiment("test-new-experiment")
    exp = mlflow.get_experiment_by_name("test-new-experiment")
    assert exp.artifact_location == f"{root}/test-new-experiment"


def test_existing_experiment_keeps_its_location(runs, tmp_path, monkeypatch):
    use_experiment("test-existing-experiment")  # created without a root
    before = mlflow.get_experiment_by_name("test-existing-experiment").artifact_location
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifact-root"))
    use_experiment("test-existing-experiment")
    after = mlflow.get_experiment_by_name("test-existing-experiment").artifact_location
    assert after == before
