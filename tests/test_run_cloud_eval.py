"""Tests for the one-command Fargate launcher. Fake clients stand in for AWS.

uv run pytest tests/test_run_cloud_eval.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).parents[1] / "scripts" / "run_cloud_eval.py"
_spec = importlib.util.spec_from_file_location("run_cloud_eval", _path)
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)

TASK_ARN = "arn:aws:ecs:us-east-1:123456789012:task/ml-debug/abc123"


class FakeEcs:
    def __init__(self, failures=None):
        self.calls = []
        self.failures = failures or []

    def run_task(self, **kwargs):
        self.calls.append(kwargs)
        return {"tasks": [{"taskArn": TASK_ARN}], "failures": self.failures}


def test_unknown_arguments_go_to_the_container():
    ours, container = launcher.parse_args(
        ["--upload-db", "--system", "two_agent", "--per-bug", "1"]
    )
    assert ours.upload_db and not ours.no_wait
    assert container == ["--system", "two_agent", "--per-bug", "1"]


def test_container_arguments_override_the_command():
    ecs = FakeEcs()
    arn = launcher.start_task(ecs, "td-arn", ["subnet-a"], "sg-1", ["--system", "baseline"])
    [call] = ecs.calls
    assert arn == TASK_ARN
    assert call["overrides"] == {
        "containerOverrides": [{"name": "eval", "command": ["--system", "baseline"]}]
    }
    network = call["networkConfiguration"]["awsvpcConfiguration"]
    assert network == {
        "subnets": ["subnet-a"],
        "securityGroups": ["sg-1"],
        "assignPublicIp": "ENABLED",
    }


def test_no_arguments_keeps_the_default_command():
    ecs = FakeEcs()
    launcher.start_task(ecs, "td-arn", ["subnet-a"], "sg-1", [])
    assert ecs.calls[0]["overrides"] == {}


def test_run_task_failures_stop_the_script():
    with pytest.raises(SystemExit):
        launcher.start_task(FakeEcs(failures=[{"reason": "RESOURCE"}]), "td", ["s"], "sg", [])


def test_env_value_reads_the_task_definition():
    task_definition = {
        "containerDefinitions": [
            {"name": "eval", "environment": [{"name": "MLFLOW_DB_S3_URI", "value": "s3://b/k"}]}
        ]
    }
    assert launcher.env_value(task_definition, "MLFLOW_DB_S3_URI") == "s3://b/k"
    assert launcher.env_value(task_definition, "MISSING") is None


def test_log_stream_and_uploaded_results():
    assert launcher.log_stream(TASK_ARN) == "eval/eval/abc123"
    lines = ["Accuracy: 100%", "Uploaded s3://b/results/t/benchmark-v2/baseline.csv"]
    assert launcher.uploaded_results(lines) == ["s3://b/results/t/benchmark-v2/baseline.csv"]
