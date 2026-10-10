"""Tests for the container entrypoint. A fake S3 client stands in for AWS.

    uv run pytest tests/test_cloud_run.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ml_debug_agent.eval import cloud_run


class FakeS3:
    """Records uploads and serves downloads from a local file."""

    def __init__(self, source: Path | None = None):
        self.source = source
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []

    def download_file(self, bucket, key, filename):
        self.downloads.append((bucket, key))
        shutil.copy(self.source, filename)

    def upload_file(self, filename, bucket, key):
        self.uploads.append((bucket, key))


def test_split_s3_uri():
    assert cloud_run.split_s3_uri("s3://bucket/state/mlflow.db") == ("bucket", "state/mlflow.db")
    assert cloud_run.split_s3_uri("s3://bucket/results/") == ("bucket", "results")
    assert cloud_run.split_s3_uri("s3://bucket") == ("bucket", "")
    with pytest.raises(ValueError):
        cloud_run.split_s3_uri("/local/path")


def test_download_db_copies_to_a_writable_place(tmp_path):
    source = tmp_path / "remote.db"
    source.write_text("pretend database")
    s3 = FakeS3(source)
    path = cloud_run.download_db("s3://bucket/state/mlflow.db", s3=s3)
    assert s3.downloads == [("bucket", "state/mlflow.db")]
    assert path.read_text() == "pretend database"


def test_upload_results_keeps_subfolders_under_a_timestamp(tmp_path):
    (tmp_path / "benchmark-v2").mkdir()
    (tmp_path / "benchmark-v2" / "baseline.csv").write_text("x")
    (tmp_path / "notes.txt").write_text("not a result")
    s3 = FakeS3()
    uploaded = cloud_run.upload_results(tmp_path, "s3://bucket/results", s3=s3)

    [(bucket, key)] = s3.uploads  # only the CSV
    prefix, stamp, *rest = key.split("/")
    assert (bucket, prefix, "/".join(rest)) == ("bucket", "results", "benchmark-v2/baseline.csv")
    assert stamp.endswith("Z")
    assert uploaded == [f"s3://bucket/{key}"]


def test_without_s3_settings_it_is_just_run_benchmark(runs, tmp_path, monkeypatch):
    import mlflow

    monkeypatch.delenv("MLFLOW_DB_S3_URI", raising=False)
    monkeypatch.delenv("RESULTS_S3_URI", raising=False)
    experiment = mlflow.get_experiment(mlflow.get_run(next(iter(runs.values()))).info.experiment_id)
    cloud_run.main(
        ["--system", "baseline", "--experiment", experiment.name, "--out-dir", str(tmp_path)]
    )
    assert (tmp_path / experiment.name / "baseline.csv").exists()
