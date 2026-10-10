"""Container entrypoint: fetch the MLflow database from S3, run the benchmark, upload results.

On Fargate there is no laptop to share folders with, so state moves through S3. Two
optional environment variables control it:

    MLFLOW_DB_S3_URI   e.g. s3://my-bucket/state/mlflow.db   downloaded before the eval
    RESULTS_S3_URI     e.g. s3://my-bucket/results           results uploaded after the eval

With neither set, this is exactly run_benchmark, so a locally mounted mlflow.db still works.
Command-line arguments are passed straight through to run_benchmark:

    python -m ml_debug_agent.eval.cloud_run --system baseline --per-bug 1
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import boto3


def split_s3_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/some/key' -> ('bucket', 'some/key')."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key.strip("/")


def download_db(uri: str, s3=None) -> Path:
    """Download the MLflow database to a writable temp folder and return its path."""
    path = Path(tempfile.gettempdir()) / "mlflow.db"
    bucket, key = split_s3_uri(uri)
    (s3 or boto3.client("s3")).download_file(bucket, key, str(path))
    return path


def upload_results(results_dir: Path, uri: str, s3=None) -> list[str]:
    """Upload every CSV under results_dir to <uri>/<UTC timestamp>/..., keeping subfolders."""
    bucket, prefix = split_s3_uri(uri)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    s3 = s3 or boto3.client("s3")
    uploaded = []
    for file in sorted(results_dir.rglob("*.csv")):
        parts = (prefix, stamp, file.relative_to(results_dir).as_posix())
        key = "/".join(part for part in parts if part)
        s3.upload_file(str(file), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    db_uri = os.environ.get("MLFLOW_DB_S3_URI")
    if db_uri:
        path = download_db(db_uri)
        # Set before MLflow is first used, so every lookup reads the downloaded copy.
        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{path}"
        print(f"Downloaded {db_uri} to {path}")

    from ml_debug_agent.eval import run_benchmark

    run_benchmark.main(argv)

    results_uri = os.environ.get("RESULTS_S3_URI")
    if results_uri:
        out_dir = Path("results")
        if "--out-dir" in argv:
            out_dir = Path(argv[argv.index("--out-dir") + 1])
        for uploaded in upload_results(out_dir, results_uri):
            print(f"Uploaded {uploaded}")


if __name__ == "__main__":
    main()
