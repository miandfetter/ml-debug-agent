"""Run the evaluation on ECS Fargate with one command, and wait for the result.

Arguments this script doesn't recognize go straight to the container (run_benchmark):

    uv run python scripts/run_cloud_eval.py
    uv run python scripts/run_cloud_eval.py --system baseline --per-bug 1
    uv run python scripts/run_cloud_eval.py --upload-db --system two_agent --bugs label_shuffle
    uv run python scripts/run_cloud_eval.py --no-wait --system baseline

It looks everything up by name (cluster, newest task definition, the default VPC's public
subnets, the security group), starts the task, polls until it stops, then prints the
container's logs and where its results were uploaded.
"""

from __future__ import annotations

import argparse
import sys
import time

import boto3
from dotenv import load_dotenv

CLUSTER = "ml-debug"
TASK_FAMILY = "ml-debug-agent-eval"
SECURITY_GROUP = "ml-debug-eval-task"
CONTAINER = "eval"
LOG_GROUP = "/ecs/ml-debug-agent"
LOG_STREAM_PREFIX = "eval"
POLL_SECONDS = 10


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split argv into this script's options and the container's arguments."""
    p = argparse.ArgumentParser(
        description="Run the evaluation on Fargate. Other arguments go to run_benchmark.",
        allow_abbrev=False,  # so e.g. --no doesn't swallow a container argument
    )
    p.add_argument("--upload-db", action="store_true", help="Upload local mlflow.db first.")
    p.add_argument("--no-wait", action="store_true", help="Start the task and exit.")
    p.add_argument("--region", default="us-east-1")
    ours, container_args = p.parse_known_args(argv)
    return ours, container_args


def env_value(task_definition: dict, name: str) -> str | None:
    """An environment variable's value from the task definition's eval container."""
    container = next(c for c in task_definition["containerDefinitions"] if c["name"] == CONTAINER)
    return next((e["value"] for e in container.get("environment", []) if e["name"] == name), None)


def public_subnets(ec2) -> list[str]:
    vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]
    subnets = ec2.describe_subnets(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc["VpcId"]]},
            {"Name": "map-public-ip-on-launch", "Values": ["true"]},
        ]
    )["Subnets"]
    return sorted(s["SubnetId"] for s in subnets)


def security_group(ec2) -> str:
    groups = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SECURITY_GROUP]}]
    )["SecurityGroups"]
    if not groups:
        sys.exit(f"Security group {SECURITY_GROUP!r} not found. Create it first (README).")
    return groups[0]["GroupId"]


def start_task(
    ecs, task_definition_arn: str, subnets: list[str], sg: str, container_args: list[str]
) -> str:
    overrides = {}
    if container_args:  # otherwise the image's default command (--system baseline) is used
        overrides = {"containerOverrides": [{"name": CONTAINER, "command": container_args}]}
    response = ecs.run_task(
        cluster=CLUSTER,
        taskDefinition=task_definition_arn,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": subnets,
                "securityGroups": [sg],
                "assignPublicIp": "ENABLED",  # public route to ECR/S3/Bedrock, no NAT gateway
            }
        },
        overrides=overrides,
    )
    if response.get("failures"):
        sys.exit(f"run_task failed: {response['failures']}")
    return response["tasks"][0]["taskArn"]


def wait_until_stopped(ecs, task_arn: str) -> dict:
    """Poll the task, printing each status change, until it stops. Returns the task."""
    last = None
    while True:
        task = ecs.describe_tasks(cluster=CLUSTER, tasks=[task_arn])["tasks"][0]
        if task["lastStatus"] != last:
            last = task["lastStatus"]
            print(f"  {time.strftime('%H:%M:%S')}  {last}")
        if last == "STOPPED":
            return task
        time.sleep(POLL_SECONDS)


def log_stream(task_arn: str) -> str:
    """CloudWatch stream name: <prefix>/<container>/<task id>."""
    return f"{LOG_STREAM_PREFIX}/{CONTAINER}/{task_arn.rsplit('/', 1)[-1]}"


def read_logs(logs, task_arn: str) -> list[str]:
    lines, token = [], None
    while True:
        kwargs = {"nextToken": token} if token else {}
        page = logs.get_log_events(
            logGroupName=LOG_GROUP, logStreamName=log_stream(task_arn), startFromHead=True, **kwargs
        )
        lines += [e["message"] for e in page["events"]]
        if page["nextForwardToken"] == token:  # same token back means no more pages
            return lines
        token = page["nextForwardToken"]


def uploaded_results(log_lines: list[str]) -> list[str]:
    return [
        line.removeprefix("Uploaded ") for line in log_lines if line.startswith("Uploaded s3://")
    ]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # AWS_PROFILE from .env
    ours, container_args = parse_args(sys.argv[1:] if argv is None else argv)
    session = boto3.Session(region_name=ours.region)
    ecs, ec2 = session.client("ecs"), session.client("ec2")

    task_definition = ecs.describe_task_definition(taskDefinition=TASK_FAMILY)["taskDefinition"]
    print(f"Task definition: {TASK_FAMILY}:{task_definition['revision']}")

    if ours.upload_db:
        db_uri = env_value(task_definition, "MLFLOW_DB_S3_URI")
        bucket, _, key = db_uri.removeprefix("s3://").partition("/")
        session.client("s3").upload_file("mlflow.db", bucket, key)
        print(f"Uploaded mlflow.db to {db_uri}")

    task_arn = start_task(
        ecs,
        task_definition["taskDefinitionArn"],
        public_subnets(ec2),
        security_group(ec2),
        container_args,
    )
    print(
        f"Started task {task_arn.rsplit('/', 1)[-1]} with: {container_args or '(default command)'}"
    )
    if ours.no_wait:
        print(f"Not waiting. Follow the logs with: aws logs tail {LOG_GROUP} --follow")
        return 0

    task = wait_until_stopped(ecs, task_arn)
    container = next(c for c in task["containers"] if c["name"] == CONTAINER)
    exit_code = container.get("exitCode")

    if exit_code is None:  # never started, e.g. the image couldn't be pulled
        print(f"\nThe container did not run: {task.get('stoppedReason', 'no reason given')}")
        return 1

    print(f"\n--- Container logs ({LOG_GROUP}) ---")
    lines = read_logs(session.client("logs"), task_arn)
    print("\n".join(lines))

    print(f"\nExit code: {exit_code}")
    for uri in uploaded_results(lines):
        print(f"Results: {uri}")
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
