resource "aws_cloudwatch_log_group" "eval" {
  name              = "/ecs/ml-debug-agent"
  retention_in_days = var.log_retention_days
}

# With Fargate a cluster holds no servers and costs nothing; it groups the tasks.
resource "aws_ecs_cluster" "this" {
  name = var.name

  setting {
    name  = "containerInsights"
    value = "disabled" # detailed metrics are billed; the logs are enough here
  }
}

# The blueprint for one evaluation run: the cloud version of `docker run`.
# Any change here registers a new revision; tasks already running keep their revision.
resource "aws_ecs_task_definition" "eval" {
  family                   = "ml-debug-agent-eval"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    cpu_architecture        = "ARM64" # matches images built on an Apple-silicon Mac
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([{
    name      = "eval"
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    essential = true
    command   = ["--system", "baseline"]
    environment = [
      { name = "AWS_REGION", value = var.region },
      { name = "BENCHMARK_EXPERIMENT", value = var.benchmark_experiment },
      { name = "MLFLOW_DB_S3_URI", value = "s3://${aws_s3_bucket.artifacts.bucket}/state/mlflow.db" },
      { name = "RESULTS_S3_URI", value = "s3://${aws_s3_bucket.artifacts.bucket}/results" },
      { name = "LLM_PROVIDER", value = "bedrock" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.eval.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "eval"
      }
    }
  }])
}
