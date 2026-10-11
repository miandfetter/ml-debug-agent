# Both roles can be used only by ECS tasks in this account. The SourceAccount and
# SourceArn conditions stop another account from getting ECS to use them on its behalf.
data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    sid     = "AllowEcsTasksInThisAccountOnly"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:ecs:${var.region}:${local.account_id}:*"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

# Used by ECS to start a task: pull the image from ECR and write logs.
resource "aws_iam_role" "execution" {
  name               = "${var.name}-ecs-execution-role"
  description        = "ECS pulls the ml-debug-agent image and writes logs"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Used by the evaluation code while it runs. Least privilege: read the benchmark, write
# results, call one Bedrock model. It can't delete or overwrite anything else.
resource "aws_iam_role" "task" {
  name               = "${var.name}-ecs-task-role"
  description        = "ml-debug-agent code: read benchmark from S3, write results, call Bedrock"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "task_permissions" {
  statement {
    sid     = "ReadBenchmarkArtifactsAndState"
    actions = ["s3:GetObject"]
    resources = [
      "${local.bucket_arn}/mlflow/*",
      "${local.bucket_arn}/state/*",
    ]
  }

  # MLflow checks whether an artifact path is a file or a folder before downloading it.
  statement {
    sid       = "ListOnlyTheArtifactArea"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["mlflow/*"]
    }
  }

  statement {
    sid       = "WriteResults"
    actions   = ["s3:PutObject"]
    resources = ["${local.bucket_arn}/results/*"]
  }

  statement {
    sid     = "InvokeTheConfiguredBedrockModel"
    actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = [
      "arn:aws:bedrock:${var.region}:${local.account_id}:inference-profile/${var.bedrock_model_id}",
      "arn:aws:bedrock:*::foundation-model/${local.bedrock_foundation_model}",
    ]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "${var.name}-task-permissions"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_permissions.json
}
