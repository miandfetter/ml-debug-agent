output "bucket" {
  description = "S3 bucket for MLflow artifacts, the mlflow.db snapshot and results."
  value       = aws_s3_bucket.artifacts.bucket
}

output "mlflow_artifact_root" {
  description = "Value for MLFLOW_ARTIFACT_ROOT in .env."
  value       = "s3://${aws_s3_bucket.artifacts.bucket}/mlflow"
}

output "ecr_repository_url" {
  description = "Where to tag and push the evaluation image."
  value       = aws_ecr_repository.app.repository_url
}

output "cluster" {
  value = aws_ecs_cluster.this.name
}

output "task_definition" {
  description = "Family and current revision."
  value       = "${aws_ecs_task_definition.eval.family}:${aws_ecs_task_definition.eval.revision}"
}

output "security_group_id" {
  value = aws_security_group.eval_task.id
}

output "public_subnet_ids" {
  value = data.aws_subnets.public.ids
}
