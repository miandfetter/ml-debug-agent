# One private bucket with three areas:
#   mlflow/   run artifacts, written by MLflow when a benchmark is generated
#   state/    mlflow.db, the snapshot the cloud evaluation downloads
#   results/  evaluation outputs, written by the Fargate task
resource "aws_s3_bucket" "artifacts" {
  bucket = local.bucket_name

  # Holds the benchmark. Terraform refuses to delete it, even with `terraform destroy`.
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Objects always belong to this account; access is controlled only by IAM, not ACLs.
resource "aws_s3_bucket_ownership_controls" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}
