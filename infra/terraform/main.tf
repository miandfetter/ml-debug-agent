data "aws_caller_identity" "current" {}

locals {
  account_id  = data.aws_caller_identity.current.account_id
  bucket_name = "${var.name}-artifacts-${local.account_id}"
  # Built from the name rather than the bucket resource, so the IAM policy is known at
  # plan time and `terraform plan` can show exactly what would change in it.
  bucket_arn = "arn:aws:s3:::${local.bucket_name}"
  # The us.amazon... inference profile routes to the underlying amazon... model.
  bedrock_foundation_model = trimprefix(var.bedrock_model_id, "us.")
}
