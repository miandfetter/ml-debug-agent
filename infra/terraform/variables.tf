variable "region" {
  description = "AWS region for everything in this project."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Prefix for resource names."
  type        = string
  default     = "ml-debug"
}

variable "image_tag" {
  description = "Tag of the evaluation image in ECR that the task definition runs."
  type        = string
  default     = "latest"
}

variable "benchmark_experiment" {
  description = "MLflow experiment the evaluation task runs on by default."
  type        = string
  default     = "benchmark-v2"
}

variable "bedrock_model_id" {
  description = "Bedrock inference profile the agents call (the task role may call only this)."
  type        = string
  default     = "us.amazon.nova-2-lite-v1:0"
}

variable "log_retention_days" {
  description = "How long CloudWatch keeps the evaluation logs."
  type        = number
  default     = 14
}

variable "github_repository" {
  description = "GitHub repository (owner/name) whose main branch may push images to ECR."
  type        = string
  default     = "miandfetter/ml-debug-agent"
}
