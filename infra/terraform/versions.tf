terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0" # any 6.x; a new major version may have breaking changes
    }
  }
}

# Credentials come from your AWS CLI profile (AWS_PROFILE=ml-debug plus `aws sso login`),
# never from keys in these files.
provider "aws" {
  region = var.region

  # Added to every resource Terraform manages, so costs can be grouped by project.
  default_tags {
    tags = {
      Project   = "ml-debug-agent"
      ManagedBy = "terraform"
    }
  }
}
