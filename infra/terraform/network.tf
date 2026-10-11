# Tasks run in the default VPC's public subnets with a public IP. That gives them a route
# to ECR, S3 and Bedrock without a NAT gateway, which is billed hourly even when idle.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}

# A batch job needs no incoming connections, so nothing can connect in.
resource "aws_security_group" "eval_task" {
  name        = "${var.name}-eval-task"
  description = "ml-debug eval tasks: no inbound, all outbound"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
