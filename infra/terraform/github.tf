# Lets GitHub Actions push images to ECR without any AWS keys stored in GitHub.
#
# How it works: GitHub gives each workflow run a signed token describing it ("repository
# X, branch main"). AWS checks the signature against GitHub's identity provider below and,
# if the role's conditions match, hands back short-lived credentials for that role only.

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

# Who may use the role: only workflow runs on the main branch of this one repository,
# identified by its permanent numeric IDs (see github_oidc_subject_prefix). A fork, another
# branch, a pull request, or a new repository reusing the name gets a different "sub" value
# and is refused.
data "aws_iam_policy_document" "github_actions_trust" {
  statement {
    sid     = "GitHubActionsMainBranchOnly"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["${var.github_oidc_subject_prefix}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name                 = "${var.name}-github-actions-ecr-push"
  description          = "GitHub Actions (main branch) pushes the ml-debug-agent image to ECR"
  assume_role_policy   = data.aws_iam_policy_document.github_actions_trust.json
  max_session_duration = 3600
}

# What the role may do: push images to this one repository, nothing else.
data "aws_iam_policy_document" "github_actions_permissions" {
  # Logging Docker in to ECR. This action doesn't support narrowing to a repository.
  statement {
    sid       = "GetEcrLoginToken"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "PushToTheProjectRepository"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.app.arn]
  }
}

resource "aws_iam_role_policy" "github_actions" {
  name   = "${var.name}-ecr-push"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.github_actions_permissions.json
}
