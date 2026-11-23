# Least-privilege roles for the two functions, the API Gateway logging role, and
# GitHub OIDC roles for CI (no long-lived AWS keys in GitHub).

locals {
  state_bucket_arn = "arn:aws:s3:::debbie-tfstate-${var.account_id}"
  role_prefix_arn  = "arn:aws:iam::${var.account_id}:role/debbie-*"
  oidc_host        = "token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# ------------------------------------------------------------------ debbie-api

resource "aws_iam_role" "api" {
  name               = "debbie-api"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "api" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/lambda/debbie-api:*"]
  }
  statement {
    sid       = "Tracing"
    actions   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
    resources = ["*"]
  }
  statement {
    sid       = "ReadIndex"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.kb.arn}/index/*"]
  }
  statement {
    sid       = "OpenAIKey"
    actions   = ["ssm:GetParameter"]
    resources = [local.openai_key_param_arn]
  }
  statement {
    sid       = "RateLimitCounters"
    actions   = ["dynamodb:UpdateItem"]
    resources = [aws_dynamodb_table.ratelimit.arn]
  }
  statement {
    sid       = "Feedback"
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.feedback.arn]
  }
}

resource "aws_iam_role_policy" "api" {
  name   = "debbie-api"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}

# -------------------------------------------------------------- debbie-indexer

resource "aws_iam_role" "indexer" {
  name               = "debbie-indexer"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

data "aws_iam_policy_document" "indexer" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/lambda/debbie-indexer:*"]
  }
  statement {
    sid       = "Tracing"
    actions   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
    resources = ["*"]
  }
  statement {
    sid       = "ListDocsAndIndex"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.kb.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["docs/*", "index/*"]
    }
  }
  statement {
    sid       = "ReadDocs"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.kb.arn}/docs/*"]
  }
  statement {
    sid       = "WriteIndex"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.kb.arn}/index/*"]
  }
  statement {
    sid       = "OpenAIKey"
    actions   = ["ssm:GetParameter"]
    resources = [local.openai_key_param_arn]
  }
}

resource "aws_iam_role_policy" "indexer" {
  name   = "debbie-indexer"
  role   = aws_iam_role.indexer.id
  policy = data.aws_iam_policy_document.indexer.json
}

# ------------------------------------------------- API Gateway -> CloudWatch Logs
# Regional, account-wide setting (one role per account+region); destroy leaves it set.

resource "aws_iam_role" "apigw_logs" {
  name = "debbie-apigw-cloudwatch"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "apigateway.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "apigw_logs" {
  role       = aws_iam_role.apigw_logs.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs"
}

# ------------------------------------------------------------- GitHub OIDC (CI)

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://${local.oidc_host}"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_policy_document" "github_plan_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "${local.oidc_host}:sub"
      values = ["repo:${var.github_repo}:pull_request",
      "repo:${var.github_repo}:ref:refs/heads/main"]
    }
  }
}

data "aws_iam_policy_document" "github_deploy_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Only jobs running in the protected `production` environment (manual approval).
    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = ["repo:${var.github_repo}:environment:production"]
    }
  }
}

# Read-only role for `terraform plan` on pull requests.
resource "aws_iam_role" "github_plan" {
  name                 = "debbie-plan-github"
  assume_role_policy   = data.aws_iam_policy_document.github_plan_trust.json
  max_session_duration = 3600
}

resource "aws_iam_role_policy_attachment" "github_plan_readonly" {
  role       = aws_iam_role.github_plan.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

data "aws_iam_policy_document" "github_plan_guard" {
  # ReadOnlyAccess would otherwise let a PR workflow read the OpenAI key or the
  # internal DEVCON documents; plan needs neither.
  statement {
    sid       = "NoSecrets"
    effect    = "Deny"
    actions   = ["ssm:GetParameter*", "kms:Decrypt"]
    resources = [local.openai_key_param_arn, "arn:aws:kms:${var.region}:${var.account_id}:key/*"]
  }
  statement {
    sid       = "NoKnowledgeBaseContent"
    effect    = "Deny"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.kb.arn}/*"]
  }
}

resource "aws_iam_role_policy" "github_plan_guard" {
  name   = "deny-secrets-and-docs"
  role   = aws_iam_role.github_plan.id
  policy = data.aws_iam_policy_document.github_plan_guard.json
}

# Apply role, assumable only from the `production` environment (required reviewer).
# It can manage debbie-* resources, including debbie-* IAM roles — inherent to
# running Terraform for this stack; the approval gate + OIDC sub pin protect it.
resource "aws_iam_role" "github_deploy" {
  name                 = "debbie-deploy-github"
  assume_role_policy   = data.aws_iam_policy_document.github_deploy_trust.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "github_deploy" {
  statement {
    sid       = "TerraformState"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [local.state_bucket_arn, "${local.state_bucket_arn}/*"]
  }
  statement {
    sid       = "KnowledgeBaseBucket"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.kb.arn, "${aws_s3_bucket.kb.arn}/*"]
  }
  statement {
    sid       = "DynamoDB"
    actions   = ["dynamodb:*"]
    resources = ["arn:aws:dynamodb:${var.region}:${var.account_id}:table/debbie-*"]
  }
  statement {
    sid     = "Lambda"
    actions = ["lambda:*"]
    resources = ["arn:aws:lambda:${var.region}:${var.account_id}:function:debbie-*",
    "arn:aws:lambda:${var.region}:${var.account_id}:function:debbie-*:*"]
  }
  statement {
    sid       = "LambdaRead"
    actions   = ["lambda:GetLayerVersion", "lambda:ListFunctions", "lambda:GetAccountSettings"]
    resources = ["*"]
  }
  statement {
    sid       = "ApiGateway"
    actions   = ["apigateway:GET", "apigateway:POST", "apigateway:PUT", "apigateway:PATCH", "apigateway:DELETE"]
    resources = ["arn:aws:apigateway:${var.region}::/*"]
  }
  statement {
    sid     = "Logs"
    actions = ["logs:*"]
    resources = ["arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/lambda/debbie-*",
      "arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/lambda/debbie-*:*",
      "arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/apigateway/debbie-*",
    "arn:aws:logs:${var.region}:${var.account_id}:log-group:/aws/apigateway/debbie-*:*"]
  }
  statement {
    sid       = "Describe"
    actions   = ["logs:DescribeLogGroups", "cloudwatch:DescribeAlarms", "ssm:DescribeParameters", "s3:ListAllMyBuckets"]
    resources = ["*"]
  }
  statement {
    sid       = "Alarms"
    actions   = ["cloudwatch:PutMetricAlarm", "cloudwatch:DeleteAlarms", "cloudwatch:ListTagsForResource", "cloudwatch:TagResource", "cloudwatch:UntagResource"]
    resources = ["arn:aws:cloudwatch:${var.region}:${var.account_id}:alarm:debbie-*"]
  }
  statement {
    sid       = "Sns"
    actions   = ["sns:*"]
    resources = ["arn:aws:sns:${var.region}:${var.account_id}:debbie-*"]
  }
  statement {
    sid       = "Budget"
    actions   = ["budgets:ViewBudget", "budgets:ModifyBudget", "budgets:ListTagsForResource", "budgets:TagResource", "budgets:UntagResource"]
    resources = ["arn:aws:budgets::${var.account_id}:budget/debbie-*"]
  }
  statement {
    sid       = "AlertEmailParam"
    actions   = ["ssm:GetParameter", "ssm:ListTagsForResource"]
    resources = [data.aws_ssm_parameter.alert_email.arn]
  }
  statement {
    sid = "IamRoles"
    actions = ["iam:CreateRole", "iam:DeleteRole", "iam:GetRole", "iam:UpdateRole", "iam:UpdateAssumeRolePolicy",
      "iam:TagRole", "iam:UntagRole", "iam:ListRolePolicies", "iam:ListAttachedRolePolicies", "iam:GetRolePolicy",
      "iam:PutRolePolicy", "iam:DeleteRolePolicy", "iam:AttachRolePolicy", "iam:DetachRolePolicy",
    "iam:ListInstanceProfilesForRole"]
    resources = [local.role_prefix_arn]
  }
  statement {
    sid       = "PassOnlyDebbieRolesToLambdaAndApiGateway"
    actions   = ["iam:PassRole"]
    resources = [local.role_prefix_arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["lambda.amazonaws.com", "apigateway.amazonaws.com"]
    }
  }
  statement {
    sid = "GithubOidcProvider"
    actions = ["iam:GetOpenIDConnectProvider", "iam:TagOpenIDConnectProvider", "iam:UntagOpenIDConnectProvider",
    "iam:UpdateOpenIDConnectProviderThumbprint", "iam:AddClientIDToOpenIDConnectProvider", "iam:RemoveClientIDFromOpenIDConnectProvider"]
    resources = [aws_iam_openid_connect_provider.github.arn]
  }
  statement {
    sid       = "ReadManagedPolicies"
    actions   = ["iam:GetPolicy", "iam:GetPolicyVersion", "iam:ListPolicyVersions"]
    resources = ["arn:aws:iam::aws:policy/*"]
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "debbie-deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy.json
}
