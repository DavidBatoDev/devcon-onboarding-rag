# Build the packages first: python scripts/build_lambda.py all  (-> ../build/api, ../build/indexer)

data "archive_file" "api" {
  type             = "zip"
  source_dir       = "${path.module}/../build/api"
  output_path      = "${path.module}/../build/api.zip"
  output_file_mode = "0755" # run.sh must be executable (Windows builds lose the bit)
}

data "archive_file" "indexer" {
  type             = "zip"
  source_dir       = "${path.module}/../build/indexer"
  output_path      = "${path.module}/../build/indexer.zip"
  output_file_mode = "0644"
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/debbie-api"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "indexer" {
  name              = "/aws/lambda/debbie-indexer"
  retention_in_days = var.log_retention_days
}

# ------------------------------------------------------------------ debbie-api
# FastAPI app run by Lambda Web Adapter in response_stream mode. Every apply that
# changes code/config publishes a new version; API Gateway invokes the `live`
# alias, which the release step (scripts/release.py) moves only after the new
# version passes a smoke test — a broken build never takes traffic.

resource "aws_lambda_function" "api" {
  function_name    = "debbie-api"
  role             = aws_iam_role.api.arn
  runtime          = "python3.13"
  architectures    = ["arm64"]
  handler          = "run.sh"
  filename         = data.archive_file.api.output_path
  source_code_hash = data.archive_file.api.output_base64sha256
  memory_size      = var.api_memory_mb
  timeout          = var.api_timeout_s
  layers           = [var.lwa_layer_arn]
  publish          = true

  reserved_concurrent_executions = var.api_reserved_concurrency == null ? -1 : var.api_reserved_concurrency

  tracing_config {
    mode = "Active"
  }

  logging_config {
    log_format = "Text" # the app already writes one JSON object per line
    log_group  = aws_cloudwatch_log_group.api.name
  }

  environment {
    variables = {
      AWS_LAMBDA_EXEC_WRAPPER      = "/opt/bootstrap"
      AWS_LWA_INVOKE_MODE          = "response_stream"
      AWS_LWA_READINESS_CHECK_PATH = "/api/v1/status"
      PORT                         = "8000"
      INDEX_BUCKET                 = aws_s3_bucket.kb.bucket
      INDEX_PREFIX                 = "index/"
      RATELIMIT_TABLE              = aws_dynamodb_table.ratelimit.name
      FEEDBACK_TABLE               = aws_dynamodb_table.feedback.name
      OPENAI_API_KEY_PARAM         = local.openai_key_param
      OPENAI_LLM_MODEL             = var.answer_model
      OPENAI_REASONING_EFFORT      = var.reasoning_effort
      OPENAI_API_STYLE             = var.openai_api_style
      QUERY_TIMEOUT_SECONDS        = tostring(var.api_timeout_s - 10)
      MAX_CONCURRENT_QUERIES       = "2"
    }
  }

  depends_on = [aws_iam_role_policy.api]
}

resource "aws_lambda_alias" "api_live" {
  name             = "live"
  function_name    = aws_lambda_function.api.function_name
  function_version = aws_lambda_function.api.version

  lifecycle {
    # Moved by scripts/release.py after the smoke test, not by terraform apply.
    ignore_changes = [function_version]
  }
}

# -------------------------------------------------------------- debbie-indexer

resource "aws_lambda_function" "indexer" {
  function_name    = "debbie-indexer"
  role             = aws_iam_role.indexer.arn
  runtime          = "python3.13"
  architectures    = ["arm64"]
  handler          = "indexer.handler.handler"
  filename         = data.archive_file.indexer.output_path
  source_code_hash = data.archive_file.indexer.output_base64sha256
  memory_size      = var.indexer_memory_mb
  timeout          = 300

  reserved_concurrent_executions = var.indexer_reserved_concurrency == null ? -1 : var.indexer_reserved_concurrency

  tracing_config {
    mode = "Active"
  }

  logging_config {
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.indexer.name
  }

  environment {
    variables = {
      INDEX_BUCKET         = aws_s3_bucket.kb.bucket
      DOCS_PREFIX          = "docs/"
      INDEX_PREFIX         = "index/"
      OPENAI_API_KEY_PARAM = local.openai_key_param
      OPENAI_EMBED_MODEL   = "text-embedding-3-small"
      MIN_KEEP_RATIO       = "0.5"
      TIKTOKEN_CACHE_DIR   = "/var/task/tiktoken_cache"
    }
  }

  depends_on = [aws_iam_role_policy.indexer]
}

resource "aws_lambda_function_event_invoke_config" "indexer" {
  function_name                = aws_lambda_function.indexer.function_name
  maximum_retry_attempts       = 1
  maximum_event_age_in_seconds = 3600
}

resource "aws_lambda_permission" "s3_invokes_indexer" {
  statement_id   = "AllowS3DocsEvents"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.indexer.function_name
  principal      = "s3.amazonaws.com"
  source_arn     = aws_s3_bucket.kb.arn
  source_account = var.account_id
}

resource "aws_s3_bucket_notification" "docs" {
  bucket = aws_s3_bucket.kb.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.indexer.arn
    events              = ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]
    filter_prefix       = "docs/"
  }

  depends_on = [aws_lambda_permission.s3_invokes_indexer]
}
