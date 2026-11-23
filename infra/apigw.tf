# REST API (HTTP APIs buffer responses, so they can't stream). Every integration
# is a Lambda proxy in STREAM mode: with LWA in response_stream mode a BUFFERED
# integration returns 502 (Phase 0 spike). CORS is handled by FastAPI — its
# headers and the OPTIONS preflight pass through the proxy unchanged.

resource "aws_api_gateway_rest_api" "api" {
  name        = "debbie-api"
  description = "DEBBIE RAG API (Lambda, response streaming)"
  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

locals {
  # InvokeWithResponseStream on the `live` alias.
  stream_uri = "arn:aws:apigateway:${var.region}:lambda:path/2021-11-15/functions/${aws_lambda_alias.api_live.arn}/response-streaming-invocations"
  root_id    = aws_api_gateway_rest_api.api.root_resource_id
}

resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = local.root_id
  path_part   = "{proxy+}"
}

resource "aws_api_gateway_resource" "api" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = local.root_id
  path_part   = "api"
}

resource "aws_api_gateway_resource" "v1" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_resource.api.id
  path_part   = "v1"
}

resource "aws_api_gateway_resource" "v1_proxy" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_resource.v1.id
  path_part   = "{proxy+}"
}

# Explicit resource so /retrieve (the playground's retrieval inspector) gets its own throttle.
resource "aws_api_gateway_resource" "retrieve" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_resource.v1.id
  path_part   = "retrieve"
}

locals {
  routes = {
    root          = { resource_id = local.root_id, method = "ANY" }
    proxy         = { resource_id = aws_api_gateway_resource.proxy.id, method = "ANY" }
    v1_proxy      = { resource_id = aws_api_gateway_resource.v1_proxy.id, method = "ANY" }
    retrieve_post = { resource_id = aws_api_gateway_resource.retrieve.id, method = "POST" }
    retrieve_opts = { resource_id = aws_api_gateway_resource.retrieve.id, method = "OPTIONS" }
  }
}

resource "aws_api_gateway_method" "route" {
  for_each      = local.routes
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = each.value.resource_id
  http_method   = each.value.method
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "route" {
  for_each                = local.routes
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = each.value.resource_id
  http_method             = aws_api_gateway_method.route[each.key].http_method
  type                    = "AWS_PROXY"
  integration_http_method = "POST"
  uri                     = local.stream_uri
  response_transfer_mode  = "STREAM"
  timeout_milliseconds    = var.api_timeout_s * 1000
}

# Errors API Gateway generates itself never reach FastAPI, so they'd lack CORS headers
# and (for Lambda concurrency throttles) surface as a bare 500. Phase 9 drill: 30
# parallel questions -> 10 answered, 20 x "500 DEFAULT_5XX" when the concurrency
# ceiling was hit. Map them to an honest, retryable, browser-readable response.
# The 5XXError alarm still counts these 503s.
locals {
  gateway_responses = {
    DEFAULT_5XX = { status = "503", message = "DEBBIE is busy or restarting. Please try again in a few seconds." }
    THROTTLED   = { status = "429", message = "Too many requests. Please slow down and try again shortly." }
    DEFAULT_4XX = { status = null, message = "$context.error.message" }
  }
}

resource "aws_api_gateway_gateway_response" "this" {
  for_each      = local.gateway_responses
  rest_api_id   = aws_api_gateway_rest_api.api.id
  response_type = each.key
  status_code   = each.value.status

  response_templates = {
    "application/json" = jsonencode({ detail = each.value.message, requestId = "$context.requestId" })
  }

  response_parameters = {
    "gatewayresponse.header.Access-Control-Allow-Origin" = "'*'"
    "gatewayresponse.header.Retry-After"                 = "'5'"
    "gatewayresponse.header.Cache-Control"               = "'no-store'"
  }
}

resource "aws_lambda_permission" "apigw_invokes_api" {
  statement_id  = "AllowApiGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  qualifier     = aws_lambda_alias.api_live.name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api.execution_arn}/*/*"
}

resource "aws_api_gateway_deployment" "api" {
  rest_api_id = aws_api_gateway_rest_api.api.id

  triggers = {
    redeploy = sha1(jsonencode([
      local.routes,
      [for k, i in aws_api_gateway_integration.route : [i.id, i.uri, i.response_transfer_mode, i.timeout_milliseconds]],
      local.gateway_responses,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [aws_api_gateway_integration.route, aws_api_gateway_gateway_response.this]
}

# Account-level (per region) role that lets API Gateway write logs.
resource "aws_api_gateway_account" "this" {
  cloudwatch_role_arn = aws_iam_role.apigw_logs.arn
  depends_on          = [aws_iam_role_policy_attachment.apigw_logs]
}

resource "aws_cloudwatch_log_group" "apigw_access" {
  name              = "/aws/apigateway/debbie-api-access"
  retention_in_days = var.log_retention_days
}

resource "aws_api_gateway_stage" "prod" {
  rest_api_id          = aws_api_gateway_rest_api.api.id
  deployment_id        = aws_api_gateway_deployment.api.id
  stage_name           = "prod"
  xray_tracing_enabled = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.apigw_access.arn
    format = jsonencode({
      requestId  = "$context.requestId"
      ip         = "$context.identity.sourceIp"
      method     = "$context.httpMethod"
      path       = "$context.path"
      status     = "$context.status"
      latencyMs  = "$context.responseLatency"
      integLatMs = "$context.integrationLatency"
      bytes      = "$context.responseLength"
      throttled  = "$context.error.responseType"
      ua         = "$context.identity.userAgent"
    })
  }

  depends_on = [aws_api_gateway_account.this]
}

resource "aws_api_gateway_method_settings" "all" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  method_path = "*/*"

  settings {
    throttling_rate_limit  = var.throttle_rate
    throttling_burst_limit = var.throttle_burst
    metrics_enabled        = true
    logging_level          = "ERROR"
  }
}

resource "aws_api_gateway_method_settings" "retrieve" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  stage_name  = aws_api_gateway_stage.prod.stage_name
  method_path = "api/v1/retrieve/POST"

  settings {
    throttling_rate_limit  = var.retrieve_throttle_rate
    throttling_burst_limit = var.retrieve_throttle_rate * 2
    metrics_enabled        = true
    logging_level          = "ERROR"
  }
}
