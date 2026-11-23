# Alarms -> one SNS topic -> email (address from SSM, not in the repo). Stays inside
# the free tier: 7 standard alarms, 1 topic, email delivery, 1 budget without actions.

resource "aws_sns_topic" "alerts" {
  name = "debbie-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = data.aws_ssm_parameter.alert_email.value # confirm via the email AWS sends
}

locals {
  api_name = "debbie-api"
  stage    = "prod"
  alarms = {
    "debbie-api-5xx" = {
      namespace = "AWS/ApiGateway", metric = "5XXError", stat = "Sum", threshold = 0, op = "GreaterThanThreshold"
      dims      = { ApiName = local.api_name, Stage = local.stage }, periods = 1
      desc      = "API Gateway returned 5xx (Lambda crash, timeout, or malformed stream)"
    }
    "debbie-api-4xx-spike" = {
      namespace = "AWS/ApiGateway", metric = "4XXError", stat = "Sum", threshold = 50, op = "GreaterThanThreshold"
      dims      = { ApiName = local.api_name, Stage = local.stage }, periods = 1
      desc      = "Many 4xx/429 in 5 min: abuse, a bot, or real growth hitting the limits"
    }
    "debbie-api-latency-p95" = {
      namespace = "AWS/ApiGateway", metric = "Latency", stat = "p95", threshold = var.latency_alarm_ms, op = "GreaterThanThreshold"
      dims      = { ApiName = local.api_name, Stage = local.stage }, periods = 3
      desc      = "p95 end-to-end latency (whole streamed answer) above threshold for 15 min"
    }
    "debbie-api-lambda-errors" = {
      namespace = "AWS/Lambda", metric = "Errors", stat = "Sum", threshold = 0, op = "GreaterThanThreshold"
      dims      = { FunctionName = "debbie-api" }, periods = 1
      desc      = "debbie-api invocation errors"
    }
    "debbie-api-throttles" = {
      namespace = "AWS/Lambda", metric = "Throttles", stat = "Sum", threshold = 0, op = "GreaterThanThreshold"
      dims      = { FunctionName = "debbie-api" }, periods = 1
      desc      = "debbie-api hit its concurrency ceiling"
    }
    "debbie-indexer-errors" = {
      namespace = "AWS/Lambda", metric = "Errors", stat = "Sum", threshold = 1, op = "GreaterThanOrEqualToThreshold"
      dims      = { FunctionName = "debbie-indexer" }, periods = 1
      desc      = "Indexer failed or refused a build (the live index is unchanged)"
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "this" {
  for_each = local.alarms

  alarm_name          = each.key
  alarm_description   = each.value.desc
  namespace           = each.value.namespace
  metric_name         = each.value.metric
  dimensions          = each.value.dims
  statistic           = contains(["Sum", "Average", "Maximum", "Minimum", "SampleCount"], each.value.stat) ? each.value.stat : null
  extended_statistic  = contains(["Sum", "Average", "Maximum", "Minimum", "SampleCount"], each.value.stat) ? null : each.value.stat
  period              = 300
  evaluation_periods  = each.value.periods
  threshold           = each.value.threshold
  comparison_operator = each.value.op
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

# DEBBIE-only monthly budget (resources tagged project=debbie; the tag is activated
# as a cost-allocation tag). The account-wide $30 budget sits above it.
resource "aws_budgets_budget" "debbie" {
  name         = "debbie-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:project$debbie"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [data.aws_ssm_parameter.alert_email.value]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [data.aws_ssm_parameter.alert_email.value]
  }
}
