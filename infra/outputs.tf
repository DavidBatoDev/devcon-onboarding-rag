output "api_url" {
  description = "Base URL for the frontend (VITE_BACKEND_URL) and eval --base-url"
  value       = aws_api_gateway_stage.prod.invoke_url
}

output "api_function" {
  value = aws_lambda_function.api.function_name
}

output "api_latest_version" {
  value = aws_lambda_function.api.version
}

output "indexer_function" {
  value = aws_lambda_function.indexer.function_name
}

output "kb_bucket" {
  value = aws_s3_bucket.kb.bucket
}

output "github_plan_role_arn" {
  value = aws_iam_role.github_plan.arn
}

output "github_deploy_role_arn" {
  value = aws_iam_role.github_deploy.arn
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}
