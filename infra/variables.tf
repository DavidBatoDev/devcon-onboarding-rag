variable "region" {
  type    = string
  default = "ap-southeast-1"
}

variable "account_id" {
  type    = string
  default = "484907520476"
}

variable "github_repo" {
  description = "owner/name of the repository allowed to deploy via OIDC"
  type        = string
  default     = "DavidBatoDev/devcon-onboarding-rag"
}

# --- API Gateway throttling (stage-wide, plus a tighter override on /retrieve)
variable "throttle_rate" {
  type    = number
  default = 5
}

variable "throttle_burst" {
  type    = number
  default = 10
}

variable "retrieve_throttle_rate" {
  type    = number
  default = 2
}

# --- Lambda
# Reserved concurrency is the hard ceiling on parallel answers (and so on AWS +
# OpenAI spend). It needs the account's concurrency quota above 10 (requested
# 2026-11-23); null = unreserved until then, when the account limit of 10 is the cap.
variable "api_reserved_concurrency" {
  type    = number
  default = null
}

variable "indexer_reserved_concurrency" {
  type    = number
  default = null
}

variable "api_memory_mb" {
  type    = number
  default = 1024
}

variable "api_timeout_s" {
  description = "Lambda timeout; the STREAM integrations use the same value."
  type        = number
  default     = 90
}

variable "indexer_memory_mb" {
  type    = number
  default = 1024
}

variable "lwa_layer_arn" {
  description = "Lambda Web Adapter v1.1.0, arm64 (verified in ap-southeast-1, Phase 0)"
  type        = string
  default     = "arn:aws:lambda:ap-southeast-1:753240598075:layer:LambdaAdapterLayerArm64:30"
}

variable "log_retention_days" {
  type    = number
  default = 14
}

# --- Models (switch back = change one variable and apply)
variable "answer_model" {
  type    = string
  default = "gpt-4o-mini"
}

variable "openai_api_style" {
  description = "\"chat\" (Chat Completions, as the v2 baseline) or \"responses\""
  type        = string
  default     = "chat"
}

variable "reasoning_effort" {
  description = "\"none\" for gpt-6-luna (keeps temperature valid, no reasoning tokens); \"\" for gpt-4o-mini"
  type        = string
  default     = ""
}

# --- Guardrails
variable "budget_usd" {
  type    = number
  default = 5
}

variable "latency_alarm_ms" {
  description = "API Gateway p95 latency alarm threshold (full streamed answers included)"
  type        = number
  default     = 20000
}
