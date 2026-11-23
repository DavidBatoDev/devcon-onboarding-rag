# Secrets live in SSM Parameter Store and are written with scripts/set_secrets.py,
# never by Terraform: managing them here would copy the decrypted value into state.
# Terraform only needs their names/ARNs (for IAM) — nothing reads the secret value.
#
#   /debbie/prod/openai_api_key   SecureString (AWS-managed aws/ssm key)
#   /debbie/prod/alert_email      String (not secret; just kept out of the repo)

locals {
  openai_key_param     = "/debbie/prod/openai_api_key"
  openai_key_param_arn = "arn:aws:ssm:${var.region}:${var.account_id}:parameter${local.openai_key_param}"
}

data "aws_ssm_parameter" "alert_email" {
  name = "/debbie/prod/alert_email"
}
