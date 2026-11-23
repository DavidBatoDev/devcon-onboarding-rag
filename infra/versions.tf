terraform {
  required_version = ">= 1.10"

  required_providers {
    # REST API response streaming (response_transfer_mode) needs >= 6.25.
    aws     = { source = "hashicorp/aws", version = "~> 6.73" }
    archive = { source = "hashicorp/archive", version = "~> 2.7" }
  }

  # State bucket is created by hand once (see README "First run").
  backend "s3" {
    bucket       = "debbie-tfstate-484907520476"
    key          = "debbie/prod/terraform.tfstate"
    region       = "ap-southeast-1"
    encrypt      = true
    use_lockfile = true
  }
}

# Credentials: locally the default chain (profile `default`); in CI, GitHub OIDC.
provider "aws" {
  region              = var.region
  allowed_account_ids = [var.account_id]

  default_tags {
    tags = {
      project    = "debbie"
      env        = "prod"
      managed-by = "terraform"
    }
  }
}
