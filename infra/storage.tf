# Knowledge base: docs/ (source documents, private) and index/ (built versions + pointer).

resource "aws_s3_bucket" "kb" {
  bucket = "debbie-kb-${var.account_id}"
}

resource "aws_s3_bucket_public_access_block" "kb" {
  bucket                  = aws_s3_bucket.kb.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "kb" {
  bucket = aws_s3_bucket.kb.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_versioning" "kb" {
  bucket = aws_s3_bucket.kb.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "kb" {
  bucket = aws_s3_bucket.kb.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "kb" {
  bucket = aws_s3_bucket.kb.id

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
  # Old index/ versions are pruned to the newest 10 by the indexer itself.
}

data "aws_iam_policy_document" "kb_tls_only" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.kb.arn, "${aws_s3_bucket.kb.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "kb" {
  bucket     = aws_s3_bucket.kb.id
  policy     = data.aws_iam_policy_document.kb_tls_only.json
  depends_on = [aws_s3_bucket_public_access_block.kb]
}

# Feedback (thumbs up/down) — durable across deploys; no client IP stored.
resource "aws_dynamodb_table" "feedback" {
  name         = "debbie-feedback"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }
}

# Per-IP fixed-window counters; items expire via TTL a few minutes after their window.
resource "aws_dynamodb_table" "ratelimit" {
  name         = "debbie-ratelimit"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"

  attribute {
    name = "pk"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}
