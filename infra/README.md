# DEBBIE infrastructure (Terraform, AWS ap-southeast-1)

Everything the backend runs on, in one root module. Account `484907520476`
(guarded by `allowed_account_ids`), state in `s3://debbie-tfstate-484907520476`
with S3-native locking.

```
 Browser (Vercel) ──HTTPS, NDJSON stream──► API Gateway REST  "debbie-api" / stage prod
                                            │  every route: Lambda proxy, response_transfer_mode=STREAM
                                            │  throttle 5 rps / burst 10 (retrieve: 2 rps)
                                            │  gateway responses: 5xx → 503 + Retry-After, CORS on errors
                                            ▼
                         Lambda debbie-api:live  (alias moved by scripts/release.py)
                         Python 3.13 arm64 · FastAPI via Lambda Web Adapter (response_stream)
                         ├─ SSM  /debbie/prod/openai_api_key   (cold start)
                         ├─ S3   index/current.json → index/<version>/   (cold start + every 5 min)
                         ├─ DynamoDB debbie-ratelimit (per-IP window, TTL) · debbie-feedback
                         └─ OpenAI (gpt-4o-mini answers, text-embedding-3-small queries)

 s3://debbie-kb-484907520476/docs/*  ──ObjectCreated/Removed──►  Lambda debbie-indexer
   private · versioned · SSE-S3 · TLS-only                        → index/<version>/ + CAS pointer swap

 GitHub Actions ──OIDC──► debbie-plan-github (PRs, read-only) · debbie-deploy-github (production env)
 CloudWatch alarms ──► SNS debbie-alerts ──► email · AWS Budget debbie-monthly ($5, tag project=debbie)
```

| File | What |
|---|---|
| `versions.tf` | Terraform ≥ 1.10, AWS provider `~> 6.73` (REST streaming needs ≥ 6.25), S3 backend, default tags |
| `variables.tf` | throttles, concurrency, memory/timeouts, models, budget, latency alarm |
| `storage.tf` | KB bucket (BPA, versioning, SSE-S3, TLS-only, lifecycle), DynamoDB tables |
| `secrets.tf` | SSM parameter names/ARNs only — values are never managed here |
| `iam.tf` | function roles (least privilege), API Gateway logging role, GitHub OIDC + plan/deploy roles |
| `lambda.tf` | both functions, `live` alias, log groups, S3 → indexer notification |
| `apigw.tf` | REST API, STREAM integrations, gateway responses, stage, access logs, throttles |
| `monitoring.tf` | SNS topic + email, 6 alarms, budget |

## First run (done once, 2026-11-23)

1. State bucket by hand: `debbie-tfstate-484907520476` (versioning, SSE-S3, BPA, TLS-only policy).
2. Secrets: `python scripts/set_secrets.py --alert-email <you>` (OpenAI key from `server/.env`).
3. `python scripts/build_lambda.py all` → `terraform -chdir=infra init && terraform -chdir=infra apply`.
4. Confirm the SNS subscription email AWS sends.
5. KB: `python scripts/kb_admin.py upload-index app/storage/index/<v>` + `promote <v>`, then
   `python scripts/kb_admin.py seed-docs` (the indexer builds a new version from `docs/`).

## Deploy (normally CI)

```
python scripts/build_lambda.py all
terraform -chdir=infra apply          # publishes a new debbie-api version; `live` does not move
python scripts/release.py             # smoke-test that version directly, then move `live`
```

A version that fails the smoke test never receives traffic. Rollback: `python scripts/release.py --rollback`.

## Knowledge base

- Update: upload/replace `.docx`/`.txt`/`.md` under `docs/` (optional `docs/manifest.json`
  maps file name → `file_id`/`title`/`drive_link`). The indexer builds and validates a new
  version and moves `index/current.json`; warm API instances pick it up within 5 minutes.
- Rollback: `python scripts/kb_admin.py promote <previous-version>` (one object write).
- Status: `python scripts/kb_admin.py status`.
- The indexer refuses invalid input (bad manifest, no text, an index shrinking below 50% of
  the live one) and never moves the pointer in that case; overlapping builds are resolved by
  a conditional (If-Match) pointer write.

## Model switch

`answer_model` / `reasoning_effort` / `openai_api_style` variables → apply → release.
GPT-6 Luna was evaluated and did not meet the faithfulness gate (see `docs/eval.md`).

## Known limits

- Lambda concurrency quota is 10 (new account); reserved concurrency (5 api / 1 indexer) is
  set via `api_reserved_concurrency` / `indexer_reserved_concurrency` once the requested
  increase to 1000 is approved. Until then the account limit of 10 is the ceiling.
- `aws_api_gateway_account` sets a regional, account-wide setting that `destroy` leaves.
- The deploy role can manage `debbie-*` IAM roles (inherent to Terraform-managed IAM);
  it is only assumable from the `production` environment with a required reviewer.
- Handled pipeline errors (e.g. OpenAI down) return an `error` stream event and are logged,
  but are not alarmed (a log metric filter would cost $0.30/month).
