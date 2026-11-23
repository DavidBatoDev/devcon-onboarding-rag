# DEBBIE RAG — Evaluation

Golden set: 20 answerable questions (all 15 documents) + 6 unanswerable. Judge pinned to
`gpt-4o-mini` (temperature 0.1, Chat Completions) with the exact LlamaIndex
Faithfulness/Relevancy prompts the v2 baseline used, so numbers stay comparable when the
answer model or platform changes. Harness: `eval/run_eval.py` (in-process or
`--base-url <deployment>`). Raw report of the deployed run: [eval_aws_api_gateway.md](eval_aws_api_gateway.md).

## 1. Platform swap (Cloud Run + Chroma → API Gateway + Lambda + NumPy), same model

**Retrieval is identical.** The index is an export of the same vectors; on all 26 golden
questions the new exact-cosine search returns the same top-20 candidates in the same order
with the same confidence tier (max |Δcos| 9e-5, OpenAI embedding jitter). The version
later rebuilt from S3 by the indexer Lambda (134 chunks) scores the same.

| Metric | v2 baseline (Jul 2026) | Old code, today (3 runs) | New code, in-process (2 runs) | **Deployed API (AWS)** |
|---|---|---|---|---|
| hit@5 (file) | 95% | 95% | 95% | **95%** |
| MRR | 0.835 | 0.835 | 0.835 | **0.810** ¹ |
| section-hit@5 | 100% | 100% | 100% | **100%** |
| false refusals | 0% | 0% | 0% | **0%** |
| faithfulness | 90% | 90 / 85 / 90% | 95 / 80% | **90%** |
| relevancy | 100% | 100% | 100% | **100%** |
| citation coverage | 100% | 100% | 100% | **100%** |
| honesty on unanswerable | 100% | 83 / 100 / 100% | 83 / 100% | **100%** |
| first token, warm p50 | 1.86 s ² | — | 1.13 s | **1.21 s** (p90 1.76 s) |

¹ One question (g03) has its top two candidates 0.00001 apart in fused score; embedding
jitter swaps ranks 1 and 2 between calls (hit@5 unaffected). The same flip reproduces
in-process. ² Cloud Run, measured 2026-11-23 from the Philippines.

Faithfulness on n=20 moves ±5–10 points between identical runs (the old code alone scored
85–90% the same day), so compare means, not single runs.

**Judge endpoint matters.** The same judge model on the Responses API scored the same kind
of answers ~7 points lower (85 / 75 / 75%) than on Chat Completions (the endpoint LlamaIndex
used). The judge is therefore pinned to Chat Completions, and production answers also use
Chat Completions (`OPENAI_API_STYLE=chat`) so the platform swap changed no model call.

## 2. Model swap: GPT-6 Luna (in-process, `reasoning.effort=none`)

| Metric | gpt-4o-mini | gpt-6-luna |
|---|---|---|
| faithfulness (5 runs) | 80–95% | **70 / 75 / 70 / 70 / 80%** |
| relevancy | 95–100% | 90–95% |
| citation coverage / honesty | 100% / 83–100% | 100% / 100% |
| first token p50 | 0.97–1.14 s | 1.03–1.37 s |
| output tokens / question | ~172 | ~127 |
| reasoning tokens | — | 0 (at effort `none`; `medium`, the default, spends them) |
| cost / question (uncached) | $0.00047 | **$0.00031 (−34%)** |

**Decision: stay on gpt-4o-mini.** Luna misses the gate (faithfulness ≥ 90%, relevancy
≥ 95%). Reading its failing answers, most are accurate and cited; what fails is the binary
judge marking sentences like "the sources don't specify the review timeline" as
unsupported — Luna applies the prompt's "say what the sources don't cover" rule far more
often. Re-deciding needs a claim-level judge or a prompt tuned for Luna, evaluated the same
way; switching back is one Terraform variable (`answer_model`, `reasoning_effort`).

Notes: Luna rejects `temperature` unless `reasoning.effort` is `none`; prompt-cache hits
appeared only when identical prompts repeated (eval re-runs), so production savings come
from the lower token price, not caching.

## 3. Latency and cold start

| | Cloud Run (v2) | Lambda (serverless) |
|---|---|---|
| first request after idle | 25.8 s to first byte, 32.9 s to first token | Lambda init p50 **2.6 s** (max 3.4 s, 12 cold starts); app startup incl. S3 index load 0.24 s |
| warm first token p50 | 1.86 s | 1.21 s |
| memory used | 1 GiB container | 191 MB of 1024 MB (p95) |

## 4. Failure drills (2026-11-23, before cutover)

| Drill | Result |
|---|---|
| 30 parallel questions | 10 answered (account concurrency quota = 10), rest rejected fast. First run returned bare **500**s (API Gateway reports a Lambda throttle as 500); fixed with gateway responses → **503 + Retry-After** (17) and 429 (3), CORS on errors. Throttles + 5xx alarms fired and recovered. |
| Deploy a broken build | Version 3 crashed on import; `release.py` smoke test failed; **`live` stayed on version 2**, public `/health` healthy throughout; Errors alarm fired. |
| Corrupt KB upload (invalid `docs/manifest.json`) | Indexer refused (`BuildRefused`), **pointer unchanged**, indexer-errors alarm fired; restoring the file → "unchanged". |
| 16 simultaneous S3 events (seeding 15 docs + manifest) | 1 build won, 8 lost the conditional pointer write and stood down, 7 saw an unchanged fingerprint. |
| KB rollback | `kb_admin.py promote <previous>`: API served it after **152 s**; roll-forward after 303 s (≤ 5-min refresh). |
| Invalid OpenAI key | `/health` healthy; `/ask/stream` → HTTP 200 with `meta` + `error` event (friendly message), no 500/stack trace. Not alarmed (handled error) — known gap. |
| Rate limits | 11th `/retrieve` and 21st `/ask` per minute → 429 + Retry-After; a forged `x-amzn-request-context` header is overwritten by the adapter. |
| Budget | `debbie-monthly` $5 (tag `project=debbie`), actual $0.00 so far; account-wide $30 budget above it. |

_Regenerate: `.venv/Scripts/python -m eval.run_eval --mode full [--base-url <api_url>] --out docs/<name>.md`._
