# DEBBIE RAG — Evaluation Results

_Generated 2026-11-23 20:24 · target: http `https://pi1xpaozw1.execute-api.ap-southeast-1.amazonaws.com/prod` · golden set: 20 answerable + 6 unanswerable · embed `text-embedding-3-small` · answer model `gpt-4o-mini` · judge `gpt-4o-mini` · index `20261123T120943Z`_

## Retrieval

| Metric | Value |
|---|---|
| hit@1 (file) | 70% |
| hit@3 (file) | 90% |
| hit@5 (file) | 95% |
| MRR | 0.810 |
| section-hit@5 | 100% |
| gate hard-refusal on unanswerable (tier=none) | 17% |
| false-refusal rate (answerable → none) | 0% |

## Answer quality (LLM-as-judge)

| Metric | Value |
|---|---|
| Faithfulness (grounded answers) | 90% |
| Relevancy (grounded answers) | 100% |
| Citation coverage (≥1 inline citation) | 100% |
| Grounded rate on answerable | 100% |
| Answer-level honesty on unanswerable | 100% |
| First token p50 / p90 | 1.21 s / 1.76 s |
