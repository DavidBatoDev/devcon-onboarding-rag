# DEBBIE RAG — Evaluation Results

_Generated 2026-07-11 15:50 · golden set: 20 answerable + 6 unanswerable questions · embed `text-embedding-3-small` · LLM `gpt-4o-mini` · collection `devcon_chunks_v2`_

## Retrieval

| Metric | Value |
|---|---|
| hit@1 (file) | 75% |
| hit@3 (file) | 90% |
| hit@5 (file) | 95% |
| MRR | 0.835 |
| section-hit@5 | 100% |
| gate hard-refusal on unanswerable (tier=none) | 17% |
| false-refusal rate (answerable → none) | 0% |

Top-1 cosine distribution — answerable: min/med/max = 0.50/0.64/0.78; unanswerable: 0.00/0.54/0.69 (gate: strong ≥ 0.5, weak ≥ 0.35)

## Configuration comparison

| Config | hit@5 | MRR | false-refusal |
|---|---|---|---|
| hybrid + no rerank | 95% | 0.835 | 0% |
| vector-only | 95% | 0.764 | 0% |
| hybrid + LLM rerank | 95% | 0.917 | 0% |

## Answer quality (LLM-as-judge)

| Metric | Value |
|---|---|
| Faithfulness (grounded answers) | 90% |
| Relevancy (grounded answers) | 100% |
| Citation coverage (≥1 inline citation) | 100% |
| Grounded rate on answerable | 100% |
| Answer-level honesty on unanswerable | 100% |

_Run: `venv/Scripts/python -m eval.run_eval --mode full --compare`_
