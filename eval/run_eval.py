"""Evaluation harness for the DEBBIE RAG pipeline.

Modes:
  retrieval (default) — hit@k / MRR / section-hit / gate honesty. Embedding
                        cost only (one query embedding per golden item).
  full                — also generates answers and judges them with
                        LLM-as-judge (faithfulness + relevancy) plus citation
                        coverage. ~3 gpt-4o-mini calls per item.

  --compare           — retrieval matrix: {vector-only, hybrid} x {rerank off/on}.
                        Decides RERANK_MODE and validates the hybrid choice.

Usage (from server/):
  venv/Scripts/python -m eval.run_eval
  venv/Scripts/python -m eval.run_eval --compare
  venv/Scripts/python -m eval.run_eval --mode full --out docs/eval.md
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from app.core.config import settings  # noqa: E402

GOLDEN = SERVER_ROOT / "eval" / "golden.jsonl"
RESULTS_DIR = SERVER_ROOT / "eval" / "results"


def load_golden():
    items = [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]
    return [i for i in items if i["answerable"]], [i for i in items if not i["answerable"]]


def file_ids_of(pipeline, candidates, k):
    out = []
    for c in candidates[:k]:
        node = pipeline.nodes.get(c.node_id)
        if node:
            out.append(node.metadata.get("file_id", ""))
    return out


def eval_retrieval(pipeline, answerable, unanswerable, use_rerank=False):
    """Core retrieval metrics against the golden set."""
    hits = {1: 0, 3: 0, 5: 0}
    rr = []
    section_hits, section_total = 0, 0
    refusals_ok, false_refusals = 0, 0
    top_scores_ans, top_scores_unans = [], []

    for item in answerable:
        candidates = pipeline.retrieve(item["question"])
        if use_rerank:
            candidates = pipeline._rerank(item["question"], candidates,
                                          settings.TOP_K_RETRIEVAL, streaming=False)
        tier = pipeline.gate(candidates)
        if tier == "none":
            false_refusals += 1
        cos = [c.cosine for c in candidates if c.cosine is not None]
        top_scores_ans.append(max(cos, default=0.0))

        expected = set(item["expected_file_ids"])
        got = file_ids_of(pipeline, candidates, 5)
        for k in hits:
            if expected & set(got[:k]):
                hits[k] += 1
        rank = next((i + 1 for i, fid in enumerate(got) if fid in expected), None)
        rr.append(1.0 / rank if rank else 0.0)

        if item.get("expected_section"):
            section_total += 1
            secs = [pipeline.nodes[c.node_id].metadata.get("section", "")
                    for c in candidates[:5] if c.node_id in pipeline.nodes]
            if item["expected_section"] in secs:
                section_hits += 1

    for item in unanswerable:
        candidates = pipeline.retrieve(item["question"])
        tier = pipeline.gate(candidates)
        cos = [c.cosine for c in candidates if c.cosine is not None]
        top_scores_unans.append(max(cos, default=0.0))
        if tier == "none":
            refusals_ok += 1

    n = len(answerable)
    return {
        "hit@1": hits[1] / n, "hit@3": hits[3] / n, "hit@5": hits[5] / n,
        "mrr": sum(rr) / n,
        "section_hit@5": (section_hits / section_total) if section_total else None,
        "refusal_accuracy": refusals_ok / len(unanswerable) if unanswerable else None,
        "false_refusal_rate": false_refusals / n,
        "top_cosine_answerable": (min(top_scores_ans), statistics.median(top_scores_ans),
                                  max(top_scores_ans)),
        "top_cosine_unanswerable": (min(top_scores_unans), statistics.median(top_scores_unans),
                                    max(top_scores_unans)) if top_scores_unans else None,
    }


def eval_full(service, answerable, unanswerable):
    """Answer-quality metrics: LLM-as-judge + citation coverage + honesty."""
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
    from llama_index.core import Settings as LlamaSettings

    faith = FaithfulnessEvaluator(llm=LlamaSettings.llm)
    relev = RelevancyEvaluator(llm=LlamaSettings.llm)

    faith_pass, relev_pass, cite_cov, grounded_n = 0, 0, 0, 0
    honest_unans = 0
    per_item = []

    for item in answerable:
        result = service.query(item["question"], nocache=True)
        contexts = [s.text for s in result.sources]
        rec = {"id": item["id"], "grounded": result.grounded,
               "confidence": result.confidence,
               "cited": [s.n for s in result.sources if s.cited]}
        if result.grounded and contexts:
            grounded_n += 1
            if rec["cited"]:
                cite_cov += 1
            try:
                f = faith.evaluate(query=item["question"], response=result.answer,
                                   contexts=contexts)
                r = relev.evaluate(query=item["question"], response=result.answer,
                                   contexts=contexts)
                rec["faithful"], rec["relevant"] = bool(f.passing), bool(r.passing)
                faith_pass += bool(f.passing)
                relev_pass += bool(r.passing)
            except Exception as e:
                rec["judge_error"] = str(e)
        per_item.append(rec)
        print(f"  {item['id']}: grounded={result.grounded} conf={result.confidence} "
              f"cited={rec['cited']} faithful={rec.get('faithful')} relevant={rec.get('relevant')}")

    for item in unanswerable:
        result = service.query(item["question"], nocache=True)
        # Near-domain unanswerables legitimately pass the gate (they embed close
        # to DEVCON content), so honesty is judged at the ANSWER level: does the
        # reply avoid fabricating and acknowledge the docs don't cover it?
        if not result.grounded:
            ok = True  # hard refusal — honest by construction
        else:
            verdict = str(LlamaSettings.llm.complete(
                "You are auditing an assistant answer for honesty. The assistant's "
                "documents do NOT contain the information needed to answer the "
                "question. Reply PASS if the answer acknowledges the information "
                "isn't available/covered (general guidance clearly labelled as such "
                "is fine). Reply FAIL if it fabricates specific facts, names, "
                "numbers, or policies as if documented.\n\n"
                f"Question: {item['question']}\n\nAnswer:\n{result.answer}\n\nVerdict:"
            )).strip().upper()
            ok = verdict.startswith("PASS")
        honest_unans += ok
        per_item.append({"id": item["id"], "grounded": result.grounded, "honest": ok})
        print(f"  {item['id']}: grounded={result.grounded} honest={ok}")

    return {
        "faithfulness": faith_pass / grounded_n if grounded_n else None,
        "relevancy": relev_pass / grounded_n if grounded_n else None,
        "citation_coverage": cite_cov / grounded_n if grounded_n else None,
        "grounded_rate_answerable": grounded_n / len(answerable),
        "honest_refusal_rate": honest_unans / len(unanswerable) if unanswerable else None,
        "per_item": per_item,
    }


def fmt_pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def write_report(out_path, retrieval, compare_rows, full, n_ans, n_unans):
    lines = [
        "# DEBBIE RAG — Evaluation Results",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} · golden set: {n_ans} answerable + "
        f"{n_unans} unanswerable questions · embed `{settings.OPENAI_EMBED_MODEL}` · "
        f"LLM `{settings.OPENAI_LLM_MODEL}` · collection `{settings.CHROMA_COLLECTION_NAME}`_",
        "",
        "## Retrieval",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| hit@1 (file) | {fmt_pct(retrieval['hit@1'])} |",
        f"| hit@3 (file) | {fmt_pct(retrieval['hit@3'])} |",
        f"| hit@5 (file) | {fmt_pct(retrieval['hit@5'])} |",
        f"| MRR | {retrieval['mrr']:.3f} |",
        f"| section-hit@5 | {fmt_pct(retrieval['section_hit@5'])} |",
        f"| gate hard-refusal on unanswerable (tier=none) | {fmt_pct(retrieval['refusal_accuracy'])} |",
        f"| false-refusal rate (answerable → none) | {fmt_pct(retrieval['false_refusal_rate'])} |",
        "",
        "Top-1 cosine distribution — answerable: min/med/max = "
        f"{'/'.join(f'{v:.2f}' for v in retrieval['top_cosine_answerable'])}; "
        "unanswerable: "
        + ("/".join(f"{v:.2f}" for v in retrieval["top_cosine_unanswerable"])
           if retrieval["top_cosine_unanswerable"] else "—")
        + f" (gate: strong ≥ {settings.CONFIDENCE_STRONG}, weak ≥ {settings.CONFIDENCE_WEAK})",
        "",
    ]
    if compare_rows:
        lines += [
            "## Configuration comparison",
            "",
            "| Config | hit@5 | MRR | false-refusal |",
            "|---|---|---|---|",
        ]
        for name, m in compare_rows:
            lines.append(f"| {name} | {fmt_pct(m['hit@5'])} | {m['mrr']:.3f} | "
                         f"{fmt_pct(m['false_refusal_rate'])} |")
        lines.append("")
    if full:
        lines += [
            "## Answer quality (LLM-as-judge)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Faithfulness (grounded answers) | {fmt_pct(full['faithfulness'])} |",
            f"| Relevancy (grounded answers) | {fmt_pct(full['relevancy'])} |",
            f"| Citation coverage (≥1 inline citation) | {fmt_pct(full['citation_coverage'])} |",
            f"| Grounded rate on answerable | {fmt_pct(full['grounded_rate_answerable'])} |",
            f"| Answer-level honesty on unanswerable | {fmt_pct(full['honest_refusal_rate'])} |",
            "",
        ]
    lines.append("_Run: `venv/Scripts/python -m eval.run_eval --mode full --compare`_\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", default=str(SERVER_ROOT / "docs" / "eval.md"))
    args = ap.parse_args()

    answerable, unanswerable = load_golden()
    print(f"golden set: {len(answerable)} answerable, {len(unanswerable)} unanswerable")

    from app.services.llamaindex_service import rag_service
    pipeline = rag_service.pipeline

    print("\n== retrieval (hybrid, rerank off) ==")
    retrieval = eval_retrieval(pipeline, answerable, unanswerable)
    for k, v in retrieval.items():
        print(f"  {k}: {v}")

    compare_rows = []
    if args.compare:
        compare_rows.append(("hybrid + no rerank", retrieval))
        print("\n== vector-only ==")
        bm25 = pipeline._bm25
        pipeline._bm25 = None
        compare_rows.append(("vector-only", eval_retrieval(pipeline, answerable, unanswerable)))
        pipeline._bm25 = bm25
        print("\n== hybrid + LLM rerank ==")
        old_mode = settings.RERANK_MODE
        settings.RERANK_MODE = "always"
        compare_rows.append(("hybrid + LLM rerank",
                             eval_retrieval(pipeline, answerable, unanswerable, use_rerank=True)))
        settings.RERANK_MODE = old_mode
        for name, m in compare_rows:
            print(f"  {name}: hit@5={m['hit@5']:.2f} mrr={m['mrr']:.3f}")

    full = None
    if args.mode == "full":
        print("\n== full answer-quality eval (LLM-as-judge) ==")
        full = eval_full(rag_service, answerable, unanswerable)
        for k, v in full.items():
            if k != "per_item":
                print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"run_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"retrieval": retrieval,
                   "compare": [{"name": n, **m} for n, m in compare_rows],
                   "full": full}, f, indent=2, default=str)

    write_report(Path(args.out), retrieval, compare_rows, full,
                 len(answerable), len(unanswerable))


if __name__ == "__main__":
    main()
