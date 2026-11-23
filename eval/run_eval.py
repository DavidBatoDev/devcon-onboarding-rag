"""Evaluation harness for the DEBBIE RAG pipeline.

Targets:
  (default)            in-process pipeline over the local/S3 index (server/.env config)
  --base-url URL       a deployment over HTTP (/retrieve + /ask/stream), e.g. the
                       API Gateway stage URL; requests are paced under the rate limits

Modes:
  retrieval (default)  hit@k / MRR / section-hit / gate honesty (embedding cost only)
  full                 also generates answers and judges them (faithfulness + relevancy,
                       LLM-as-judge) plus citation coverage, honesty on unanswerables,
                       first-token latency and (in-process) tokens + cost per question

The judge is pinned (JUDGE_MODEL, default gpt-4o-mini, temperature 0.1) and uses the
exact LlamaIndex Faithfulness/Relevancy prompts the v2 baseline was scored with, so
results stay comparable when the answer model changes.

Usage (from server/):
  .venv/Scripts/python -m eval.run_eval
  .venv/Scripts/python -m eval.run_eval --mode full --compare --out docs/eval_local.md
  .venv/Scripts/python -m eval.run_eval --mode full --base-url https://<api>/prod --out docs/eval_aws.md
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from app.core.config import settings  # noqa: E402

GOLDEN = SERVER_ROOT / "eval" / "golden.jsonl"
RESULTS_DIR = SERVER_ROOT / "eval" / "results"

# USD per 1M tokens (input, cached input, output) — OpenAI pricing page, 2026-11-23.
PRICES = {"gpt-6-luna": (0.10, 0.01, 0.50), "gpt-4o-mini": (0.15, 0.075, 0.60)}

# Verbatim from llama_index.core.evaluation (faithfulness.py / relevancy.py),
# the judges that produced the v2 baseline in docs/eval.md.
FAITHFULNESS_TEMPLATE = (
    "Please tell if a given piece of information is supported by the context.\n"
    "You need to answer with either YES or NO.\n"
    "Answer YES if any of the context supports the information, even if most of the context "
    "is unrelated. Some examples are provided below. \n\n"
    "Information: Apple pie is generally double-crusted.\n"
    "Context: An apple pie is a fruit pie in which the principal filling ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard "
    "or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above and below the filling; the upper "
    "crust may be solid or latticed (woven of crosswise strips).\n"
    "Answer: YES\n"
    "Information: Apple pies tastes bad.\n"
    "Context: An apple pie is a fruit pie in which the principal filling ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard "
    "or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above and below the filling; the upper "
    "crust may be solid or latticed (woven of crosswise strips).\n"
    "Answer: NO\n"
    "Information: {query_str}\n"
    "Context: {context_str}\n"
    "Answer: "
)
RELEVANCY_TEMPLATE = (
    "Your task is to evaluate if the response for the query     is in line with the context "
    "information provided.\n"
    "You have two options to answer. Either YES/ NO.\n"
    "Answer - YES, if the response for the query     is in line with context information "
    "otherwise NO.\n"
    "Query and Response: \n {query_str}\n"
    "Context: \n {context_str}\n"
    "Answer: "
)
HONESTY_TEMPLATE = (
    "You are auditing an assistant answer for honesty. The assistant's documents do NOT "
    "contain the information needed to answer the question. Reply PASS if the answer "
    "acknowledges the information isn't available/covered (general guidance clearly labelled "
    "as such is fine). Reply FAIL if it fabricates specific facts, names, numbers, or "
    "policies as if documented.\n\nQuestion: {question}\n\nAnswer:\n{answer}\n\nVerdict:"
)


def load_golden():
    items = [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]
    return [i for i in items if i["answerable"]], [i for i in items if not i["answerable"]]


# ------------------------------------------------------------------ targets

class LocalTarget:
    name = "in-process"

    def __init__(self):
        from app.services import llm
        from app.services.rag_service import rag_service

        self.models = []

        def factory(temperature=None):
            m = llm.answer_model(temperature)
            self.models.append(m)
            return m

        rag_service._llm_factory = factory
        rag_service.startup()
        self.svc = rag_service
        self.pipeline = rag_service.pipeline
        self.index_version = rag_service.store.version

    def retrieve(self, question):
        cands = self.pipeline.retrieve(question)
        out = []
        for c in cands:
            md = self.pipeline.nodes[c.node_id].metadata
            out.append({"chunk_id": c.node_id, "file_id": md.get("file_id", ""),
                        "section": md.get("section", ""), "cosine": c.cosine})
        return out, self.pipeline.gate(cands)

    def answer(self, question):
        t0 = time.perf_counter()
        first, done, sources = None, None, []
        n_models = len(self.models)
        for kind, payload in self.svc.query_stream(question, nocache=True):
            if kind == "token" and first is None:
                first = time.perf_counter() - t0
            elif kind == "sources":
                sources = payload["sources"]
            elif kind == "done":
                done = payload
        usage = [u for m in self.models[n_models:] for u in m.usage]
        return {"answer": done.answer, "grounded": done.grounded, "confidence": done.confidence,
                "contexts": [s["text"] for s in sources],
                "cited": [s.n for s in done.sources if s.cited],
                "first_token_s": first, "usage": usage}


class RemoteTarget:
    name = "http"

    def __init__(self, base_url, retrieve_pace=6.5, ask_pace=3.2):
        self.base = base_url.rstrip("/")
        self.retrieve_pace, self.ask_pace = retrieve_pace, ask_pace
        self._last = {}
        stats = self._json("GET", "/api/v1/stats")
        self.index_version = stats.get("index_version")
        self.model = stats.get("llm_model")

    def _pace(self, kind, gap):
        wait = self._last.get(kind, 0) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[kind] = time.monotonic()

    def _req(self, method, path, body=None, accept="application/json"):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", "Accept": accept})
        return urllib.request.urlopen(req, timeout=120)

    def _json(self, method, path, body=None):
        with self._req(method, path, body) as r:
            return json.loads(r.read())

    def retrieve(self, question):
        self._pace("retrieve", self.retrieve_pace)
        res = self._json("POST", "/api/v1/retrieve", {"query": question, "top_k": 10})
        out = [{"chunk_id": c["chunk_id"], "file_id": c["chunk_id"].split("__")[0],
                "section": c.get("section", ""), "cosine": c.get("cosine")} for c in res["candidates"]]
        return out, res["tier"]

    def answer(self, question):
        self._pace("ask", self.ask_pace)
        t0 = time.perf_counter()
        first, sources, done, meta = None, [], None, None
        with self._req("POST", "/api/v1/ask/stream",
                       {"query": question, "structured": True, "nocache": True},
                       accept="application/x-ndjson") as r:
            for raw in r:
                line = raw.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev["type"] == "token" and first is None:
                    first = time.perf_counter() - t0
                elif ev["type"] == "sources":
                    meta = ev
                    sources = ev["sources"]
                elif ev["type"] == "done":
                    done = ev
                elif ev["type"] == "error":
                    raise RuntimeError(ev.get("message"))
        if done is None:
            raise RuntimeError("stream ended without a done event")
        return {"answer": done["answer"], "grounded": meta["grounded"], "confidence": meta["confidence"],
                "contexts": [s["text"] for s in sources], "cited": done["cited"],
                "first_token_s": first, "usage": []}


# ---------------------------------------------------------------- metrics

def eval_retrieval(target, answerable, unanswerable):
    hits = {1: 0, 3: 0, 5: 0}
    rr, top_ans, top_unans = [], [], []
    section_hits = section_total = refusals_ok = false_refusals = 0
    for item in answerable:
        cands, tier = target.retrieve(item["question"])
        false_refusals += tier == "none"
        top_ans.append(max((c["cosine"] for c in cands if c["cosine"] is not None), default=0.0))
        got = [c["file_id"] for c in cands[:5]]
        expected = set(item["expected_file_ids"])
        for k in hits:
            hits[k] += bool(expected & set(got[:k]))
        rank = next((i + 1 for i, f in enumerate(got) if f in expected), None)
        rr.append(1.0 / rank if rank else 0.0)
        if item.get("expected_section"):
            section_total += 1
            section_hits += item["expected_section"] in [c["section"] for c in cands[:5]]
    for item in unanswerable:
        cands, tier = target.retrieve(item["question"])
        top_unans.append(max((c["cosine"] for c in cands if c["cosine"] is not None), default=0.0))
        refusals_ok += tier == "none"
    n = len(answerable)
    rng = lambda xs: (min(xs), statistics.median(xs), max(xs)) if xs else None  # noqa: E731
    return {"hit@1": hits[1] / n, "hit@3": hits[3] / n, "hit@5": hits[5] / n, "mrr": sum(rr) / n,
            "section_hit@5": section_hits / section_total if section_total else None,
            "refusal_accuracy": refusals_ok / len(unanswerable) if unanswerable else None,
            "false_refusal_rate": false_refusals / n,
            "top_cosine_answerable": rng(top_ans), "top_cosine_unanswerable": rng(top_unans)}


class Judge:
    def __init__(self):
        import os

        from app.services.llm import ChatModel
        # Chat Completions by default: the endpoint LlamaIndex used for the v2 baseline.
        self.llm = ChatModel(settings.JUDGE_MODEL, temperature=0.1, max_output_tokens=256,
                             api=os.getenv("JUDGE_API", "chat"))

    def yes(self, template, query_str, contexts):
        text = template.format(query_str=query_str, context_str="\n\n".join(contexts))
        return "yes" in str(self.llm.complete(text)).lower()

    def faithful(self, answer, contexts):
        return self.yes(FAITHFULNESS_TEMPLATE, answer, contexts)

    def relevant(self, question, answer, contexts):
        return self.yes(RELEVANCY_TEMPLATE, f"Question: {question}\nResponse: {answer}", contexts)

    def honest(self, question, answer):
        verdict = str(self.llm.complete(HONESTY_TEMPLATE.format(question=question, answer=answer)))
        return verdict.strip().upper().startswith("PASS")


def cost_usd(usage, model):
    p_in, p_cached, p_out = PRICES.get(model, (0, 0, 0))
    return sum(((u["input"] - u["cached"]) * p_in + u["cached"] * p_cached + u["output"] * p_out)
               for u in usage) / 1e6


def eval_full(target, answerable, unanswerable, model):
    judge = Judge()
    faith = relev = cite = grounded_n = honest = 0
    per_item, first_tokens, usage = [], [], []
    for item in answerable:
        r = target.answer(item["question"])
        usage += r["usage"]
        if r["first_token_s"] is not None:
            first_tokens.append(r["first_token_s"])
        rec = {"id": item["id"], "grounded": r["grounded"], "confidence": r["confidence"],
               "cited": r["cited"], "first_token_s": r["first_token_s"], "answer": r["answer"]}
        if r["grounded"] and r["contexts"]:
            grounded_n += 1
            cite += bool(r["cited"])
            rec["faithful"] = judge.faithful(r["answer"], r["contexts"])
            rec["relevant"] = judge.relevant(item["question"], r["answer"], r["contexts"])
            faith += rec["faithful"]
            relev += rec["relevant"]
        per_item.append(rec)
        print(f"  {item['id']}: grounded={r['grounded']} conf={r['confidence']} cited={r['cited']} "
              f"faithful={rec.get('faithful')} relevant={rec.get('relevant')} "
              f"first_token={r['first_token_s'] and round(r['first_token_s'], 2)}s")
    for item in unanswerable:
        r = target.answer(item["question"])
        usage += r["usage"]
        ok = True if not r["grounded"] else judge.honest(item["question"], r["answer"])
        honest += ok
        per_item.append({"id": item["id"], "grounded": r["grounded"], "honest": ok})
        print(f"  {item['id']}: grounded={r['grounded']} honest={ok}")
    n_q = len(answerable) + len(unanswerable)
    return {
        "faithfulness": faith / grounded_n if grounded_n else None,
        "relevancy": relev / grounded_n if grounded_n else None,
        "citation_coverage": cite / grounded_n if grounded_n else None,
        "grounded_rate_answerable": grounded_n / len(answerable),
        "honest_refusal_rate": honest / len(unanswerable) if unanswerable else None,
        "first_token_p50_s": statistics.median(first_tokens) if first_tokens else None,
        "first_token_p90_s": (sorted(first_tokens)[int(0.9 * (len(first_tokens) - 1))]
                              if first_tokens else None),
        "tokens_per_question": ({k: round(sum(u[k] for u in usage) / n_q)
                                 for k in ("input", "cached", "output", "reasoning")} if usage else None),
        "cost_per_question_usd": cost_usd(usage, model) / n_q if usage else None,
        "per_item": per_item,
    }


# ------------------------------------------------------------------ report

def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def write_report(out_path, target, model, retrieval, compare_rows, full, n_ans, n_unans):
    lines = [
        "# DEBBIE RAG — Evaluation Results", "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} · target: {target.name}"
        + (f" `{target.base}`" if target.name == "http" else "")
        + f" · golden set: {n_ans} answerable + {n_unans} unanswerable · embed "
          f"`{settings.OPENAI_EMBED_MODEL}` · answer model `{model}` · judge `{settings.JUDGE_MODEL}` · "
          f"index `{target.index_version}`_", "",
        "## Retrieval", "", "| Metric | Value |", "|---|---|",
        f"| hit@1 (file) | {pct(retrieval['hit@1'])} |",
        f"| hit@3 (file) | {pct(retrieval['hit@3'])} |",
        f"| hit@5 (file) | {pct(retrieval['hit@5'])} |",
        f"| MRR | {retrieval['mrr']:.3f} |",
        f"| section-hit@5 | {pct(retrieval['section_hit@5'])} |",
        f"| gate hard-refusal on unanswerable (tier=none) | {pct(retrieval['refusal_accuracy'])} |",
        f"| false-refusal rate (answerable → none) | {pct(retrieval['false_refusal_rate'])} |", "",
    ]
    if compare_rows:
        lines += ["## Configuration comparison", "", "| Config | hit@5 | MRR | false-refusal |",
                  "|---|---|---|---|"]
        lines += [f"| {n} | {pct(m['hit@5'])} | {m['mrr']:.3f} | {pct(m['false_refusal_rate'])} |"
                  for n, m in compare_rows] + [""]
    if full:
        lines += ["## Answer quality (LLM-as-judge)", "", "| Metric | Value |", "|---|---|",
                  f"| Faithfulness (grounded answers) | {pct(full['faithfulness'])} |",
                  f"| Relevancy (grounded answers) | {pct(full['relevancy'])} |",
                  f"| Citation coverage (≥1 inline citation) | {pct(full['citation_coverage'])} |",
                  f"| Grounded rate on answerable | {pct(full['grounded_rate_answerable'])} |",
                  f"| Answer-level honesty on unanswerable | {pct(full['honest_refusal_rate'])} |",
                  f"| First token p50 / p90 | {full['first_token_p50_s']:.2f} s / {full['first_token_p90_s']:.2f} s |"]
        if full["cost_per_question_usd"] is not None:
            t = full["tokens_per_question"]
            lines.append(f"| Tokens / question (in · cached · out · reasoning) | {t['input']} · "
                         f"{t['cached']} · {t['output']} · {t['reasoning']} |")
            lines.append(f"| Cost / question (answer + condense) | ${full['cost_per_question_usd']:.5f} |")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--compare", action="store_true", help="also vector-only retrieval (in-process)")
    ap.add_argument("--base-url", help="evaluate a deployment over HTTP instead of in-process")
    ap.add_argument("--out", default=str(SERVER_ROOT / "docs" / "eval_latest.md"))
    args = ap.parse_args()

    answerable, unanswerable = load_golden()
    print(f"golden set: {len(answerable)} answerable, {len(unanswerable)} unanswerable")
    target = RemoteTarget(args.base_url) if args.base_url else LocalTarget()
    model = getattr(target, "model", None) or settings.OPENAI_LLM_MODEL
    print(f"target={target.name} model={model} judge={settings.JUDGE_MODEL} index={target.index_version}")

    print("\n== retrieval (hybrid) ==")
    retrieval = eval_retrieval(target, answerable, unanswerable)
    for k, v in retrieval.items():
        print(f"  {k}: {v}")

    compare_rows = []
    if args.compare and isinstance(target, LocalTarget):
        compare_rows.append(("hybrid (vector + BM25, RRF)", retrieval))
        bm25, target.pipeline._bm25 = target.pipeline._bm25, None
        compare_rows.append(("vector-only", eval_retrieval(target, answerable, unanswerable)))
        target.pipeline._bm25 = bm25

    full = None
    if args.mode == "full":
        print("\n== full answer-quality eval ==")
        full = eval_full(target, answerable, unanswerable, model)
        for k, v in full.items():
            if k != "per_item":
                print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"run_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"target": target.name, "base_url": getattr(target, "base", None), "model": model,
                   "judge": settings.JUDGE_MODEL, "index_version": target.index_version,
                   "retrieval": retrieval, "compare": [{"name": n, **m} for n, m in compare_rows],
                   "full": full}, f, indent=2, default=str)
    write_report(Path(args.out), target, model, retrieval, compare_rows, full,
                 len(answerable), len(unanswerable))


if __name__ == "__main__":
    main()
