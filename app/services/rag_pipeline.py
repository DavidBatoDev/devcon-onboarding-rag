"""Unified answer pipeline (v2): one code path for streaming AND non-streaming.

Stages: condense -> hybrid retrieve (vector + BM25, RRF fusion with raw cosine
preserved) -> confidence gate -> single prompt -> stream synthesis ->
deterministic citation sanitation.

Design notes:
- The vector leg is an exact cosine search over the in-memory index
  (index_store.IndexStore: ids + cosine distances), so the confidence gate
  always sees TRUE cosine similarities. RRF scores are rank-only and are never
  used for gating (fusing scores would destroy the gate's semantics).
- The LLM rerank stage (LlamaIndex LLMRerank, off in production) was dropped
  with LlamaIndex; the eval showed hybrid retrieval alone meets the bar.
- Grounding is decided deterministically by the gate BEFORE synthesis; the
  LLM is never trusted to self-report whether it used the documents.
"""

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Generator, List, Optional, Tuple

from cachetools import TTLCache

from app.core.config import settings
from app.services import prompts

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
_CITE_RE = re.compile(r"\[(\d{1,2})\]")
SNIPPET_CHARS = 280


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def rrf_fuse(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal-rank fusion over id rankings. Rank-only: input scores are
    intentionally ignored so incomparable score scales can't skew the fusion."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, 1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


def sanitize_citations(text: str, k: int) -> Tuple[str, List[int]]:
    """Keep [n] markers with 1 <= n <= k, strip hallucinated ones.
    Returns (clean_text, sorted list of cited source numbers)."""
    cited = set()

    def repl(m):
        n = int(m.group(1))
        if 1 <= n <= k:
            cited.add(n)
            return m.group(0)
        return ""

    clean = _CITE_RE.sub(repl, text)
    # Defensive: strip any legacy marker a fine-tuned habit might emit.
    clean = clean.replace("SHOW_REFERENCES: true", "").replace("SHOW_REFERENCES: false", "")
    return clean.strip(), sorted(cited)


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind(" ")] + "…" if " " in cut else cut + "…"


@dataclass
class SourceRef:
    n: int
    title: str
    section: str
    snippet: str
    text: str
    drive_link: str
    score: Optional[float]      # true cosine similarity (None if BM25-only hit)
    chunk_id: str
    cited: bool = False


@dataclass
class Candidate:
    node_id: str
    cosine: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf: float = 0.0


@dataclass
class AnswerResult:
    answer: str
    sources: List[SourceRef] = field(default_factory=list)
    grounded: bool = False
    confidence: str = "none"                 # strong | weak | none
    timings: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    request_id: str = ""
    kb_version: str = ""
    cached: bool = False

    def metadata_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "confidence": self.confidence,
            "sources": [asdict(s) for s in self.sources],
            "timings": self.timings,
            "model": self.model,
            "request_id": self.request_id,
            "kb_version": self.kb_version,
            "cached": self.cached,
        }


class AnswerPipeline:
    """Everything injected for testability: `llm_factory(temperature)` returns
    an object with .complete/.stream_complete; `embed_fn(str)` returns a query
    embedding; `chroma_query(embedding, n)` returns (ids, cosine distances);
    `nodes` is {node_id: node} with .text/.metadata; `condense_llm` (optional)
    rewrites follow-ups; `kb_version` labels answers with the index version."""

    def __init__(self, llm_factory, embed_fn, chroma_query, nodes: Dict[str, object],
                 condense_llm=None, kb_version: str = ""):
        self.llm_factory = llm_factory
        self.embed_fn = embed_fn
        self.chroma_query = chroma_query
        self.nodes = nodes
        self.condense_llm = condense_llm
        self.kb_version = kb_version
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._build_bm25()
        self._cache = TTLCache(maxsize=settings.QUERY_CACHE_SIZE, ttl=settings.QUERY_CACHE_TTL)
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------ BM25 corpus

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 leg disabled, vector-only retrieval")
            return
        if not self.nodes:
            logger.warning("No nodes loaded — BM25 leg disabled")
            return
        self._bm25_ids = list(self.nodes.keys())
        corpus = [tokenize(self.nodes[i].text) for i in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 corpus built over %d chunks", len(self._bm25_ids))

    # -------------------------------------------------------------- retrieval

    def _condense(self, question: str, history: List[Dict[str, str]]) -> str:
        if not history or not settings.ENABLE_QUERY_CONDENSE:
            return question
        try:
            convo = prompts.format_history(history, settings.CONDENSE_HISTORY_TURNS)
            llm = self.condense_llm or self.llm_factory(None)
            condensed = str(llm.complete(prompts.build_condense_prompt(question, convo))).strip()
            if condensed and len(condensed) <= 1000:
                logger.info("Condensed query: %s", condensed)
                return condensed
        except Exception as e:
            logger.warning("Query condensation failed, using original question: %s", e)
        return question

    def retrieve(self, query: str, candidates: Optional[int] = None) -> List[Candidate]:
        """Hybrid retrieval with preserved raw cosine scores."""
        n = candidates or settings.RETRIEVE_CANDIDATES
        # Vector leg (true cosine via direct Chroma query)
        emb = self.embed_fn(query)
        ids, distances = self.chroma_query(emb, n)
        by_id: Dict[str, Candidate] = {}
        vector_ranking = []
        for node_id, dist in zip(ids, distances):
            cos = 1.0 - dist
            if cos < settings.SIMILARITY_CUTOFF:
                continue  # pre-fusion floor applies to the vector leg only
            by_id[node_id] = Candidate(node_id=node_id, cosine=cos)
            vector_ranking.append(node_id)

        # BM25 leg
        bm25_ranking: List[str] = []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokenize(query))
            ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: -x[1])
            bm25_ranking = [i for i, s in ranked[: settings.BM25_TOP_K] if s > 0]
            for rank, node_id in enumerate(bm25_ranking, 1):
                cand = by_id.setdefault(node_id, Candidate(node_id=node_id))
                cand.bm25_rank = rank

        fused = rrf_fuse([r for r in (vector_ranking, bm25_ranking) if r])
        for node_id, score in fused.items():
            if node_id in by_id:
                by_id[node_id].rrf = score
        out = sorted(by_id.values(), key=lambda c: -c.rrf)
        return out

    def gate(self, candidates: List[Candidate]) -> str:
        """Confidence tier from the best TRUE cosine similarity."""
        cosines = [c.cosine for c in candidates if c.cosine is not None]
        top = max(cosines, default=0.0)
        if top >= settings.CONFIDENCE_STRONG:
            return "strong"
        if top >= settings.CONFIDENCE_WEAK:
            return "weak"
        return "none"

    def build_sources(self, candidates: List[Candidate]) -> List[SourceRef]:
        refs = []
        for i, c in enumerate(candidates, 1):
            node = self.nodes.get(c.node_id)
            if node is None:
                continue
            md = node.metadata or {}
            refs.append(SourceRef(
                n=i,
                title=md.get("title") or md.get("source", "Unknown"),
                section=md.get("section", "") or "",
                snippet=_snippet(node.text),
                text=node.text,
                drive_link=md.get("drive_link", ""),
                score=round(c.cosine, 4) if c.cosine is not None else None,
                chunk_id=c.node_id,
            ))
        return refs

    # -------------------------------------------------------------- answering

    def run_stream(self, question: str, history: Optional[List[Dict[str, str]]] = None,
                   top_k: Optional[int] = None, temperature: Optional[float] = None,
                   nocache: bool = False) -> Generator[Tuple[str, object], None, None]:
        """Yields typed events: ("meta", dict), ("sources", dict), ("token", str),
        ("done", AnswerResult), ("error", str). The single answer path."""
        history = history or []
        key = self._cache_key(question, history, top_k, temperature)

        if settings.ENABLE_QUERY_CACHE and not nocache:
            with self._cache_lock:
                hit: Optional[AnswerResult] = self._cache.get(key)
            if hit is not None:
                logger.info("Query cache hit")
                cached = AnswerResult(**{**asdict(hit), "cached": True})
                cached.sources = [SourceRef(**s) if isinstance(s, dict) else s
                                  for s in cached.sources]
                yield "meta", {"request_id": cached.request_id, "model": cached.model,
                               "kb_version": cached.kb_version, "cached": True}
                yield "sources", {"grounded": cached.grounded, "confidence": cached.confidence,
                                  "sources": [asdict(s) for s in cached.sources]}
                yield "token", cached.answer
                yield "done", cached
                return

        t0 = time.perf_counter()
        timings: Dict[str, int] = {}
        result = AnswerResult(
            answer="",
            model=settings.OPENAI_LLM_MODEL,
            request_id=uuid.uuid4().hex[:12],
            kb_version=self.kb_version,
        )
        yield "meta", {"request_id": result.request_id, "model": result.model,
                       "kb_version": result.kb_version, "cached": False}

        try:
            # 1. condense
            t = time.perf_counter()
            retrieval_query = self._condense(question, history)
            timings["condense_ms"] = int((time.perf_counter() - t) * 1000)

            # 2. retrieve + gate
            t = time.perf_counter()
            candidates = self.retrieve(retrieval_query)
            tier = self.gate(candidates)
            timings["retrieval_ms"] = int((time.perf_counter() - t) * 1000)

            # 3. sources
            final_top_k = top_k or settings.TOP_K_RETRIEVAL
            history_block = prompts.format_history(history)
            llm = self.llm_factory(temperature)

            if tier == "none":
                result.grounded, result.confidence = False, "none"
                yield "sources", {"grounded": False, "confidence": "none", "sources": []}
                prompt = prompts.build_general_prompt(question, history_block)
            else:
                result.sources = self.build_sources(candidates[:final_top_k])
                result.grounded, result.confidence = True, tier
                yield "sources", {"grounded": True, "confidence": tier,
                                  "sources": [asdict(s) for s in result.sources]}
                prompt = prompts.build_answer_prompt(
                    question,
                    [{"title": s.title, "section": s.section, "text": s.text}
                     for s in result.sources],
                    history_block=history_block,
                    weak=(tier == "weak"),
                )

            # 4. synthesize (single streaming path for both modes)
            t = time.perf_counter()
            raw_parts: List[str] = []
            for chunk in llm.stream_complete(prompt):
                if chunk.delta:
                    raw_parts.append(chunk.delta)
                    yield "token", chunk.delta
            timings["synthesis_ms"] = int((time.perf_counter() - t) * 1000)

            # 5. sanitize citations deterministically
            raw = "".join(raw_parts)
            clean, cited = sanitize_citations(raw, len(result.sources))
            for s in result.sources:
                s.cited = s.n in cited
            result.answer = clean or "I wasn't able to generate an answer — please try rephrasing."
            timings["total_ms"] = int((time.perf_counter() - t0) * 1000)
            result.timings = timings

            logger.info(json.dumps({
                "event": "answer", "request_id": result.request_id, "tier": result.confidence,
                "grounded": result.grounded, "sources": len(result.sources),
                "cited": cited, "timings": timings, "cached": False,
                "model": result.model, "kb_version": result.kb_version,
            }))

            if settings.ENABLE_QUERY_CACHE and result.answer:
                with self._cache_lock:
                    self._cache[key] = result
            yield "done", result

        except Exception as e:
            logger.exception("Pipeline error: %s", e)
            yield "error", "I encountered an error while processing your question. Please try again."

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None,
            top_k: Optional[int] = None, temperature: Optional[float] = None,
            nocache: bool = False) -> AnswerResult:
        """Non-streaming = drained stream (identical answer path)."""
        result: Optional[AnswerResult] = None
        error: Optional[str] = None
        for kind, payload in self.run_stream(question, history, top_k, temperature, nocache):
            if kind == "done":
                result = payload
            elif kind == "error":
                error = payload
        if result is None:
            return AnswerResult(answer=error or "Something went wrong. Please try again.",
                                model=settings.OPENAI_LLM_MODEL,
                                request_id=uuid.uuid4().hex[:12],
                                kb_version=self.kb_version)
        return result

    # ------------------------------------------------------------------ cache

    def _cache_key(self, question, history, top_k, temperature) -> str:
        hist = json.dumps(history or [], sort_keys=True, default=str)
        raw = (f"{settings.PIPELINE_VERSION}|{self.kb_version}|{settings.OPENAI_LLM_MODEL}|"
               f"{(question or '').strip().lower()}|{hist}|{top_k}|{temperature}")
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def invalidate_cache(self):
        with self._cache_lock:
            self._cache.clear()
        logger.info("Query cache cleared")


def legacy_answer_text(result: AnswerResult) -> str:
    """Backwards-compatible plain-markdown rendering for old clients: the
    answer plus the classic sources footer (no Beta footer — removed in v2)."""
    if not result.sources:
        return result.answer
    seen, lines = set(), []
    for s in result.sources:
        key = (s.title, s.drive_link)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{len(seen)}. {s.title}" + (f" — {s.section}" if s.section else ""))
        if s.drive_link:
            lines.append(f"   [[View Document]]({s.drive_link})")
        if len(seen) >= 3:
            break
    return f"{result.answer}\n\n📚 **Sources Referenced:**\n" + "\n".join(lines)
