"""RAG service: owns the knowledge-base index and the answer pipeline.

The index is loaded at startup (cold start) from S3 or a local directory, and a
warm instance re-checks the `current.json` pointer at most every
INDEX_REFRESH_SECONDS. A new version gets a fresh pipeline (BM25 corpus + query
cache); a broken version is refused and the previous one keeps serving.
"""

import logging
import threading
from typing import Dict, List, Optional

from app.core.config import settings
from app.services import chunking, llm
from app.services.index_store import IndexStore, from_settings
from app.services.rag_pipeline import AnswerPipeline, AnswerResult

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, store: Optional[IndexStore] = None, llm_factory=None,
                 embed_fn=None, condense_llm=None):
        self._store = store
        self._llm_factory = llm_factory or llm.answer_model
        self._embed_fn = embed_fn or llm.embed_query
        self._condense_llm = condense_llm
        self.pipeline: Optional[AnswerPipeline] = None
        self._lock = threading.Lock()

    @property
    def store(self) -> IndexStore:
        if self._store is None:
            self._store = from_settings(settings)
        return self._store

    # ----------------------------------------------------------- lifecycle

    def startup(self) -> None:
        with self._lock:
            if self.store.load() or self.pipeline is None:
                self._rebuild_pipeline()

    def refresh(self) -> None:
        """Cheap check (rate-limited inside the store); swaps in a new version."""
        if self.pipeline is None:
            self.startup()
            return
        with self._lock:
            if self.store.maybe_refresh():
                self._rebuild_pipeline()

    def _rebuild_pipeline(self) -> None:
        if not self.store.nodes:
            self.pipeline = None
            return
        self.pipeline = AnswerPipeline(
            llm_factory=self._llm_factory,
            embed_fn=self._embed_fn,
            chroma_query=self.store.query,
            nodes=self.store.nodes,
            condense_llm=self._condense_llm or llm.condense_model(),
            kb_version=self.store.version or "",
        )
        logger.info("Answer pipeline ready (%d chunks, index %s)",
                    len(self.store.nodes), self.store.version)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None and bool(self.pipeline.nodes)

    # ----------------------------------------------------------- answering

    def query(self, question: str, history: Optional[List[Dict[str, str]]] = None,
              top_k: Optional[int] = None, temperature: Optional[float] = None,
              nocache: bool = False) -> AnswerResult:
        return self.pipeline.run(question, history, top_k, temperature, nocache)

    def query_stream(self, question: str, history: Optional[List[Dict[str, str]]] = None,
                     top_k: Optional[int] = None, temperature: Optional[float] = None,
                     nocache: bool = False):
        """Yields ("meta"|"sources"|"token"|"done"|"error", payload) events."""
        yield from self.pipeline.run_stream(question, history, top_k, temperature, nocache)

    def retrieve_debug(self, query: str, top_k: Optional[int] = None) -> dict:
        """Retrieval inspector for the RAG Playground: hybrid candidates with
        per-leg scores and the gate decision."""
        candidates = self.pipeline.retrieve(query)
        tier = self.pipeline.gate(candidates)
        final_top_k = top_k or settings.TOP_K_RETRIEVAL
        out = []
        for rank, c in enumerate(candidates[: max(final_top_k * 2, 10)], 1):
            node = self.pipeline.nodes.get(c.node_id)
            if node is None:
                continue
            md = node.metadata or {}
            out.append({
                "final_rank": rank,
                "chunk_id": c.node_id,
                "title": md.get("title") or md.get("source", "Unknown"),
                "section": md.get("section", ""),
                "text": node.text,
                "cosine": round(c.cosine, 4) if c.cosine is not None else None,
                "bm25_rank": c.bm25_rank,
                "rrf": round(c.rrf, 5),
            })
        return {"query": query, "tier": tier,
                "thresholds": {"strong": settings.CONFIDENCE_STRONG, "weak": settings.CONFIDENCE_WEAK},
                "candidates": out}

    def get_index_stats(self) -> dict:
        m = self.store.manifest if self._store is not None else {}
        count = len(self.store.nodes) if self._store is not None else 0
        return {
            "status": "ready" if count else "empty",
            "document_count": count,
            "index_version": self.store.version if self._store is not None else None,
            "index_created": m.get("created"),
            "collection_name": self.store.version if self._store is not None else None,
            "chunk_size": chunking.CHUNK_TOKENS,
            "chunk_overlap": chunking.CHUNK_OVERLAP_TOKENS,
            "top_k_retrieval": settings.TOP_K_RETRIEVAL,
            "embedding_model": m.get("embed_model", settings.OPENAI_EMBED_MODEL),
            "llm_model": settings.OPENAI_LLM_MODEL,
            "kb_version": m.get("kb_version"),
            "hybrid_search": bool(self.pipeline and self.pipeline._bm25 is not None),
            "rerank_mode": "off",
            "documents": [{"title": d.get("title"), "chunk_count": d.get("chunk_count"),
                           "sections": d.get("sections")} for d in m.get("documents", [])],
        }


rag_service = RAGService()
