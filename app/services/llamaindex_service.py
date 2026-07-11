# backend/app/services/llamaindex_service.py
"""RAG service bootstrap (v2).

This module owns infrastructure: OpenAI LLM/embeddings config, the ChromaDB
collection, the in-memory node corpus, and the (local-only, admin-gated)
Google Drive rebuild paths. All ANSWERING logic lives in
app/services/rag_pipeline.py — one unified pipeline for streaming and
non-streaming, with hybrid retrieval, a confidence gate, and inline citations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Settings as LlamaSettings
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.core.config import settings
from app.services import chunking
from app.services.rag_pipeline import AnswerPipeline, AnswerResult

logger = logging.getLogger(__name__)


class LlamaIndexRAGService:
    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self.pipeline: Optional[AnswerPipeline] = None
        self._setup_llama_index()
        self._initialize_components()
        self.index_state_file = Path(settings.RAG_INDEX_DIR) / "index_state.json"
        self._load_index_state()
        self._ensure_directories_exist()

    # ------------------------------------------------------------- bootstrap

    def _setup_llama_index(self):
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set — LLM and embedding calls will fail.")
        LlamaSettings.llm = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.MAX_OUTPUT_TOKENS,
            timeout=settings.OPENAI_REQUEST_TIMEOUT,
            max_retries=settings.OPENAI_MAX_RETRIES,
        )
        LlamaSettings.embed_model = OpenAIEmbedding(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_EMBED_MODEL,
            embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
            timeout=settings.OPENAI_REQUEST_TIMEOUT,
            max_retries=settings.OPENAI_MAX_RETRIES,
        )
        logger.info("Initialized OpenAI LLM (%s) and embeddings (%s)",
                    settings.OPENAI_LLM_MODEL, settings.OPENAI_EMBED_MODEL)

    def _initialize_components(self):
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        name = settings.CHROMA_COLLECTION_NAME
        try:
            self.chroma_collection = self.chroma_client.get_collection(name)
            logger.info("Loaded ChromaDB collection: %s (%d chunks)",
                        name, self.chroma_collection.count())
        except Exception:
            self.chroma_collection = self.chroma_client.create_collection(
                name, metadata={"hnsw:space": "cosine"}, embedding_function=None)
            logger.info("Created new ChromaDB collection: %s (cosine)", name)

        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        # The VectorStoreIndex is only needed by the (local) ingestion paths;
        # answering queries Chroma directly through the pipeline.
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex([], storage_context=storage_context)
        self._init_pipeline()

    def _load_nodes(self) -> Dict[str, TextNode]:
        """Load the full chunk corpus into memory (135 chunks — trivial). Feeds
        the BM25 leg, the vector-leg id->node mapping, and the playground."""
        data = self.chroma_collection.get(include=["documents", "metadatas"])
        nodes: Dict[str, TextNode] = {}
        for node_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            clean_meta = {k: v for k, v in (meta or {}).items()
                          if not k.startswith("_node")}
            nodes[node_id] = TextNode(id_=node_id, text=text or "", metadata=clean_meta)
        return nodes

    def _init_pipeline(self):
        nodes = self._load_nodes()

        def llm_factory(temperature: Optional[float]):
            if temperature is None:
                return LlamaSettings.llm
            return OpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_LLM_MODEL,
                temperature=temperature,
                max_tokens=settings.MAX_OUTPUT_TOKENS,
                timeout=settings.OPENAI_REQUEST_TIMEOUT,
                max_retries=settings.OPENAI_MAX_RETRIES,
            )

        def chroma_query(embedding, n):
            count = self.chroma_collection.count()
            if count == 0:
                return [], []
            res = self.chroma_collection.query(
                query_embeddings=[embedding],
                n_results=min(n, count),
                include=["distances"],
            )
            return res["ids"][0], res["distances"][0]

        self.pipeline = AnswerPipeline(
            llm_factory=llm_factory,
            embed_fn=lambda q: LlamaSettings.embed_model.get_query_embedding(q),
            chroma_query=chroma_query,
            nodes=nodes,
        )
        logger.info("Answer pipeline ready (%d chunks in corpus)", len(nodes))

    @property
    def ready(self) -> bool:
        return self.pipeline is not None and bool(self.pipeline.nodes)

    # ------------------------------------------------------------- answering

    def query(self, question: str, history: Optional[List[Dict[str, str]]] = None,
              top_k: Optional[int] = None, temperature: Optional[float] = None,
              nocache: bool = False) -> AnswerResult:
        return self.pipeline.run(question, history, top_k, temperature, nocache)

    def query_stream(self, question: str, history: Optional[List[Dict[str, str]]] = None,
                     top_k: Optional[int] = None, temperature: Optional[float] = None,
                     nocache: bool = False):
        """Yields ("meta"|"sources"|"token"|"done"|"error", payload) events."""
        yield from self.pipeline.run_stream(question, history, top_k, temperature, nocache)

    def retrieve_debug(self, query: str, top_k: Optional[int] = None,
                       use_rerank: bool = False) -> dict:
        """Retrieval inspector for the RAG Playground: shows the hybrid pipeline's
        candidates with per-leg scores and the gate decision."""
        condensed = query  # playground sends standalone queries; no history
        candidates = self.pipeline.retrieve(condensed)
        tier = self.pipeline.gate(candidates)
        final_top_k = top_k or settings.TOP_K_RETRIEVAL
        shown = candidates[: max(final_top_k * 2, 10)]
        out = []
        for rank, c in enumerate(shown, 1):
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
        return {
            "query": query,
            "tier": tier,
            "thresholds": {"strong": settings.CONFIDENCE_STRONG,
                           "weak": settings.CONFIDENCE_WEAK},
            "candidates": out,
        }

    # ------------------------------------------------------------ index state

    def _load_index_state(self):
        self.index_state = {}
        if self.index_state_file.exists():
            try:
                with open(self.index_state_file, "r") as f:
                    self.index_state = json.load(f)
                logger.info("Loaded index state with %d files", len(self.index_state))
            except Exception as e:
                logger.error("Error loading index state: %s", e)

    def save_index_state(self):
        try:
            self.index_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_state_file, "w") as f:
                json.dump(self.index_state, f, indent=2, default=str)
        except Exception as e:
            logger.error("Error saving index state: %s", e)

    # ----------------------------------------------- ingestion (local-only)
    # These paths need Google Drive credentials and are admin-gated in the API.
    # They produce v2-format chunks via the shared section-aware chunker, so a
    # local rebuild stays compatible with the deployed pipeline.

    def _documents_to_nodes(self, documents: List[Document]) -> List[TextNode]:
        nodes: List[TextNode] = []
        for doc in documents:
            md = doc.metadata or {}
            file_id = md.get("file_id", doc.id_ or "unknown")
            title = re.sub(r"\.docx$", "", md.get("title") or md.get("source", "Unknown")).strip()
            pieces = chunking.chunk_document_text(doc.text or "")
            for idx, (section, text) in enumerate(pieces):
                meta = {
                    "source": md.get("source", ""),
                    "title": title,
                    "file_id": file_id,
                    "drive_link": md.get("drive_link", ""),
                    "modified_time": str(md.get("modified_time", "")),
                    "section": section,
                    "chunk_index": idx,
                    "kb_version": settings.KB_VERSION,
                }
                node = TextNode(id_=f"{file_id}__{idx:03d}", text=text, metadata=meta)
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=file_id)
                node.excluded_llm_metadata_keys = list(meta.keys())
                node.excluded_embed_metadata_keys = list(meta.keys())
                nodes.append(node)
        return nodes

    def add_documents(self, documents: List[Document]) -> bool:
        try:
            nodes = self._documents_to_nodes(documents)
            logger.info("Inserting %d v2 chunks from %d documents", len(nodes), len(documents))
            self.index.insert_nodes(nodes)  # auto-embeds via the configured model
            for doc in documents:
                file_id = doc.metadata.get("file_id")
                if file_id:
                    self.index_state[file_id] = {
                        "file_id": file_id,
                        "file_name": doc.metadata.get("source", "Unknown"),
                        "title": doc.metadata.get("title", "Unknown"),
                        "modified_time": doc.metadata.get("modified_time"),
                        "chunk_count": sum(1 for n in nodes
                                           if n.metadata.get("file_id") == file_id),
                    }
            self.save_index_state()
            self._init_pipeline()  # refresh node corpus + BM25 + cache
            return True
        except Exception as e:
            logger.exception("Error adding documents: %s", e)
            return False

    def remove_document(self, file_id: str) -> bool:
        try:
            self.chroma_collection.delete(where={"file_id": file_id})
            self.index_state.pop(file_id, None)
            logger.info("Removed document %s from index", file_id)
            return True
        except Exception as e:
            logger.error("Error removing document %s: %s", file_id, e)
            return False

    def rebuild_index_from_scratch(self, documents: List[Document]) -> bool:
        try:
            logger.info("Rebuilding collection %s from scratch (%d documents)",
                        settings.CHROMA_COLLECTION_NAME, len(documents))
            if not self.clear_index():
                return False
            ok = self.add_documents(documents)
            if ok:
                logger.info("Rebuild complete")
            return ok
        except Exception as e:
            logger.exception("Error in rebuild: %s", e)
            return False

    def incremental_update(self, drive_files: List[Dict[str, Any]]) -> dict:
        """Sync with Drive: re-embed new/changed files, drop deleted ones."""
        from app.services.document_processor import document_processor

        summary = {"added": 0, "changed": 0, "skipped": 0, "deleted": 0, "failed": 0,
                   "added_files": [], "changed_files": [], "deleted_files": []}
        try:
            current_ids = set()
            to_process = []
            for f in drive_files:
                file_id = f.get("id")
                if not file_id:
                    continue
                current_ids.add(file_id)
                state = self.index_state.get(file_id)
                if state is None:
                    summary["added"] += 1
                    summary["added_files"].append(f.get("name", file_id))
                    to_process.append(f)
                elif str(state.get("modified_time")) != str(f.get("modifiedTime", "Unknown")):
                    summary["changed"] += 1
                    summary["changed_files"].append(f.get("name", file_id))
                    to_process.append(f)
                else:
                    summary["skipped"] += 1

            deleted_ids = [fid for fid in list(self.index_state.keys())
                           if fid not in current_ids]
            for fid in deleted_ids:
                name = self.index_state.get(fid, {}).get("file_name", fid)
                if self.remove_document(fid):
                    summary["deleted"] += 1
                    summary["deleted_files"].append(name)

            for f in to_process:
                if self.index_state.get(f.get("id")):
                    self.remove_document(f.get("id"))

            if to_process:
                documents = document_processor.process_drive_files(to_process)
                summary["failed"] = len(to_process) - len(documents)
                if documents and not self.add_documents(documents):
                    summary["error"] = "add_documents failed"
            elif deleted_ids:
                self.save_index_state()
                self._init_pipeline()

            logger.info("Incremental update: %s", summary)
            return summary
        except Exception as e:
            logger.exception("Error during incremental update: %s", e)
            summary["error"] = str(e)
            return summary

    def clear_index(self) -> bool:
        """Delete and recreate ONLY the configured collection. Never touches
        other collections in the same store (the legacy collection is kept as
        a rollback lever)."""
        try:
            name = settings.CHROMA_COLLECTION_NAME
            try:
                self.chroma_client.delete_collection(name)
                logger.info("Deleted ChromaDB collection: %s", name)
            except Exception as e:
                logger.info("Could not delete collection %s: %s", name, e)
            self.chroma_collection = self.chroma_client.create_collection(
                name, metadata={"hnsw:space": "cosine"}, embedding_function=None)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex([], storage_context=storage_context)
            self.index_state = {}
            self.save_index_state()
            self._init_pipeline()
            logger.info("Collection %s cleared and reset", name)
            return True
        except Exception as e:
            logger.exception("Error clearing index: %s", e)
            return False

    # ------------------------------------------------------------------ misc

    def get_index_stats(self) -> dict:
        try:
            count = self.chroma_collection.count()
            stats = {
                "status": "ready" if count else "empty",
                "document_count": count,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "chunk_size": chunking.CHUNK_TOKENS,
                "chunk_overlap": chunking.CHUNK_OVERLAP_TOKENS,
                "top_k_retrieval": settings.TOP_K_RETRIEVAL,
                "embedding_model": settings.OPENAI_EMBED_MODEL,
                "embedding_dimensions": settings.EMBEDDING_DIM,
                "llm_model": settings.OPENAI_LLM_MODEL,
                "kb_version": settings.KB_VERSION,
                "hybrid_search": self.pipeline._bm25 is not None if self.pipeline else False,
                "rerank_mode": settings.RERANK_MODE,
            }
            manifest_path = Path(settings.CHROMA_PERSIST_DIR).parent / "corpus" / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                stats["documents"] = [
                    {"title": d["title"], "chunk_count": d["chunk_count"],
                     "sections": len(d.get("sections", []))}
                    for d in manifest.get("documents", [])
                ]
            return stats
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _ensure_directories_exist(self):
        try:
            Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            Path(settings.RAG_INDEX_DIR).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Error ensuring directories exist: %s", e)


# Global instance
rag_service = LlamaIndexRAGService()
