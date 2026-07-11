"""Retrieval score probe: legacy (l2) vs v2 (cosine) collection, side by side.

Used to calibrate CONFIDENCE_STRONG / CONFIDENCE_WEAK from the real score
distribution before wiring the confidence gate. Read-only.

Usage (from server/):
  venv/Scripts/python scripts/inspect_retrieval.py
"""

import sys
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from app.core.config import settings  # noqa: E402

# Answerable queries (docs cover these) vs out-of-scope queries (docs don't).
ANSWERABLE = [
    "How do I apply for the DEVCON internship program?",
    "What should a new chapter do in its first 6 months?",
    "What are the requirements for starting a new DEVCON chapter?",
    "What is the strategy canvas for the Bohol chapter?",
    "How should chapters handle receipts and finances?",
    "What is the DEVCON brand kit link?",
    "What happens at the leaders oath taking ceremony?",
    "How do I plan and organize a chapter event safely?",
    "What are the 2025 DEVCON programs and calendar?",
    "What insights came from volunteer applications?",
]
OUT_OF_SCOPE = [
    "What is the weather in Manila today?",
    "How do I file Philippine income taxes?",
    "What is the best programming language to learn in 2026?",
    "Who won the last Philippine election?",
    "How do I set up a Kubernetes cluster?",
]


def probe(collection_name: str, space: str, queries, embed_model):
    import chromadb

    client = chromadb.PersistentClient(path=str(SERVER_ROOT / settings.CHROMA_PERSIST_DIR))
    col = client.get_collection(collection_name)
    rows = []
    for q in queries:
        emb = embed_model.get_query_embedding(q)
        res = col.query(query_embeddings=[emb], n_results=5,
                        include=["metadatas", "distances"])
        hits = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            # cosine space: similarity = 1 - distance. l2 space: report raw distance.
            score = 1 - dist if space == "cosine" else dist
            hits.append((score, meta.get("title", "?")[:44], meta.get("section", "")[:30]))
        rows.append((q, hits))
    return rows


def main():
    from llama_index.embeddings.openai import OpenAIEmbedding

    embed_model = OpenAIEmbedding(model=settings.OPENAI_EMBED_MODEL)

    for label, queries in (("ANSWERABLE", ANSWERABLE), ("OUT-OF-SCOPE", OUT_OF_SCOPE)):
        print(f"\n{'=' * 100}\n{label} QUERIES — v2 collection (cosine similarity, higher=better)\n{'=' * 100}")
        tops = []
        for q, hits in probe("devcon_chunks_v2", "cosine", queries, embed_model):
            top = hits[0][0]
            tops.append(top)
            print(f"\n  Q: {q}")
            for score, title, section in hits[:3]:
                print(f"     {score:.4f}  {title:<46} {section}")
        print(f"\n  >> top-1 scores: min={min(tops):.4f}  max={max(tops):.4f}  "
              f"mean={sum(tops) / len(tops):.4f}")


if __name__ == "__main__":
    main()
