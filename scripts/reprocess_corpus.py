"""Offline corpus reprocessing: rebuild the knowledge base from the existing
ChromaDB collection — no Google Drive access required.

Pipeline:
  1. Back up chroma_db/ -> server/backups/chroma_db_<date>/
  2. Extract all chunks from the legacy collection (devcon_documents_openai_1536)
  3. Reconstruct each source document from verbatim chunk slices (start/end char offsets)
  4. Clean text artifacts (docx2txt table noise, CRLF)
  5. Detect section headings (docx2txt keeps headings as short standalone lines)
  6. Re-chunk: heading-bounded, token-based (512 tokens / 64 overlap)
  7. Embed with OpenAI text-embedding-3-small
  8. Write a NEW collection `devcon_chunks_v2` (cosine space) + corpus JSONL artifacts

Usage (from server/):
  venv/Scripts/python scripts/reprocess_corpus.py --verify      # stats only, no writes
  venv/Scripts/python scripts/reprocess_corpus.py --dry-run     # everything except embed+write
  venv/Scripts/python scripts/reprocess_corpus.py               # full run
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from datetime import date
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.chunking import (  # noqa: E402
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TOKENS,
    chunk_sections,
    split_sections,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("reprocess")

LEGACY_COLLECTION = "devcon_documents_openai_1536"
NEW_COLLECTION_DEFAULT = "devcon_chunks_v2"
CORPUS_DIR = SERVER_ROOT / "app" / "storage" / "corpus"
BACKUP_ROOT = SERVER_ROOT / "backups"

# Doc with docx2txt table artifacts ("nan:" rows) — the Strategy Canvas summary.
STRATEGY_CANVAS_FILE_ID = "1LFD79zPSUUrGOShyjWrYxEZ0lUZ6iGBd"


# ---------------------------------------------------------------- extraction

def load_legacy_chunks():
    """Return {file_id: {"meta": doc_meta, "chunks": [(start, end, text)]}}."""
    import chromadb

    client = chromadb.PersistentClient(path=str(SERVER_ROOT / settings.CHROMA_PERSIST_DIR))
    col = client.get_collection(LEGACY_COLLECTION)
    data = col.get(include=["documents", "metadatas"])

    docs = {}
    for text, meta in zip(data["documents"], data["metadatas"]):
        node = json.loads(meta["_node_content"])
        start, end = node.get("start_char_idx"), node.get("end_char_idx")
        if start is None or end is None:
            raise ValueError(f"chunk missing char offsets (file_id={meta.get('file_id')})")
        fid = meta["file_id"]
        entry = docs.setdefault(fid, {"meta": {
            "file_id": fid,
            "source": meta.get("source", ""),
            "title": meta.get("title") or meta.get("source", "Unknown"),
            "drive_link": meta.get("drive_link", ""),
            "modified_time": meta.get("modified_time", ""),
            "mime_type": meta.get("mime_type", ""),
        }, "chunks": []})
        entry["chunks"].append((start, end, text))
    log.info("legacy collection: %d chunks across %d docs",
             sum(len(d["chunks"]) for d in docs.values()), len(docs))
    return docs


def reconstruct(doc):
    """Stitch verbatim [start:end] slices back into the original document text."""
    chunks = sorted(doc["chunks"], key=lambda c: c[0])
    total = max(end for _, end, _ in chunks)
    buf = [None] * total
    conflicts = 0
    for start, end, text in chunks:
        if end - start != len(text):
            raise ValueError(f"offset/len mismatch in {doc['meta']['title']}")
        for i, ch in enumerate(text):
            pos = start + i
            if buf[pos] is not None and buf[pos] != ch:
                conflicts += 1
            buf[pos] = ch
    gaps = []  # [(start, end)] of never-written positions
    in_gap = None
    for i, ch in enumerate(buf):
        if ch is None and in_gap is None:
            in_gap = i
        elif ch is not None and in_gap is not None:
            gaps.append((in_gap, i))
            in_gap = None
    if in_gap is not None:
        gaps.append((in_gap, total))
    text = "".join(ch if ch is not None else "\n" for ch in buf)
    return text, gaps, conflicts


# ------------------------------------------------------------------ cleaning

def clean_text(text: str, file_id: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if file_id == STRATEGY_CANVAS_FILE_ID:
        lines = []
        for line in text.split("\n"):
            s = line.strip()
            if s == "nan" or s == "nan:":
                continue
            lines.append(re.sub(r"^nan:\s*", "", line))
        text = "\n".join(lines)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="reconstruction/section stats only")
    ap.add_argument("--dry-run", action="store_true", help="skip embedding + collection write")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--collection", default=NEW_COLLECTION_DEFAULT)
    args = ap.parse_args()

    persist_dir = SERVER_ROOT / settings.CHROMA_PERSIST_DIR

    # 1. backup (kept outside app/storage so it never ships in the image)
    if not (args.verify or args.dry_run or args.no_backup):
        backup_dir = BACKUP_ROOT / f"chroma_db_{date.today():%Y%m%d}"
        if not backup_dir.exists():
            log.info("backing up %s -> %s", persist_dir, backup_dir)
            shutil.copytree(persist_dir, backup_dir)
        else:
            log.info("backup already exists: %s", backup_dir)

    # 2-3. extract + reconstruct
    docs = load_legacy_chunks()
    corpus = []
    total_gap_chars = 0
    for fid, doc in docs.items():
        text, gaps, conflicts = reconstruct(doc)
        gap_chars = sum(e - s for s, e in gaps)
        total_gap_chars += gap_chars
        if gaps:
            log.warning("%-60s %d gap(s), %d chars (boundary whitespace)",
                        doc["meta"]["title"][:60], len(gaps), gap_chars)
        if conflicts:
            log.warning("%s: %d overlap conflicts", doc["meta"]["title"], conflicts)
        # 4. clean
        cleaned = text if args.no_clean else clean_text(text, fid)
        corpus.append({**doc["meta"], "text": cleaned,
                       "raw_chars": len(text), "gap_chars": gap_chars})
    log.info("reconstruction complete: %d docs, %d gap chars corpus-wide",
             len(corpus), total_gap_chars)

    # 5-6. sections + chunks
    import tiktoken
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS)
    enc = tiktoken.get_encoding("cl100k_base")
    ntokens = lambda s: len(enc.encode(s, disallowed_special=()))  # noqa: E731

    all_chunks = []   # dict records, ready for jsonl + collection
    for doc in corpus:
        sections = split_sections(doc["text"])
        pieces = chunk_sections(sections, splitter, ntokens)
        title_display = re.sub(r"\.docx$", "", doc["title"]).strip()
        doc["title"] = title_display
        for idx, (heading, piece) in enumerate(pieces):
            # Packed chunks join sections with \n\n, so match on a prefix; the
            # offsets are provenance metadata (approximate), not used in retrieval.
            pos = doc["text"].find(piece[:80])
            all_chunks.append({
                "chunk_id": f"{doc['file_id']}__{idx:03d}",
                "file_id": doc["file_id"],
                "source": doc["source"],
                "title": title_display,
                "drive_link": doc["drive_link"],
                "modified_time": doc["modified_time"],
                "section": heading or "",
                "chunk_index": idx,
                "start_char": pos,
                "end_char": pos + len(piece) if pos >= 0 else -1,
                "text": piece,
            })
        doc["sections"] = [h for h, _ in sections if h]
        doc["chunk_count"] = len(pieces)

    # report
    log.info("%-58s %6s %8s", "document", "chunks", "sections")
    for doc in sorted(corpus, key=lambda d: -d["chunk_count"]):
        log.info("%-58s %6d %8d", doc["title"][:58], doc["chunk_count"], len(doc["sections"]))
    log.info("TOTAL chunks: %d (legacy had 92)", len(all_chunks))
    sizes = sorted(len(c["text"]) for c in all_chunks)
    log.info("chunk chars: min=%d p50=%d p90=%d max=%d",
             sizes[0], sizes[len(sizes) // 2], sizes[int(len(sizes) * .9)], sizes[-1])

    if args.verify:
        for doc in corpus:
            heads = doc["sections"][:8]
            log.info("sections[%s]: %s", doc["title"][:40], "; ".join(heads) or "(none)")
        return

    # write corpus artifacts
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_DIR / "docs.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps({k: doc[k] for k in
                                ("file_id", "source", "title", "drive_link",
                                 "modified_time", "text", "gap_chars")},
                               ensure_ascii=False) + "\n")
    with open(CORPUS_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    manifest = {
        "kb_version": "2025-06",
        "generated": str(date.today()),
        "collection": args.collection,
        "embed_model": settings.OPENAI_EMBED_MODEL,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "total_chunks": len(all_chunks),
        "documents": [{
            "file_id": d["file_id"], "title": d["title"], "source": d["source"],
            "chunk_count": d["chunk_count"], "sections": d["sections"],
            "chars": len(d["text"]),
            "checksum": hashlib.md5(d["text"].encode("utf-8")).hexdigest(),
        } for d in corpus],
    }
    with open(CORPUS_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("corpus artifacts written to %s", CORPUS_DIR)

    if args.dry_run:
        log.info("dry-run: skipping embedding + collection write")
        return

    # 7. embed
    from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
    from llama_index.embeddings.openai import OpenAIEmbedding

    embed_model = OpenAIEmbedding(model=settings.OPENAI_EMBED_MODEL,
                                  embed_batch_size=settings.EMBEDDING_BATCH_SIZE)
    texts = [c["text"] for c in all_chunks]
    log.info("embedding %d chunks with %s ...", len(texts), settings.OPENAI_EMBED_MODEL)
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

    # 8. write new collection (cosine space) via llama-index for compatible node payloads
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(args.collection)
        log.info("deleted pre-existing collection %s", args.collection)
    except Exception:
        pass
    col = client.create_collection(args.collection, metadata={"hnsw:space": "cosine"})
    store = ChromaVectorStore(chroma_collection=col)

    nodes = []
    for c, emb in zip(all_chunks, embeddings):
        meta = {k: c[k] for k in ("source", "title", "file_id", "drive_link",
                                  "modified_time", "section", "chunk_index",
                                  "start_char", "end_char")}
        meta["kb_version"] = manifest["kb_version"]
        node = TextNode(id_=c["chunk_id"], text=c["text"], metadata=meta, embedding=emb)
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=c["file_id"])
        node.excluded_llm_metadata_keys = list(meta.keys())
        node.excluded_embed_metadata_keys = list(meta.keys())
        nodes.append(node)
    store.add(nodes)
    log.info("collection %s written: %d chunks (space=cosine)", args.collection, col.count())
    log.info("done. Legacy collection %s left untouched for rollback.", LEGACY_COLLECTION)


if __name__ == "__main__":
    main()
