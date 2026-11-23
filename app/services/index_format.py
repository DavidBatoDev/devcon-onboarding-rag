"""On-disk / S3 format of a knowledge-base index version, shared by the API
(reader), the indexer Lambda (writer) and the one-off Chroma export.

    index/current.json            {"version": "<v>"}   <- the pointer; rollback = rewrite it
    index/<v>/chunks.jsonl        {"id", "text", "metadata"} per line, row i <-> embeddings[i]
    index/<v>/embeddings.npy      float32 (n, dim), L2-normalised -> cosine = E @ q
    index/<v>/manifest.json       version, counts, model, documents, sha256 of both files
"""

import hashlib
import io
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

POINTER_FILE = "current.json"
CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
MANIFEST_FILE = "manifest.json"
KEEP_METADATA = ("title", "source", "file_id", "drive_link", "modified_time",
                 "section", "chunk_index")


class IndexValidationError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_pointer(root: Path, version: str) -> None:
    (Path(root) / POINTER_FILE).write_text(json.dumps({"version": version}), encoding="utf-8")


def parse_chunks(raw: bytes) -> List[dict]:
    # split on "\n" only: str.splitlines() also breaks on U+2028/U+000C etc.,
    # which json.dumps(ensure_ascii=False) leaves unescaped inside strings.
    return [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]


def load_embeddings(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw), allow_pickle=False)


def dump_embeddings(emb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(emb, dtype=np.float32), allow_pickle=False)
    return buf.getvalue()


def validate(manifest: dict, chunks_raw: bytes, emb_raw: bytes,
             expected_dim: int = 1536) -> Tuple[List[dict], np.ndarray]:
    """Refuse anything that would break retrieval. Used by the indexer before it
    moves the pointer and by the API before it swaps a loaded index in."""
    for f, raw in ((CHUNKS_FILE, chunks_raw), (EMBEDDINGS_FILE, emb_raw)):
        want = (manifest.get("sha256") or {}).get(f)
        if want and want != sha256_bytes(raw):
            raise IndexValidationError(f"{f}: checksum mismatch")
    chunks = parse_chunks(chunks_raw)
    emb = load_embeddings(emb_raw)
    if not chunks:
        raise IndexValidationError("index has no chunks")
    if emb.ndim != 2 or emb.shape != (len(chunks), expected_dim):
        raise IndexValidationError(f"embeddings shape {emb.shape} != ({len(chunks)}, {expected_dim})")
    if manifest.get("count") not in (None, len(chunks)):
        raise IndexValidationError(f"manifest count {manifest.get('count')} != {len(chunks)} chunks")
    if not np.isfinite(emb).all():
        raise IndexValidationError("embeddings contain NaN/inf")
    norms = np.linalg.norm(emb, axis=1)
    if np.abs(norms - 1.0).max() > 1e-3:
        raise IndexValidationError("embeddings are not L2-normalised")
    ids = [c.get("id") for c in chunks]
    if len(set(ids)) != len(ids) or not all(ids):
        raise IndexValidationError("chunk ids missing or not unique")
    if not all(isinstance(c.get("text"), str) and c["text"].strip() for c in chunks):
        raise IndexValidationError("empty chunk text")
    return chunks, emb.astype(np.float32, copy=False)
