"""One-off migration: export the v2 Chroma collection to the serverless index format.

Writes <out>/<version>/{chunks.jsonl, embeddings.npy, manifest.json} and
<out>/current.json -> that version. Same vectors as Chroma, so retrieval (and the
confidence-gate thresholds calibrated on them) stays identical.

Needs `chromadb`, which is no longer a runtime dependency. Run it with the
pre-migration virtualenv (from server/):
    venv/Scripts/python scripts/export_index.py
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SERVER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_ROOT))

from app.services.index_format import (  # noqa: E402
    CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE, KEEP_METADATA, sha256_file, write_pointer,
)

COLLECTION = "devcon_chunks_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", default=str(SERVER_ROOT / "app" / "storage" / "chroma_db"))
    ap.add_argument("--out", default=str(SERVER_ROOT / "app" / "storage" / "index"))
    ap.add_argument("--kb-version", default="2025-06")
    args = ap.parse_args()

    import chromadb

    col = chromadb.PersistentClient(path=args.chroma_dir).get_collection(COLLECTION)
    data = col.get(include=["documents", "metadatas", "embeddings"])
    ids, texts, metas = data["ids"], data["documents"], data["metadatas"]
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    if emb.shape != (len(ids), 1536):
        raise SystemExit(f"unexpected embedding shape {emb.shape}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / norms).astype(np.float32)

    # Stable order: by chunk id (file_id__NNN), so diffs between versions are readable.
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-chroma-v2"
    vdir = Path(args.out) / version
    vdir.mkdir(parents=True, exist_ok=True)

    docs = {}
    with open(vdir / CHUNKS_FILE, "w", encoding="utf-8", newline="\n") as f:
        for i in order:
            md = {k: metas[i].get(k) for k in KEEP_METADATA if metas[i].get(k) is not None}
            f.write(json.dumps({"id": ids[i], "text": texts[i] or "", "metadata": md},
                               ensure_ascii=False) + "\n")
            d = docs.setdefault(md.get("file_id", ""), {"file_id": md.get("file_id", ""),
                                                        "title": md.get("title", ""),
                                                        "chunk_count": 0, "sections": set()})
            d["chunk_count"] += 1
            if md.get("section"):
                d["sections"].add(md["section"])
    np.save(vdir / EMBEDDINGS_FILE, emb[order])

    manifest = {
        "version": version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"chroma export of collection {COLLECTION}",
        "kb_version": args.kb_version,
        "embed_model": "text-embedding-3-small",
        "dim": int(emb.shape[1]),
        "count": len(ids),
        "documents": [{"file_id": d["file_id"], "title": d["title"],
                       "chunk_count": d["chunk_count"], "sections": len(d["sections"])}
                      for d in sorted(docs.values(), key=lambda d: d["title"])],
        "sha256": {CHUNKS_FILE: sha256_file(vdir / CHUNKS_FILE),
                   EMBEDDINGS_FILE: sha256_file(vdir / EMBEDDINGS_FILE)},
    }
    (vdir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    write_pointer(Path(args.out), version)
    print(f"exported {len(ids)} chunks, {len(docs)} documents -> {vdir}")
    print("checksum chunks:", manifest["sha256"][CHUNKS_FILE][:16], "embeddings:",
          hashlib.sha256(emb[order].tobytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
