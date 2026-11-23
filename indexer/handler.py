"""debbie-indexer: builds a new knowledge-base index version from s3://<bucket>/docs/.

Trigger: S3 ObjectCreated/ObjectRemoved under docs/ (or a manual invoke).
Every run is a full, idempotent rebuild of whatever is in docs/:

  list docs/ -> fingerprint (keys + ETags) -> unchanged since the live index? stop
  -> extract text (.docx via docx2txt, .txt/.md as UTF-8) -> clean -> section-aware
  chunks (512 tok / 64 overlap) -> embeddings (reused from the live index for
  unchanged chunk text, new ones from text-embedding-3-small) -> validate ->
  write index/<UTC version>/ -> move index/current.json -> prune old versions.

Safety: the pointer only moves after validation passes; a build that would shrink
the index below MIN_KEEP_RATIO of the live one is refused (protects against a
partial upload or an accidental delete). Overlapping builds after a multi-file
upload are safe without serialising the function: a build whose docs/ listing
changed while it ran stands down (a newer event covers it), and the pointer is
moved with an S3 conditional write (If-Match on the ETag read at the start), so
a slower build can never overwrite a newer pointer.

Optional sidecar docs/manifest.json maps file name -> {"file_id","title","drive_link"}
so citations keep their titles/links and the eval's expected file ids still match.

Manual invokes:
  {"action": "rebuild", "force": true}            ignore fingerprint + shrink guard
  {"action": "promote", "version": "<version>"}   point current.json at a version (rollback)
"""

import hashlib
import io
import json
import logging
import os
import posixpath
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np

from app.services import index_format as fmt
from app.services.chunking import (CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, chunk_document_text,
                                   clean_text, default_tokenizer)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SIDECAR = "manifest.json"
TEXT_SUFFIXES = (".txt", ".md")
KEEP_VERSIONS = 10


class BuildRefused(RuntimeError):
    pass


class PointerConflict(RuntimeError):
    """current.json changed since this build read it (a concurrent build won)."""


# ------------------------------------------------------------------ storage

class S3Store:
    def __init__(self, bucket: str, client=None):
        self.bucket = bucket
        if client is None:
            import boto3
            client = boto3.client("s3")
        self.s3 = client

    def list(self, prefix: str) -> List[dict]:
        out, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kw)
            out += [{"key": o["Key"], "etag": o["ETag"].strip('"'), "size": o["Size"]}
                    for o in resp.get("Contents", [])]
            if not resp.get("IsTruncated"):
                return out
            token = resp["NextContinuationToken"]

    def get(self, key: str) -> Optional[bytes]:
        return self.get_with_etag(key)[0]

    def get_with_etag(self, key: str):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read(), resp["ETag"]
        except self.s3.exceptions.NoSuchKey:
            return None, None

    def put_if(self, key: str, data: bytes, etag: Optional[str]):
        """Compare-and-swap: If-Match the ETag we read, or If-None-Match * if there was none."""
        kw = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data,
                               ContentType="application/json", **kw)
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code")
            if code in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise PointerConflict(key) from e
            raise

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def delete(self, keys: List[str]):
        for i in range(0, len(keys), 1000):
            self.s3.delete_objects(Bucket=self.bucket,
                                   Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]})


# ---------------------------------------------------------------- extraction

def extract_text(key: str, data: bytes) -> str:
    name = key.lower()
    if name.endswith(".docx"):
        import docx2txt
        return docx2txt.process(io.BytesIO(data)) or ""
    if name.endswith(TEXT_SUFFIXES):
        return data.decode("utf-8-sig")
    raise ValueError(f"unsupported document type: {key}")


def _default_file_id(key: str) -> str:
    return "s3-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _title_from_key(key: str) -> str:
    stem = posixpath.splitext(posixpath.basename(key))[0]
    return re.sub(r"\s+", " ", stem).strip()


# ------------------------------------------------------------------- build

def build(store, docs_prefix: str, index_prefix: str,
          embed_texts: Callable[[List[str]], List[List[float]]],
          ntokens: Optional[Callable[[str], int]] = None,
          min_keep_ratio: float = 0.5, force: bool = False, dim: int = 1536,
          embed_model: str = "text-embedding-3-small", now: Optional[datetime] = None) -> dict:
    ntokens = ntokens or default_tokenizer()
    pointer_key = index_prefix + fmt.POINTER_FILE

    def listing():
        objs = [o for o in store.list(docs_prefix) if not o["key"].endswith("/") and o["size"] > 0]
        fp = hashlib.sha256(json.dumps(
            [[o["key"], o["etag"]] for o in sorted(objs, key=lambda o: o["key"])]).encode()).hexdigest()
        return objs, fp

    objects, fingerprint = listing()
    sidecar_obj = next((o for o in objects if o["key"] == docs_prefix + SIDECAR), None)
    doc_objs = sorted((o for o in objects if o is not sidecar_obj), key=lambda o: o["key"])

    # live index (may not exist yet)
    live_version, live_manifest, live_chunks, live_emb = None, {}, [], None
    raw_pointer, pointer_etag = store.get_with_etag(pointer_key)
    if raw_pointer:
        live_version = json.loads(raw_pointer).get("version")
        raw_m = store.get(f"{index_prefix}{live_version}/{fmt.MANIFEST_FILE}")
        live_manifest = json.loads(raw_m) if raw_m else {}
    if not force and live_manifest.get("source_fingerprint") == fingerprint:
        logger.info(json.dumps({"event": "index_unchanged", "version": live_version}))
        return {"status": "unchanged", "version": live_version}
    if not doc_objs:
        raise BuildRefused(f"no documents under {docs_prefix}")

    sidecar = {}
    if sidecar_obj:
        try:
            sidecar = json.loads(store.get(sidecar_obj["key"]).decode("utf-8-sig"))
        except ValueError as e:
            raise BuildRefused(f"{sidecar_obj['key']} is not valid JSON: {e}")
        if not isinstance(sidecar, dict):
            raise BuildRefused(f"{sidecar_obj['key']} must be a JSON object")

    # chunk every document
    records, documents = [], []
    for obj in doc_objs:
        name = obj["key"][len(docs_prefix):]
        meta = sidecar.get(name, {})
        file_id = meta.get("file_id") or _default_file_id(obj["key"])
        title = meta.get("title") or _title_from_key(obj["key"])
        text = clean_text(extract_text(obj["key"], store.get(obj["key"])))
        pieces = chunk_document_text(text, ntokens)
        if not pieces:
            raise BuildRefused(f"{obj['key']}: no text extracted")
        sections = set()
        for idx, (section, piece) in enumerate(pieces):
            records.append({"id": f"{file_id}__{idx:03d}", "text": piece, "metadata": {
                "title": title, "source": name, "file_id": file_id,
                "drive_link": meta.get("drive_link", ""), "modified_time": meta.get("modified_time", ""),
                "section": section or "", "chunk_index": idx}})
            if section:
                sections.add(section)
        documents.append({"file_id": file_id, "title": title, "source": name,
                          "chunk_count": len(pieces), "sections": len(sections),
                          "etag": obj["etag"]})

    if live_manifest.get("count") and not force and \
            len(records) < min_keep_ratio * live_manifest["count"]:
        raise BuildRefused(f"new index has {len(records)} chunks, below {min_keep_ratio:.0%} of the "
                           f"live {live_manifest['count']} — refusing (use force to override)")

    # embeddings: reuse vectors of unchanged chunk text from the live index
    reuse: Dict[str, np.ndarray] = {}
    if live_version:
        raw_c = store.get(f"{index_prefix}{live_version}/{fmt.CHUNKS_FILE}")
        raw_e = store.get(f"{index_prefix}{live_version}/{fmt.EMBEDDINGS_FILE}")
        if raw_c and raw_e:
            live_chunks, live_emb = fmt.parse_chunks(raw_c), fmt.load_embeddings(raw_e)
            if live_emb.shape == (len(live_chunks), dim):
                reuse = {c["text"]: live_emb[i] for i, c in enumerate(live_chunks)}
    todo = [r["text"] for r in records if r["text"] not in reuse]
    fresh = dict(zip(todo, embed_texts(todo))) if todo else {}
    emb = np.asarray([reuse[r["text"]] if r["text"] in reuse else fresh[r["text"]] for r in records],
                     dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    now = now or datetime.now(timezone.utc)
    # unique even when several builds start in the same second (they write separate prefixes)
    version = now.strftime("%Y%m%dT%H%M%SZ") + "-" + fingerprint[:4] + hashlib.sha1(
        os.urandom(8)).hexdigest()[:4]
    chunks_raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    emb_raw = fmt.dump_embeddings(emb)
    manifest = {
        "version": version, "created": now.isoformat(timespec="seconds"),
        "source": f"indexer build of {docs_prefix}", "source_fingerprint": fingerprint,
        "previous_version": live_version, "kb_version": now.strftime("%Y-%m"),
        "embed_model": embed_model, "dim": dim, "count": len(records),
        "chunk_tokens": CHUNK_TOKENS, "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "embeddings_reused": len(records) - len(todo), "embeddings_new": len(todo),
        "documents": documents,
        "sha256": {fmt.CHUNKS_FILE: fmt.sha256_bytes(chunks_raw),
                   fmt.EMBEDDINGS_FILE: fmt.sha256_bytes(emb_raw)},
    }
    fmt.validate(manifest, chunks_raw, emb_raw, expected_dim=dim)  # raises -> pointer untouched

    vprefix = f"{index_prefix}{version}/"
    store.put(vprefix + fmt.CHUNKS_FILE, chunks_raw, "application/x-ndjson")
    store.put(vprefix + fmt.EMBEDDINGS_FILE, emb_raw)
    store.put(vprefix + fmt.MANIFEST_FILE, json.dumps(manifest, indent=2).encode(), "application/json")

    written = [vprefix + n for n in (fmt.CHUNKS_FILE, fmt.EMBEDDINGS_FILE, fmt.MANIFEST_FILE)]
    if listing()[1] != fingerprint:   # docs changed mid-build: the newer event's build wins
        store.delete(written)
        logger.info(json.dumps({"event": "index_superseded", "version": version}))
        return {"status": "superseded", "version": version}
    try:
        store.put_if(pointer_key, json.dumps({"version": version, "previous": live_version}).encode(),
                     pointer_etag)
    except PointerConflict:
        store.delete(written)          # a concurrent build won; don't leave an orphan version
        logger.info(json.dumps({"event": "index_pointer_conflict", "version": version}))
        return {"status": "conflict", "version": version}
    pruned = prune(store, index_prefix, keep={version, live_version})
    result = {"status": "built", "version": version, "previous": live_version,
              "chunks": len(records), "documents": len(documents),
              "embeddings_new": len(todo), "pruned": pruned}
    logger.info(json.dumps({"event": "index_built", **result}))
    return result


def prune(store, index_prefix: str, keep: set, keep_versions: int = KEEP_VERSIONS) -> List[str]:
    versions = sorted({o["key"][len(index_prefix):].split("/", 1)[0]
                       for o in store.list(index_prefix) if "/" in o["key"][len(index_prefix):]})
    old = [v for v in versions[:-keep_versions] if v not in keep]
    if old:
        store.delete([o["key"] for o in store.list(index_prefix)
                      if o["key"][len(index_prefix):].split("/", 1)[0] in old])
    return old


def promote(store, index_prefix: str, version: str) -> dict:
    raw_m = store.get(f"{index_prefix}{version}/{fmt.MANIFEST_FILE}")
    if not raw_m:
        raise BuildRefused(f"version {version} not found")
    manifest = json.loads(raw_m)
    fmt.validate(manifest, store.get(f"{index_prefix}{version}/{fmt.CHUNKS_FILE}"),
                 store.get(f"{index_prefix}{version}/{fmt.EMBEDDINGS_FILE}"),
                 expected_dim=manifest.get("dim", 1536))
    pointer_key = index_prefix + fmt.POINTER_FILE
    raw = store.get(pointer_key)
    previous = json.loads(raw).get("version") if raw else None
    store.put(pointer_key, json.dumps({"version": version, "previous": previous}).encode(),
              "application/json")
    logger.info(json.dumps({"event": "index_promoted", "version": version, "previous": previous}))
    return {"status": "promoted", "version": version, "previous": previous}


# ----------------------------------------------------------------- handler

def handler(event, context):
    from app.services import llm  # reads the OpenAI key from SSM on first use

    store = S3Store(os.environ["INDEX_BUCKET"])
    docs_prefix = os.environ.get("DOCS_PREFIX", "docs/")
    index_prefix = os.environ.get("INDEX_PREFIX", "index/")
    event = event or {}
    if event.get("action") == "promote":
        return promote(store, index_prefix, event["version"])
    keys = [r.get("s3", {}).get("object", {}).get("key") for r in event.get("Records", [])]
    logger.info(json.dumps({"event": "indexer_invoked", "trigger_keys": keys[:20],
                            "action": event.get("action", "s3")}))
    return build(store, docs_prefix, index_prefix, llm.embed_texts,
                 min_keep_ratio=float(os.environ.get("MIN_KEEP_RATIO", "0.5")),
                 force=bool(event.get("force")),
                 embed_model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
