"""Knowledge-base index in memory: exact cosine over one NumPy matrix.

135 chunks don't need a vector database: embeddings are L2-normalised once, so
cosine similarity for a query is a single matrix-vector product. Results are
returned as Chroma-style cosine *distances* (1 - cos) so the answer pipeline is
unchanged.

Sources: S3 (`INDEX_BUCKET`) on AWS, a local directory in development. Both
read `current.json` to find the live version (see index_format.py).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services import index_format as fmt

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Node-like record the pipeline expects (.text + .metadata)."""
    node_id: str
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


class LocalSource:
    def __init__(self, root: str):
        self.root = Path(root)

    def read_pointer(self, etag: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
        path = self.root / fmt.POINTER_FILE
        stamp = str(path.stat().st_mtime_ns)
        if stamp == etag:
            return None, etag
        return json.loads(path.read_text(encoding="utf-8")), stamp

    def read(self, version: str, name: str) -> bytes:
        return (self.root / version / name).read_bytes()


class S3Source:
    def __init__(self, bucket: str, prefix: str, s3_client=None):
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        if s3_client is None:
            import boto3
            s3_client = boto3.client("s3")
        self.s3 = s3_client

    def read_pointer(self, etag: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
        """Conditional GET: returns (None, etag) when the pointer hasn't changed."""
        kw = {"Bucket": self.bucket, "Key": self.prefix + fmt.POINTER_FILE}
        if etag:
            kw["IfNoneMatch"] = etag
        try:
            resp = self.s3.get_object(**kw)
        except Exception as e:  # botocore ClientError 304 -> unchanged
            code = getattr(e, "response", {}).get("Error", {}).get("Code")
            if code in ("304", "NotModified"):
                return None, etag
            raise
        return json.loads(resp["Body"].read()), resp.get("ETag")

    def read(self, version: str, name: str) -> bytes:
        resp = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}{version}/{name}")
        return resp["Body"].read()


class IndexStore:
    def __init__(self, source, refresh_seconds: int = 300, expected_dim: int = 1536):
        self.source = source
        self.refresh_seconds = refresh_seconds
        self.expected_dim = expected_dim
        self.version: Optional[str] = None
        self.manifest: dict = {}
        self.nodes: Dict[str, Chunk] = {}
        self._ids: List[str] = []
        self._emb: Optional[np.ndarray] = None
        self._etag: Optional[str] = None
        self._checked_at = 0.0

    # ------------------------------------------------------------- loading

    def load(self) -> bool:
        """(Re)load if the pointer moved. Returns True if a new version was swapped in.
        A broken version is refused and the current one keeps serving."""
        self._checked_at = time.monotonic()
        pointer, etag = self.source.read_pointer(self._etag)
        if pointer is None:
            return False
        version = pointer.get("version")
        if not version or version == self.version:
            self._etag = etag
            return False
        manifest = json.loads(self.source.read(version, fmt.MANIFEST_FILE))
        chunks, emb = fmt.validate(manifest, self.source.read(version, fmt.CHUNKS_FILE),
                                   self.source.read(version, fmt.EMBEDDINGS_FILE),
                                   expected_dim=self.expected_dim)
        self._ids = [c["id"] for c in chunks]
        self.nodes = {c["id"]: Chunk(c["id"], c["text"], c.get("metadata") or {}) for c in chunks}
        self._emb = emb
        self.manifest = manifest
        previous, self.version, self._etag = self.version, version, etag
        logger.info(json.dumps({"event": "index_loaded", "version": version,
                                "previous": previous, "chunks": len(chunks)}))
        return True

    def maybe_refresh(self) -> bool:
        if time.monotonic() - self._checked_at < self.refresh_seconds:
            return False
        try:
            return self.load()
        except Exception as e:
            logger.error("Index refresh failed, keeping version %s: %s", self.version, e)
            return False

    # --------------------------------------------------------------- query

    def query(self, embedding, n: int) -> Tuple[List[str], List[float]]:
        """Top-n by exact cosine. Returns (ids, cosine distances) like Chroma."""
        if self._emb is None or not self._ids:
            return [], []
        q = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            return [], []
        sims = self._emb @ (q / norm)
        n = min(n, len(self._ids))
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [self._ids[i] for i in top], [float(1.0 - sims[i]) for i in top]


def from_settings(settings) -> IndexStore:
    if settings.INDEX_BUCKET:
        source = S3Source(settings.INDEX_BUCKET, settings.INDEX_PREFIX)
    else:
        source = LocalSource(settings.INDEX_LOCAL_DIR)
    return IndexStore(source, refresh_seconds=settings.INDEX_REFRESH_SECONDS,
                      expected_dim=settings.EMBEDDING_DIM)
