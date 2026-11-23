"""IndexStore: exact cosine vs a brute-force reference, pointer refresh, and
refusal of a corrupt version (the previous one keeps serving)."""

import json

import numpy as np
import pytest

from app.services import index_format as fmt
from app.services.index_store import IndexStore, LocalSource


def write_version(root, version, n=12, dim=16, seed=0, corrupt=False):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    chunks = [{"id": f"doc{i % 3}__{i:03d}", "text": f"chunk {i}",
               "metadata": {"title": f"Doc {i % 3}", "file_id": f"doc{i % 3}"}} for i in range(n)]
    chunks_raw = "".join(json.dumps(c) + "\n" for c in chunks).encode()
    emb_raw = fmt.dump_embeddings(emb)
    manifest = {"version": version, "count": n, "dim": dim,
                "sha256": {fmt.CHUNKS_FILE: fmt.sha256_bytes(chunks_raw),
                           fmt.EMBEDDINGS_FILE: fmt.sha256_bytes(emb_raw)}}
    vdir = root / version
    vdir.mkdir(parents=True)
    (vdir / fmt.CHUNKS_FILE).write_bytes(chunks_raw + (b"tampered" if corrupt else b""))
    (vdir / fmt.EMBEDDINGS_FILE).write_bytes(emb_raw)
    (vdir / fmt.MANIFEST_FILE).write_text(json.dumps(manifest))
    return chunks, emb


def point(root, version, store):
    fmt.write_pointer(root, version)
    store._checked_at = -1e9          # bypass the refresh interval
    store.source = LocalSource(str(root))
    store._etag = None


def test_query_matches_brute_force_cosine(tmp_path):
    chunks, emb = write_version(tmp_path, "v1")
    fmt.write_pointer(tmp_path, "v1")
    store = IndexStore(LocalSource(str(tmp_path)), expected_dim=16)
    assert store.load() is True
    q = np.random.default_rng(1).normal(size=16).astype(np.float32) * 3.7  # unnormalised
    ids, dists = store.query(q, 5)
    ref = emb @ (q / np.linalg.norm(q))
    want = np.argsort(-ref)[:5]
    assert ids == [chunks[i]["id"] for i in want]
    assert np.allclose(dists, 1.0 - ref[want], atol=1e-6)


def test_n_larger_than_index_and_zero_query(tmp_path):
    write_version(tmp_path, "v1", n=4)
    fmt.write_pointer(tmp_path, "v1")
    store = IndexStore(LocalSource(str(tmp_path)), expected_dim=16)
    store.load()
    assert len(store.query(np.ones(16), 50)[0]) == 4
    assert store.query(np.zeros(16), 3) == ([], [])


def test_refresh_swaps_version_and_refuses_corrupt_one(tmp_path):
    write_version(tmp_path, "v1")
    fmt.write_pointer(tmp_path, "v1")
    store = IndexStore(LocalSource(str(tmp_path)), expected_dim=16, refresh_seconds=300)
    store.load()
    assert store.maybe_refresh() is False            # within the interval: no I/O

    write_version(tmp_path, "v2", seed=2)
    point(tmp_path, "v2", store)
    assert store.maybe_refresh() is True and store.version == "v2"

    write_version(tmp_path, "v3", corrupt=True)
    point(tmp_path, "v3", store)
    assert store.maybe_refresh() is False            # checksum mismatch -> refused
    assert store.version == "v2" and len(store.nodes) == 12


@pytest.mark.parametrize("mutate,msg", [
    (lambda c, e: ([], e[:0]), "no chunks"),
    (lambda c, e: (c, e * 2), "normalised"),
    (lambda c, e: (c + c[:1], np.vstack([e, e[:1]])), "unique"),
])
def test_validate_rejects_bad_indexes(mutate, msg):
    rng = np.random.default_rng(0)
    e = rng.normal(size=(3, 8)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    c = [{"id": f"a__{i}", "text": "t"} for i in range(3)]
    c, e = mutate(c, e)
    raw_c = "".join(json.dumps(x) + "\n" for x in c).encode()
    with pytest.raises(fmt.IndexValidationError, match=msg):
        fmt.validate({}, raw_c, fmt.dump_embeddings(e), expected_dim=8)
