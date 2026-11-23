"""Indexer: docs -> chunks -> embeddings -> validated version -> pointer, plus the
safety rails (unchanged fingerprint, bad sidecar, shrink guard, promote/rollback,
pruning) and the sentence splitter. In-memory store, fake embeddings."""

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.services import index_format as fmt
from app.services.chunking import SentenceSplitter, clean_text
from indexer.handler import BuildRefused, build, extract_text, promote, prune

WORDS = lambda s: len(s.split())  # noqa: E731  deterministic token counter


class MemStore:
    def __init__(self):
        self.objects = {}

    def list(self, prefix):
        return [{"key": k, "etag": str(hash(v)), "size": len(v)}
                for k, v in sorted(self.objects.items()) if k.startswith(prefix)]

    def get(self, key):
        return self.objects.get(key)

    def get_with_etag(self, key):
        v = self.objects.get(key)
        return v, (str(hash(v)) if v is not None else None)

    def put_if(self, key, data, etag):
        from indexer.handler import PointerConflict
        current = self.get_with_etag(key)[1]
        if current != etag:
            raise PointerConflict(key)
        self.objects[key] = data

    def put(self, key, data, content_type=None):
        self.objects[key] = data

    def delete(self, keys):
        for k in keys:
            self.objects.pop(k, None)


class FakeEmbed:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += len(texts)
        out = []
        for t in texts:
            v = np.random.default_rng(abs(hash(t)) % 2**32).normal(size=8)
            out.append(list(v))
        return out


def make_docx(paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.'
           f'openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


DOC_A = "Intro text for chapters.\n\nHow to Apply\n\nSend your CV to HQ.\n\nDeadlines\n\nApply by June.\n"


def run(store, embed=None, **kw):
    t = kw.pop("now", datetime(2026, 11, 23, tzinfo=timezone.utc))
    return build(store, "docs/", "index/", embed or FakeEmbed(), ntokens=WORDS, dim=8, now=t, **kw)


def test_build_writes_valid_version_and_moves_pointer():
    s = MemStore()
    s.put("docs/Guide A.txt", DOC_A.encode())
    s.put("docs/Playbook.docx", make_docx(["Oath Taking", "Officers take an oath."]))
    s.put("docs/manifest.json", json.dumps({"Guide A.txt": {"file_id": "fileA", "title": "Guide A",
                                                            "drive_link": "https://d/a"}}).encode())
    res = run(s)
    assert res["status"] == "built" and res["documents"] == 2
    v = json.loads(s.get("index/current.json"))["version"]
    m = json.loads(s.get(f"index/{v}/manifest.json"))
    chunks, emb = fmt.validate(m, s.get(f"index/{v}/chunks.jsonl"), s.get(f"index/{v}/embeddings.npy"), 8)
    ids = [c["id"] for c in chunks]
    assert any(i.startswith("fileA__") for i in ids)            # sidecar file id kept
    assert any(c["metadata"]["section"] == "How to Apply" for c in chunks)
    assert emb.shape == (len(chunks), 8)


def test_unchanged_docs_skip_and_embeddings_are_reused():
    s = MemStore()
    s.put("docs/a.txt", DOC_A.encode())
    e1 = FakeEmbed()
    run(s, e1)
    assert run(s)["status"] == "unchanged"
    s.put("docs/b.txt", b"New Section\n\nBrand new text here.\n")
    e2 = FakeEmbed()
    res = run(s, e2, now=datetime(2026, 11, 24, tzinfo=timezone.utc))
    assert res["status"] == "built" and e2.calls == 1           # only the new chunk embedded


def test_invalid_sidecar_refused_and_pointer_untouched():
    s = MemStore()
    s.put("docs/a.txt", DOC_A.encode())
    run(s)
    before = s.get("index/current.json")
    s.put("docs/manifest.json", b"{not json")
    with pytest.raises(BuildRefused, match="not valid JSON"):
        run(s, now=datetime(2026, 11, 25, tzinfo=timezone.utc))
    assert s.get("index/current.json") == before


def test_shrink_guard_and_force():
    s = MemStore()
    for i in range(4):
        s.put(f"docs/d{i}.txt", f"Topic {i}\n\n{'word ' * 30}{i}\n".encode())
    run(s)
    for i in range(1, 4):
        s.delete([f"docs/d{i}.txt"])
    with pytest.raises(BuildRefused, match="refusing"):
        run(s, now=datetime(2026, 11, 25, tzinfo=timezone.utc))
    assert run(s, force=True, now=datetime(2026, 11, 26, tzinfo=timezone.utc))["status"] == "built"


def test_promote_rolls_back_and_rejects_unknown():
    s = MemStore()
    s.put("docs/a.txt", DOC_A.encode())
    v1 = run(s)["version"]
    s.put("docs/b.txt", b"More\n\ntext\n")
    v2 = run(s, now=datetime(2026, 11, 24, tzinfo=timezone.utc))["version"]
    assert promote(s, "index/", v1) == {"status": "promoted", "version": v1, "previous": v2}
    with pytest.raises(BuildRefused):
        promote(s, "index/", "nope")


def test_concurrent_build_cannot_overwrite_newer_pointer():
    s = MemStore()
    s.put("docs/a.txt", DOC_A.encode())
    run(s)

    class Racing(FakeEmbed):           # another build moves the pointer while we embed
        def __call__(self, texts):
            s.put("index/current.json", b'{"version": "someone-else"}')
            return super().__call__(texts)

    s.put("docs/b.txt", b"Other\n\nnew text\n")
    res = run(s, Racing(), now=datetime(2026, 11, 24, tzinfo=timezone.utc))
    assert res["status"] == "conflict"
    assert json.loads(s.get("index/current.json"))["version"] == "someone-else"
    assert not any(k.startswith(f"index/{res['version']}/") for k in s.objects)   # no orphan


def test_docs_changed_mid_build_stands_down():
    s = MemStore()
    s.put("docs/a.txt", DOC_A.encode())

    class Uploading(FakeEmbed):
        def __call__(self, texts):
            s.put("docs/late.txt", b"Late\n\narrived\n")
            return super().__call__(texts)

    assert run(s, Uploading())["status"] == "superseded"
    assert s.get("index/current.json") is None


def test_prune_keeps_newest_and_protected():
    s = MemStore()
    base = datetime(2026, 1, 1)
    versions = [(base + timedelta(days=i)).strftime("%Y%m%dT%H%M%SZ") for i in range(13)]
    for v in versions:
        s.put(f"index/{v}/manifest.json", b"{}")
    removed = prune(s, "index/", keep={versions[0]}, keep_versions=10)
    assert removed == versions[1:3]
    assert s.get(f"index/{versions[0]}/manifest.json") is not None


def test_extract_docx_and_clean():
    text = extract_text("docs/x.docx", make_docx(["Heading One", "Body line."]))
    assert "Heading One" in text and "Body line." in text
    assert clean_text("a\r\nnan:\nnan: b\n\n\n\nc  \n") == "a\nb\n\nc\n"
    with pytest.raises(ValueError):
        extract_text("docs/x.pdf", b"%PDF")


def test_sentence_splitter_budget_overlap_and_lossless():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    parts = SentenceSplitter(chunk_size=40, chunk_overlap=10, ntokens=WORDS).split_text(text)
    assert len(parts) > 1 and all(WORDS(p) <= 40 for p in parts)
    # ~chunk_overlap tokens of trailing sentences are carried into the next chunk
    assert parts[1].startswith(" ".join(parts[0].split()[-10:]))
    joined = " ".join(parts)
    assert all(f"Sentence number {i} is here." in joined for i in range(60))
