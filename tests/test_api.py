"""HTTP layer with fakes: NDJSON stream protocol, rate limiting (429), client IP
on Lambda, feedback storage, stats, CORS. No network."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main as main_module
from app.api.v1 import routes
from app.core import ratelimit
from app.core.config import settings
from app.services import feedback_store
from app.services import index_format as fmt
from app.services.index_store import IndexStore, LocalSource
from app.services.rag_service import RAGService
from tests.conftest import FakeLLM


@pytest.fixture
def service(tmp_path):
    texts = ["Chapter reports are due within 30 days.", "Officers take an oath.",
             "The internship application needs a CV."]
    emb = np.eye(3, 16, dtype=np.float32)
    chunks = [{"id": f"doc{i}__000", "text": t,
               "metadata": {"title": f"Doc {i}", "file_id": f"doc{i}", "section": "S"}}
              for i, t in enumerate(texts)]
    raw_c = "".join(json.dumps(c) + "\n" for c in chunks).encode()
    raw_e = fmt.dump_embeddings(emb)
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / fmt.CHUNKS_FILE).write_bytes(raw_c)
    (tmp_path / "v1" / fmt.EMBEDDINGS_FILE).write_bytes(raw_e)
    (tmp_path / "v1" / fmt.MANIFEST_FILE).write_text(json.dumps(
        {"version": "v1", "count": 3, "kb_version": "test", "documents": [{"title": "Doc 0"}]}))
    fmt.write_pointer(tmp_path, "v1")
    q = np.zeros(16, dtype=np.float32)
    q[0], q[1] = 0.9, 0.1                      # cos ~0.99 with chunk 0 -> strong tier
    llm = FakeLLM("Reports are due in 30 days [1].")
    return RAGService(store=IndexStore(LocalSource(str(tmp_path)), expected_dim=16),
                      llm_factory=lambda t: llm, embed_fn=lambda s: q, condense_llm=llm)


@pytest.fixture
def client(service, monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "rag_service", service)
    monkeypatch.setattr(main_module, "rag_service", service)
    ratelimit.set_counter(ratelimit.MemoryCounter())
    store = feedback_store.JsonlFeedback(str(tmp_path / "fb.jsonl"))
    feedback_store.set_store(store)
    with TestClient(main_module.app) as c:
        c.feedback_path = tmp_path / "fb.jsonl"
        yield c
    ratelimit.set_counter(None)
    feedback_store.set_store(None)


def ndjson(resp):
    return [json.loads(l) for l in resp.text.splitlines() if l.strip()]


def test_stream_protocol(client):
    r = client.post("/api/v1/ask/stream", json={"query": "when are reports due?", "nocache": True},
                    headers={"Accept": "application/x-ndjson"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    events = ndjson(r)
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[1] == "sources" and types[-1] == "done"
    assert "token" in types
    assert events[0]["kb_version"] == "v1"
    assert events[1]["grounded"] is True and events[1]["sources"][0]["chunk_id"] == "doc0__000"
    assert events[-1]["cited"] == [1]


def test_ask_structured_and_stats(client):
    r = client.post("/api/v1/ask", json={"query": "reports?", "structured": True, "nocache": True})
    assert r.status_code == 200 and r.json()["metadata"]["confidence"] == "strong"
    s = client.get("/api/v1/stats").json()
    assert s["index_version"] == "v1" and s["document_count"] == 3 and s["status"] == "ready"


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_RETRIEVE", 2)
    codes = [client.post("/api/v1/retrieve", json={"query": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_client_ip_from_request_context_on_lambda(monkeypatch):
    monkeypatch.setattr(ratelimit, "ON_LAMBDA", True)

    class Req:
        headers = {"x-amzn-request-context": json.dumps({"identity": {"sourceIp": "203.0.113.9"}}),
                   "x-forwarded-for": "1.2.3.4"}
        client = None

    assert ratelimit.client_ip(Req()) == "203.0.113.9"


def test_dynamo_counter_uses_atomic_add_with_ttl():
    calls = []

    class FakeDDB:
        def update_item(self, **kw):
            calls.append(kw)
            return {"Attributes": {"count": {"N": "3"}}}

    assert ratelimit.DynamoCounter("t", client=FakeDDB()).incr("ask#ab#1", 999) == 3
    kw = calls[0]
    assert kw["UpdateExpression"].startswith("ADD #c :one")
    assert kw["ExpressionAttributeValues"][":ttl"] == {"N": "999"}


def test_feedback_is_stored_without_ip(client):
    r = client.post("/api/v1/feedback", json={"request_id": "abc", "rating": "up", "query": "q"})
    assert r.status_code == 200
    rec = json.loads(client.feedback_path.read_text().splitlines()[-1])
    assert rec["request_id"] == "abc" and rec["rating"] == "up"
    assert not any("ip" in k.lower() for k in rec)


@pytest.mark.parametrize("origin,allowed", [
    ("https://devcon-onboarding-web.vercel.app", True),
    ("https://devcon-onboarding-h1p8ddl2n-davids-projects-cc0a1acc.vercel.app", True),
    ("https://devcon-onboarding-web-git-aws-serverless-davids-projects-cc0a1acc.vercel.app", True),
    ("https://devcon-onboarding-web-evil.vercel.app", False),       # someone else's project
    ("https://devcon-onboarding-x-other-team.vercel.app", False),
    ("https://evil.example.com", False),
])
def test_cors_origins(client, origin, allowed):
    r = client.get("/api/v1/status", headers={"Origin": origin})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed


def test_admin_drive_endpoints_are_gone(client):
    assert client.post("/api/v1/rebuild", json={}).status_code in (404, 405)
    assert client.delete("/api/v1/index").status_code in (404, 405)
