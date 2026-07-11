"""Pipeline behavior tests with fakes — gate tiers, event protocol, citations
wiring, caching. No network."""

import pytest

from app.core.config import settings
from tests.conftest import FakeLLM


def events_of(gen):
    return list(gen)


def kinds(events):
    return [k for k, _ in events]


class TestConfidenceGate:
    def test_strong_tier_grounds_answer(self, make_pipeline):
        pipe, _ = make_pipeline(vector_distances=[0.4, 0.5, 0.6])  # cos 0.6 -> strong
        events = events_of(pipe.run_stream("q", nocache=True))
        sources = dict(events)["sources"]
        assert sources["grounded"] is True and sources["confidence"] == "strong"

    def test_weak_tier(self, make_pipeline):
        pipe, _ = make_pipeline(vector_distances=[0.60, 0.62, 0.64])  # cos 0.40 -> weak
        events = events_of(pipe.run_stream("q", nocache=True))
        sources = dict(events)["sources"]
        assert sources["grounded"] is True and sources["confidence"] == "weak"

    def test_none_tier_refuses_with_no_sources(self, make_pipeline):
        pipe, llm = make_pipeline(vector_distances=[0.80, 0.85, 0.9])  # cos 0.20 < cutoff
        events = events_of(pipe.run_stream("q", nocache=True))
        sources = dict(events)["sources"]
        assert sources["grounded"] is False and sources["sources"] == []
        # the general prompt (not the citation prompt) was used
        assert "Numbered source excerpts" not in llm.prompts[-1]

    def test_gate_uses_raw_cosine_not_rrf(self, make_pipeline):
        # RRF scores are ~0.016-0.03; if the gate used them everything would
        # be "none". Distances give cos 0.55 -> must be strong.
        pipe, _ = make_pipeline(vector_distances=[0.45, 0.5, 0.55])
        events = events_of(pipe.run_stream("q", nocache=True))
        assert dict(events)["sources"]["confidence"] == "strong"


class TestEventProtocol:
    def test_event_order_and_done_payload(self, make_pipeline):
        pipe, _ = make_pipeline(vector_distances=[0.4, 0.5, 0.6])
        events = events_of(pipe.run_stream("q", nocache=True))
        ks = kinds(events)
        assert ks[0] == "meta" and ks[1] == "sources" and ks[-1] == "done"
        assert ks.count("token") >= 2
        done = events[-1][1]
        assert done.answer and done.timings["total_ms"] >= 0
        assert done.request_id and done.model

    def test_tokens_concatenate_to_raw_answer(self, make_pipeline):
        llm = FakeLLM("Alpha [1] beta [2].")
        pipe, _ = make_pipeline(vector_distances=[0.4, 0.5, 0.6], llm=llm)
        events = events_of(pipe.run_stream("q", nocache=True))
        streamed = "".join(p for k, p in events if k == "token")
        assert streamed == "Alpha [1] beta [2]."

    def test_cited_flags_follow_sanitizer(self, make_pipeline):
        llm = FakeLLM("Fact [2]. Hallucinated [9].")
        pipe, _ = make_pipeline(vector_distances=[0.4, 0.5, 0.6], llm=llm)
        done = dict(events_of(pipe.run_stream("q", nocache=True)))["done"]
        assert [s.n for s in done.sources if s.cited] == [2]
        assert "[9]" not in done.answer

    def test_prompt_contains_numbered_sources_and_sections(self, make_pipeline):
        llm = FakeLLM()
        pipe, _ = make_pipeline(vector_distances=[0.4, 0.5, 0.6], llm=llm)
        events_of(pipe.run_stream("q", nocache=True))
        prompt = llm.prompts[-1]
        assert "[1] " in prompt and "— " in prompt  # numbered + section suffix
        assert "Never invent a citation number" in prompt


class TestCache:
    def test_cache_roundtrip_and_replay(self, make_pipeline):
        pipe, _ = make_pipeline(vector_distances=[0.4, 0.5, 0.6])
        if not settings.ENABLE_QUERY_CACHE:
            pytest.skip("cache disabled")
        first = dict(events_of(pipe.run_stream("same q")))["done"]
        assert first.cached is False
        replay = events_of(pipe.run_stream("same q"))
        done = dict(replay)["done"]
        assert done.cached is True
        assert done.answer == first.answer
        # replay still emits the full protocol
        assert kinds(replay) == ["meta", "sources", "token", "done"]

    def test_nocache_busts(self, make_pipeline):
        pipe, _ = make_pipeline(vector_distances=[0.4, 0.5, 0.6])
        r1 = dict(events_of(pipe.run_stream("q2")))["done"]
        r2 = dict(events_of(pipe.run_stream("q2", nocache=True)))["done"]
        assert r2.cached is False and r2.request_id != r1.request_id

    def test_key_varies_with_history_and_params(self, make_pipeline):
        pipe, _ = make_pipeline()
        k = pipe._cache_key
        assert k("q", [], None, None) != k("q", [{"role": "user", "content": "x"}], None, None)
        assert k("q", [], 3, None) != k("q", [], 5, None)
        assert k("q", [], None, 0.2) != k("q", [], None, None)
        assert k(" Q ", [], None, None) == k("q", [], None, None)  # normalized

    def test_key_includes_pipeline_version(self, make_pipeline, monkeypatch):
        pipe, _ = make_pipeline()
        k1 = pipe._cache_key("q", [], None, None)
        monkeypatch.setattr(settings, "PIPELINE_VERSION", "v3-test")
        assert pipe._cache_key("q", [], None, None) != k1


class TestHybridRetrieval:
    def test_bm25_only_hit_joins_candidates(self, make_pipeline):
        # Vector returns only docA chunks; "internship CV" should pull docB via BM25.
        pipe, _ = make_pipeline(vector_ids=["docA__000", "docA__001"],
                                vector_distances=[0.5, 0.55])
        cands = pipe.retrieve("internship application CV")
        ids = [c.node_id for c in cands]
        assert "docB__000" in ids
        docB = next(c for c in cands if c.node_id == "docB__000")
        assert docB.cosine is None and docB.bm25_rank is not None

    def test_vector_cutoff_applies_pre_fusion(self, make_pipeline):
        pipe, _ = make_pipeline(vector_distances=[0.9, 0.95, 0.99])  # all cos < 0.25
        cands = pipe.retrieve("zzz qqq unrelated")
        assert all(c.cosine is None for c in cands)  # only BM25 leg survives
