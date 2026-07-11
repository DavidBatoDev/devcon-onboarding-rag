"""Unit tests for the deterministic pieces of the v2 pipeline."""

import pytest

from app.services.chunking import chunk_sections, is_heading, split_sections
from app.services.rag_pipeline import (
    _snippet,
    rrf_fuse,
    sanitize_citations,
    tokenize,
)


# ------------------------------------------------------------- citations

class TestSanitizeCitations:
    def test_keeps_valid_and_reports_cited(self):
        text, cited = sanitize_citations("Due in 30 days [1]. See the guide [3].", 3)
        assert text == "Due in 30 days [1]. See the guide [3]."
        assert cited == [1, 3]

    def test_strips_hallucinated_numbers(self):
        text, cited = sanitize_citations("Fact [1]. Made up [7]. Also [12].", 2)
        assert "[7]" not in text and "[12]" not in text
        assert "[1]" in text
        assert cited == [1]

    def test_strips_legacy_marker(self):
        text, cited = sanitize_citations("Answer.\nSHOW_REFERENCES: true", 0)
        assert "SHOW_REFERENCES" not in text
        assert cited == []

    def test_zero_sources_strips_all(self):
        text, cited = sanitize_citations("General advice [1].", 0)
        assert "[1]" not in text
        assert cited == []


# ------------------------------------------------------------------ RRF

class TestRRF:
    def test_item_in_both_rankings_wins(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60)
        assert max(fused, key=fused.get) == "b"

    def test_rank_only_ignores_scores(self):
        # identical rankings -> identical fused scores regardless of any scale
        f1 = rrf_fuse([["x", "y"]], k=60)
        assert f1["x"] > f1["y"]
        assert f1["x"] == pytest.approx(1 / 61)

    def test_empty(self):
        assert rrf_fuse([]) == {}


# ------------------------------------------------------------- tokenizer

def test_tokenize_lowercases_and_splits_hyphens():
    assert tokenize("Oath-Taking CEREMONY 2025") == ["oath", "taking", "ceremony", "2025"]


def test_snippet_cuts_on_word_boundary():
    s = _snippet("word " * 100, limit=50)
    assert len(s) <= 51 and s.endswith("…") and not s[:-1].endswith(" wor")


# ------------------------------------------------------------- headings

class TestHeadingDetection:
    def test_titleish_line_after_blank(self):
        assert is_heading("How to Apply", prev_blank=True)

    def test_requires_preceding_blank(self):
        assert not is_heading("How to Apply", prev_blank=False)

    def test_rejects_bullets_and_sentences(self):
        assert not is_heading("- a bullet item", prev_blank=True)
        assert not is_heading("1. numbered item", prev_blank=True)
        assert not is_heading("this ends like a sentence.", prev_blank=True)

    def test_accepts_colon_and_allcaps(self):
        assert is_heading("Requirements:", prev_blank=True)
        assert is_heading("NEW CHAPTER LEADERS OATH-TAKING", prev_blank=True)


class TestSections:
    TEXT = "Intro paragraph text here.\n\nHow to Apply\n\nSend your CV.\n\nDeadlines\n\nApply by June."

    def test_split_labels_sections(self):
        sections = split_sections(self.TEXT)
        headings = [h for h, _ in sections]
        assert headings == [None, "How to Apply", "Deadlines"]
        # heading line stays inside its section body
        assert sections[1][1].startswith("How to Apply")

    def test_chunk_packing_merges_small_sections(self):
        sections = split_sections(self.TEXT)

        class NoSplit:
            def split_text(self, t):
                return [t]

        chunks = chunk_sections(sections, NoSplit(), ntokens=lambda s: len(s.split()))
        # tiny sections pack into ONE chunk labelled by the first labelled section
        assert len(chunks) == 1
        label, text = chunks[0]
        assert label == "How to Apply"
        assert "Apply by June." in text

    def test_oversized_section_is_split_alone(self):
        sections = [("Big", "x " * 50), ("Small", "y")]

        class HalfSplit:
            def split_text(self, t):
                mid = len(t) // 2
                return [t[:mid], t[mid:]]

        # ntokens: force "Big" over budget (>512)
        chunks = chunk_sections(sections, HalfSplit(), ntokens=lambda s: 600 if "x" in s else 1)
        labels = [l for l, _ in chunks]
        assert labels.count("Big") == 2 and "Small" in labels
