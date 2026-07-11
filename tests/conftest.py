"""Shared fakes for pipeline tests — no network, no ChromaDB, no OpenAI."""

from types import SimpleNamespace

import pytest

from app.services.rag_pipeline import AnswerPipeline


class FakeCompletion:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


class FakeLLM:
    """LLM double: .complete returns canned text; .stream_complete yields it
    in two chunks (to exercise real streaming assembly)."""

    def __init__(self, text="The answer is 30 days [1]. See the playbook [2]."):
        self.text = text
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return FakeCompletion(self.text)

    def stream_complete(self, prompt):
        self.prompts.append(prompt)
        mid = len(self.text) // 2
        for part in (self.text[:mid], self.text[mid:]):
            yield SimpleNamespace(delta=part)


def make_node(node_id, text, title="Doc", section=""):
    return SimpleNamespace(
        text=text,
        metadata={"title": title, "section": section, "source": f"{title}.docx",
                  "file_id": node_id.split("__")[0], "drive_link": f"https://drive/{node_id}"},
    )


@pytest.fixture
def fake_nodes():
    return {
        "docA__000": make_node("docA__000", "Chapter reports are due within 30 days.",
                               "Onboarding Playbook", "Reports"),
        "docA__001": make_node("docA__001", "Officers take an oath at the ceremony.",
                               "Onboarding Playbook", "Oath"),
        "docB__000": make_node("docB__000", "The internship application needs a CV.",
                               "Internship Guide", "How to Apply"),
    }


@pytest.fixture
def make_pipeline(fake_nodes):
    """Factory: build an AnswerPipeline with controllable vector results."""

    def _make(vector_ids=None, vector_distances=None, llm=None):
        ids = vector_ids if vector_ids is not None else list(fake_nodes.keys())
        dists = vector_distances if vector_distances is not None else [0.3, 0.4, 0.5]
        the_llm = llm or FakeLLM()
        return AnswerPipeline(
            llm_factory=lambda temp: the_llm,
            embed_fn=lambda q: [0.0] * 8,
            chroma_query=lambda emb, n: (ids[:n], dists[: len(ids[:n])]),
            nodes=fake_nodes,
        ), the_llm

    return _make
