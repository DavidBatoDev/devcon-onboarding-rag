"""OpenAI adapter: request shape per model family, and stream event mapping."""

from types import SimpleNamespace

import pytest

from app.services.llm import ChatModel


class FakeResponses:
    def __init__(self, events=None, text="ok"):
        self.calls = []
        self.events = events or []
        self.text = text

    def create(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            return iter(self.events)
        return SimpleNamespace(output_text=self.text)


def fake_client(**kw):
    return SimpleNamespace(responses=FakeResponses(**kw))


def test_non_reasoning_model_sends_temperature_not_reasoning():
    c = fake_client()
    ChatModel("gpt-4o-mini", temperature=0.1, max_output_tokens=2048, openai_client=c, api="responses").complete("hi")
    kw = c.responses.calls[0]
    assert kw["temperature"] == 0.1 and kw["max_output_tokens"] == 2048
    assert "reasoning" not in kw and kw["input"] == "hi"


def test_reasoning_model_sends_effort_and_temperature():
    c = fake_client()
    ChatModel("gpt-6-luna", temperature=0.1, reasoning_effort="none", openai_client=c, api="responses").complete("hi")
    kw = c.responses.calls[0]
    assert kw["reasoning"] == {"effort": "none"} and kw["temperature"] == 0.1


def test_complete_returns_str_able_text():
    c = fake_client(text="The answer [1].")
    assert str(ChatModel("m", openai_client=c, api="responses").complete("q")) == "The answer [1]."


def test_stream_yields_text_deltas_only_in_order():
    events = [SimpleNamespace(type="response.created"),
              SimpleNamespace(type="response.output_text.delta", delta="Hel"),
              SimpleNamespace(type="response.output_text.delta", delta="lo"),
              SimpleNamespace(type="response.completed")]
    c = fake_client(events=events)
    out = [d.delta for d in ChatModel("m", openai_client=c, api="responses").stream_complete("q")]
    assert out == ["Hel", "lo"] and c.responses.calls[0]["stream"] is True


def test_stream_failure_raises():
    events = [SimpleNamespace(type="response.output_text.delta", delta="a"),
              SimpleNamespace(type="response.failed",
                              response=SimpleNamespace(error="rate limited"))]
    with pytest.raises(RuntimeError, match="rate limited"):
        list(ChatModel("m", openai_client=fake_client(events=events), api="responses").stream_complete("q"))


class FakeChat:
    def __init__(self, chunks=None):
        self.calls, self.chunks = [], chunks or []

    def create(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            return iter(self.chunks)
        msg = SimpleNamespace(content="chat answer")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def chat_client(chunks=None):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeChat(chunks)))


def test_chat_completions_request_and_stream():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3,
                            prompt_tokens_details=None, completion_tokens_details=None)
    chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))], usage=None),
              SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" there"))], usage=None),
              SimpleNamespace(choices=[], usage=usage)]
    c = chat_client(chunks)
    m = ChatModel("gpt-4o-mini", temperature=0.1, max_output_tokens=2048, openai_client=c, api="chat")
    assert [d.delta for d in m.stream_complete("q")] == ["Hi", " there"]
    kw = c.chat.completions.calls[0]
    assert kw["messages"] == [{"role": "user", "content": "q"}] and kw["temperature"] == 0.1
    assert kw["max_completion_tokens"] == 2048 and "reasoning_effort" not in kw
    assert m.usage == [{"input": 10, "cached": 0, "output": 3, "reasoning": 0}]


def test_chat_reasoning_effort_for_luna():
    c = chat_client()
    out = ChatModel("gpt-6-luna", temperature=0.1, reasoning_effort="none", openai_client=c,
                    api="chat").complete("q")
    assert str(out) == "chat answer"
    assert c.chat.completions.calls[0]["reasoning_effort"] == "none"
