"""OpenAI adapter: the `.complete / .stream_complete` interface the answer
pipeline expects, implemented on the Responses API (no LlamaIndex).

Reasoning models (gpt-6-luna) reject `temperature` unless reasoning effort is
"none" (verified in Phase 0), so the effort is sent whenever it is configured
and temperature is sent alongside it. Non-reasoning models (gpt-4o-mini)
reject the `reasoning` parameter, so it is omitted when unset.
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional

from openai import OpenAI

from app.core.config import settings

_client: Optional[OpenAI] = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY or None,
                         timeout=settings.OPENAI_REQUEST_TIMEOUT,
                         max_retries=settings.OPENAI_MAX_RETRIES)
    return _client


@dataclass
class Completion:
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass
class Delta:
    delta: str


class ChatModel:
    def __init__(self, model: str, temperature: Optional[float] = None,
                 max_output_tokens: Optional[int] = None,
                 reasoning_effort: Optional[str] = None, openai_client=None,
                 api: Optional[str] = None):
        self.api = api or settings.OPENAI_API_STYLE   # "responses" | "chat"
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort or None
        self._client = openai_client
        self.usage: List[dict] = []   # per-call token usage (eval cost accounting)

    def _record(self, usage) -> None:
        if usage is None:
            return
        details = getattr(usage, "output_tokens_details", None)
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        self.usage.append({"input": getattr(usage, "input_tokens", 0) or 0, "cached": cached,
                           "output": getattr(usage, "output_tokens", 0) or 0,
                           "reasoning": getattr(details, "reasoning_tokens", 0) or 0})

    def _kwargs(self, prompt: str) -> dict:
        kw = {"model": self.model, "input": prompt}
        if self.max_output_tokens:
            kw["max_output_tokens"] = self.max_output_tokens
        if self.reasoning_effort:
            kw["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        return kw

    @property
    def _api(self):
        return self._client or client()

    def _chat_kwargs(self, prompt: str) -> dict:
        kw = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if self.max_output_tokens:
            kw["max_completion_tokens"] = self.max_output_tokens
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        return kw

    def _record_chat(self, usage) -> None:
        if usage is None:
            return
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        self.usage.append({"input": usage.prompt_tokens, "cached": cached,
                           "output": usage.completion_tokens, "reasoning": reasoning})

    def complete(self, prompt: str) -> Completion:
        if self.api == "chat":
            resp = self._api.chat.completions.create(**self._chat_kwargs(prompt))
            self._record_chat(getattr(resp, "usage", None))
            return Completion(resp.choices[0].message.content or "")
        resp = self._api.responses.create(**self._kwargs(prompt))
        self._record(getattr(resp, "usage", None))
        return Completion(resp.output_text or "")

    def stream_complete(self, prompt: str) -> Iterator[Delta]:
        if self.api == "chat":
            stream = self._api.chat.completions.create(**self._chat_kwargs(prompt), stream=True,
                                                       stream_options={"include_usage": True})
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield Delta(chunk.choices[0].delta.content)
                if getattr(chunk, "usage", None):
                    self._record_chat(chunk.usage)
            return
        stream = self._api.responses.create(**self._kwargs(prompt), stream=True)
        for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                yield Delta(event.delta)
            elif etype == "response.completed":
                self._record(getattr(getattr(event, "response", None), "usage", None))
            elif etype in ("response.failed", "error"):
                msg = getattr(getattr(event, "response", None), "error", None) or getattr(event, "message", "")
                raise RuntimeError(f"OpenAI stream failed: {msg}")


def answer_model(temperature: Optional[float] = None) -> ChatModel:
    return ChatModel(settings.OPENAI_LLM_MODEL,
                     temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
                     max_output_tokens=settings.MAX_OUTPUT_TOKENS,
                     reasoning_effort=settings.OPENAI_REASONING_EFFORT)


def condense_model() -> ChatModel:
    return ChatModel(settings.CONDENSE_MODEL, temperature=settings.LLM_TEMPERATURE,
                     max_output_tokens=256, reasoning_effort=settings.OPENAI_REASONING_EFFORT)


def embed_query(text: str) -> List[float]:
    # Same input normalisation as the LlamaIndex embedding path the v2 index was
    # built and calibrated with (newlines -> spaces), so query vectors match.
    text = text.replace("\n", " ")
    resp = client().embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ") for t in texts[i:i + batch_size]]
        resp = client().embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=batch)
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out
