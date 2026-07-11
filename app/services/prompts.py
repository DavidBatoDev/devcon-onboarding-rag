"""All DEBBIE prompt templates in one place — the single source of truth for
both streaming and non-streaming answers (the old code had two divergent
template systems; this module replaces `prompt_engine.py` and the inline QA
templates entirely, including the LLM-emitted SHOW_REFERENCES marker).
"""

from typing import Dict, List, Optional

PERSONA = (
    "You are DEBBIE, the DEVCON Officers' Onboarding Assistant. You help DEVCON "
    "chapter officers with onboarding, chapter management, and events. Reply in "
    "clear, friendly markdown with practical, actionable advice (minimal emojis)."
)


def format_history(history: Optional[List[Dict[str, str]]], max_turns: int = 10) -> str:
    if not history:
        return ""
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-max_turns:]
    )


def build_condense_prompt(question: str, convo: str) -> str:
    return (
        "Given the conversation below and a follow-up message, rewrite the follow-up "
        "as a standalone, self-contained search query. Resolve pronouns and references "
        "using the conversation. Return ONLY the rewritten query, nothing else.\n\n"
        f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone query:"
    )


def build_answer_prompt(
    question: str,
    sources: List[Dict[str, str]],   # [{"title","section","text"}] in citation order
    history_block: str = "",
    weak: bool = False,
) -> str:
    """The single grounded-answer template. Inline [n] citations are validated
    deterministically afterwards (see rag_pipeline.sanitize_citations)."""
    numbered = []
    for i, s in enumerate(sources, 1):
        loc = f" — {s['section']}" if s.get("section") else ""
        numbered.append(f"[{i}] {s['title']}{loc}\n{s['text']}")
    sources_block = "\n\n".join(numbered)

    parts = [PERSONA]
    if history_block:
        parts.append(
            "Recent conversation (do not reintroduce yourself; answer the new "
            f"question directly):\n{history_block}"
        )
    parts.append(
        "Numbered source excerpts from official DEVCON documents:\n\n"
        f"{sources_block}"
    )
    rules = [
        "Answer ONLY from the sources above. Cite every factual claim with its "
        'source number in brackets, e.g. "Submit the report within 30 days [2]." '
        f"Use only numbers 1 to {len(sources)}. Never invent a citation number.",
        "If the sources only partially answer the question, answer what they "
        "cover and say plainly what they do not cover.",
        "Do not add a sources or references section at the end; the app displays "
        "sources automatically.",
    ]
    if weak:
        rules.insert(1, (
            "The retrieved sources are only loosely related to this question. Be "
            "explicit about what the documents do NOT cover, and keep any advice "
            "beyond them clearly labelled as general guidance."
        ))
    parts.append("Rules:\n" + "\n".join(f"- {r}" for r in rules))
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def build_general_prompt(question: str, history_block: str = "") -> str:
    """Answer path when the documents don't cover the question (or it's small
    talk). Honesty is enforced deterministically by the confidence gate — this
    template only shapes tone."""
    prompt = (
        f"{PERSONA}\n"
        "- For greetings or small talk, respond naturally and invite the user to "
        "ask about their officer role.\n"
        "- For DEVCON-specific questions: the official documents available to you "
        "do not cover this topic — say so plainly first, then (if helpful) give "
        "clearly-labelled general guidance and suggest checking with DEVCON HQ "
        "(hello@devcon.ph).\n"
        "- Never invent specific DEVCON facts, names, dates, or policies.\n"
        "- If the conversation is already ongoing, do not reintroduce yourself.\n"
    )
    if history_block:
        prompt += f"\nRecent conversation:\n{history_block}\n"
    prompt += f"\nUser: {question}\nDEBBIE:"
    return prompt
