"""Section-aware chunking shared by the offline reprocess script and the
(local-only) Drive rebuild path, so every ingestion route produces the same
v2 chunk format: heading-labelled, token-budgeted, section-aligned.

docx2txt strips heading styles, but headings survive as short standalone
lines between blank lines — `is_heading` recovers them heuristically.
"""

import re
from typing import Callable, List, Optional, Tuple

CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

Section = Tuple[Optional[str], str]        # (heading | None, body text incl. heading line)
Chunk = Tuple[str, str]                    # (section label ("" if none), chunk text)


def is_heading(line: str, prev_blank: bool) -> bool:
    """Heuristic for docx2txt output: headings survive as short standalone lines."""
    s = line.strip()
    if not s or len(s) > 80 or len(s.split()) > 12:
        return False
    if re.match(r"^[-•*·▪│├└]|^\d+[.)]\s", s):    # bullets / list items / tree art
        return False
    if s[-1] in ".,;)" and not s.endswith(":"):    # reads like a sentence fragment
        return False
    words = [w for w in s.split() if w and w[0].isalpha()]
    titleish = bool(words) and sum(w[0].isupper() for w in words) >= max(1, len(words) // 2)
    return prev_blank and (titleish or s.isupper() or s.endswith(":"))


def split_sections(text: str) -> List[Section]:
    """Split into [(heading|None, section_text)] — heading line stays in its section."""
    lines = text.split("\n")
    sections: List[Section] = []
    cur_heading, cur_lines = None, []
    prev_blank = True
    for line in lines:
        if is_heading(line, prev_blank):
            body = "\n".join(cur_lines).strip()
            if body:
                sections.append((cur_heading, body))
            cur_heading, cur_lines = line.strip().rstrip(":"), [line]
        else:
            cur_lines.append(line)
        prev_blank = line.strip() == ""
    body = "\n".join(cur_lines).strip()
    if body:
        sections.append((cur_heading, body))
    return sections


def chunk_sections(sections: List[Section], splitter, ntokens: Callable[[str], int]) -> List[Chunk]:
    """Greedily pack consecutive sections into ~CHUNK_TOKENS chunks.

    Chunks always START at a section boundary but may span several small
    sections (table-heavy docs produce hundreds of tiny 'sections' — packing
    keeps chunks semantically dense instead of fragmenting them). A section
    larger than the budget is split with the sentence splitter on its own.
    Label = heading of the first labelled section in the pack.
    """
    out: List[Chunk] = []

    def flush(pack):
        if not pack:
            return
        label = next((h for h, _ in pack if h), "")
        text = "\n\n".join(b for _, b in pack).strip()
        if text:
            out.append((label, text))

    pack, pack_tokens = [], 0
    for heading, body in sections:
        t = ntokens(body)
        if t > CHUNK_TOKENS:
            flush(pack)
            pack, pack_tokens = [], 0
            for piece in splitter.split_text(body):
                piece = piece.strip()
                if piece:
                    out.append((heading or "", piece))
        elif pack and pack_tokens + t > CHUNK_TOKENS:
            flush(pack)
            pack, pack_tokens = [(heading, body)], t
        else:
            pack.append((heading, body))
            pack_tokens += t
    flush(pack)
    return out


def default_tokenizer() -> Callable[[str], int]:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return lambda s: len(enc.encode(s, disallowed_special=()))


def chunk_document_text(text: str) -> List[Chunk]:
    """One-call helper: section-split + pack a cleaned document text."""
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS)
    return chunk_sections(split_sections(text), splitter, default_tokenizer())
