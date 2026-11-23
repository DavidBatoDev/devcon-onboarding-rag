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


# Coarse -> fine split patterns; each piece keeps its trailing delimiter/whitespace
# so joining pieces reproduces the text exactly.
_SPLIT_PATTERNS = (
    re.compile(r".+?(?:\n{2,}|$)", re.S),                       # paragraphs
    re.compile(r".+?(?:[.!?](?=\s)\s*|\n+|$)", re.S),           # sentences / lines
    re.compile(r".+?(?:[,;:](?=\s)\s*|$)", re.S),               # clauses
    re.compile(r"\S+\s*|\s+"),                                  # words
)


class SentenceSplitter:
    """Token-budgeted splitter for sections larger than one chunk (replaces
    LlamaIndex's SentenceSplitter): split recursively paragraph -> sentence ->
    clause -> word until every piece fits, then greedily merge pieces up to
    `chunk_size` tokens, carrying ~`chunk_overlap` tokens of trailing pieces
    into the next chunk."""

    def __init__(self, chunk_size: int = CHUNK_TOKENS, chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
                 ntokens: Optional[Callable[[str], int]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ntokens = ntokens or default_tokenizer()

    def _pieces(self, text: str, level: int = 0) -> List[str]:
        if self.ntokens(text) <= self.chunk_size or level >= len(_SPLIT_PATTERNS):
            if level >= len(_SPLIT_PATTERNS) and self.ntokens(text) > self.chunk_size:
                step = max(1, len(text) // (self.ntokens(text) // self.chunk_size + 1))
                return [text[i:i + step] for i in range(0, len(text), step)]
            return [text]
        parts = [p for p in _SPLIT_PATTERNS[level].findall(text) if p]
        if len(parts) <= 1:
            return self._pieces(text, level + 1)
        out: List[str] = []
        for p in parts:
            out.extend(self._pieces(p, level + 1) if self.ntokens(p) > self.chunk_size else [p])
        return out

    def split_text(self, text: str) -> List[str]:
        chunks: List[str] = []
        cur: List[Tuple[str, int]] = []
        cur_tokens = 0
        for piece in self._pieces(text):
            n = self.ntokens(piece)
            if cur and cur_tokens + n > self.chunk_size:
                chunks.append("".join(p for p, _ in cur).strip())
                overlap: List[Tuple[str, int]] = []
                total = 0
                for p, pn in reversed(cur):
                    if total + pn > self.chunk_overlap:
                        break
                    overlap.insert(0, (p, pn))
                    total += pn
                cur, cur_tokens = overlap, total
            cur.append((piece, n))
            cur_tokens += n
        if cur:
            chunks.append("".join(p for p, _ in cur).strip())
        return [c for c in chunks if c]


def clean_text(text: str) -> str:
    """Normalise extracted document text (docx2txt / table-export artifacts)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if s in ("nan", "nan:"):
            continue
        lines.append(re.sub(r"^nan:\s*", "", line).rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return text.strip() + "\n"


def chunk_document_text(text: str, ntokens: Optional[Callable[[str], int]] = None) -> List[Chunk]:
    """One-call helper: section-split + pack a cleaned document text."""
    ntokens = ntokens or default_tokenizer()
    splitter = SentenceSplitter(CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS, ntokens=ntokens)
    return chunk_sections(split_sections(text), splitter, ntokens)
