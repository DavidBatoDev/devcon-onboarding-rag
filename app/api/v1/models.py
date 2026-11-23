from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask the RAG system")
    history: List[ChatMessage] = []
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of relevant chunks to retrieve")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature for response generation")
    structured: bool = Field(False, description="Return structured metadata (v2 clients); when false the sources footer is appended to the answer text for legacy clients")
    nocache: bool = Field(False, description="Bypass the query cache (used by Regenerate)")


class SourceRefModel(BaseModel):
    n: int
    title: str
    section: str = ""
    snippet: str
    text: str
    drive_link: str = ""
    score: Optional[float] = None
    chunk_id: str
    cited: bool = False


class AskMetadata(BaseModel):
    grounded: bool
    confidence: str                      # strong | weak | none
    sources: List[SourceRefModel] = []
    timings: Dict[str, int] = {}
    model: str = ""
    request_id: str = ""
    kb_version: str = ""
    cached: bool = False


class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    metadata: Optional[AskMetadata] = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(None, ge=1, le=20)


class RetrieveCandidate(BaseModel):
    final_rank: int
    chunk_id: str
    title: str
    section: str = ""
    text: str
    cosine: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf: float


class RetrieveResponse(BaseModel):
    query: str
    tier: str
    thresholds: Dict[str, float]
    candidates: List[RetrieveCandidate]


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., max_length=64)
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = Field(None, max_length=2000)
    query: Optional[str] = Field(None, max_length=1000)


class IndexStatsResponse(BaseModel):
    status: str
    document_count: int
    index_version: Optional[str] = None
    index_created: Optional[str] = None
    collection_name: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k_retrieval: Optional[int] = None
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    kb_version: Optional[str] = None
    hybrid_search: Optional[bool] = None
    rerank_mode: Optional[str] = None
    documents: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
