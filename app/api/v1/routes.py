import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.api.v1.models import (
    AskMetadata,
    AskRequest,
    AskResponse,
    FeedbackRequest,
    IndexStatsResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.core.config import settings
from app.core.ratelimit import limit
from app.services import feedback_store
from app.services.rag_pipeline import legacy_answer_text
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Cap concurrent LLM/embedding work per instance. On Lambda each instance serves
# one request at a time anyway; reserved concurrency is the real ceiling.
_query_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUERIES)


async def _ready_or_503():
    await run_in_threadpool(rag_service.refresh)
    if not rag_service.ready:
        raise HTTPException(status_code=503, detail="RAG index is empty or not initialized")


class ServerStatusResponse(BaseModel):
    server_up: bool
    server_running: bool
    message: str


@router.get("/status", response_model=ServerStatusResponse)
async def get_server_status():
    return ServerStatusResponse(server_up=True, server_running=True,
                                message="Server is up and running")


# --------------------------------------------------------------------- ask

def _history(body: AskRequest) -> List[Dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in body.history]


@router.post("/ask", response_model=AskResponse, dependencies=[limit("ask", "RATE_LIMIT_ASK")])
async def ask_question(request: Request, body: AskRequest):
    """Ask a question. `structured=true` returns v2 metadata (sources,
    confidence, timings); otherwise the sources footer is appended to the
    answer text for legacy clients."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    await _ready_or_503()
    try:
        async with _query_semaphore:
            result = await asyncio.wait_for(
                asyncio.to_thread(rag_service.query, body.query, _history(body),
                                  body.top_k, body.temperature, body.nocache),
                timeout=settings.QUERY_TIMEOUT_SECONDS,
            )
        if body.structured:
            return AskResponse(answer=result.answer, metadata=AskMetadata(**result.metadata_dict()))
        return AskResponse(answer=legacy_answer_text(result))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504,
                            detail="The request took too long to process. Please try again.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing query: %s", e)
        raise HTTPException(status_code=500, detail="Error processing query")


@router.post("/ask/stream", dependencies=[limit("ask", "RATE_LIMIT_ASK")])
async def ask_question_stream(request: Request, body: AskRequest):
    """Streaming ask. Content negotiation:
    - Accept: application/x-ndjson  -> typed NDJSON events
      {"type":"meta"|"sources"|"token"|"done"|"error", ...} one per line
    - anything else                 -> legacy plain-text token stream
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    await _ready_or_503()

    wants_ndjson = "application/x-ndjson" in (request.headers.get("accept") or "")
    history = _history(body)

    def sync_events():
        yield from rag_service.query_stream(body.query, history,
                                            body.top_k, body.temperature, body.nocache)

    async def ndjson_stream():
        async with _query_semaphore:
            try:
                async for kind, payload in iterate_in_threadpool(sync_events()):
                    if kind == "done":
                        line = {"type": "done", "answer": payload.answer,
                                "cited": [s.n for s in payload.sources if s.cited],
                                "timings": payload.timings}
                    elif kind in ("meta", "sources"):
                        line = {"type": kind, **payload}
                    else:  # token / error
                        line = {"type": kind,
                                "text" if kind == "token" else "message": payload}
                    yield json.dumps(line, ensure_ascii=False) + "\n"
            except Exception as e:
                logger.exception("Error during NDJSON stream: %s", e)
                yield json.dumps({"type": "error",
                                  "message": "The response was interrupted. Please try again."}) + "\n"

    async def legacy_stream():
        async with _query_semaphore:
            try:
                sources_payload = None
                async for kind, payload in iterate_in_threadpool(sync_events()):
                    if kind == "token":
                        yield payload
                    elif kind == "sources":
                        sources_payload = payload
                    elif kind == "error":
                        yield f"\n\n❌ {payload}"
                if sources_payload and sources_payload.get("sources"):
                    seen, lines = set(), []
                    for s in sources_payload["sources"]:
                        key = (s["title"], s.get("drive_link", ""))
                        if key in seen:
                            continue
                        seen.add(key)
                        lines.append(f"{len(seen)}. {s['title']}"
                                     + (f" — {s['section']}" if s.get("section") else ""))
                        if s.get("drive_link"):
                            lines.append(f"   [[View Document]]({s['drive_link']})")
                        if len(seen) >= 3:
                            break
                    if lines:
                        yield "\n\n📚 **Sources Referenced:**\n" + "\n".join(lines)
            except Exception as e:
                logger.exception("Error during legacy stream: %s", e)
                yield "\n\n❌ The response was interrupted. Please try again."

    media = "application/x-ndjson" if wants_ndjson else "text/plain; charset=utf-8"
    return StreamingResponse(
        ndjson_stream() if wants_ndjson else legacy_stream(),
        media_type=media,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- playground

@router.post("/retrieve", response_model=RetrieveResponse,
             dependencies=[limit("retrieve", "RATE_LIMIT_RETRIEVE")])
async def retrieve_debug(request: Request, body: RetrieveRequest):
    """Retrieval inspector (powers the RAG Playground): hybrid candidates with
    per-leg scores (cosine / BM25 rank / RRF) and the confidence-gate tier."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    await _ready_or_503()
    try:
        async with _query_semaphore:
            result = await asyncio.wait_for(
                asyncio.to_thread(rag_service.retrieve_debug, body.query, body.top_k),
                timeout=settings.QUERY_TIMEOUT_SECONDS,
            )
        return RetrieveResponse(**result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Retrieval took too long")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in retrieve: %s", e)
        raise HTTPException(status_code=500, detail="Error running retrieval")


@router.post("/feedback", dependencies=[limit("feedback", "RATE_LIMIT_DEFAULT")])
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Thumbs feedback -> DynamoDB on AWS (JSONL locally) + a structured log line."""
    record = feedback_store.new_record(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        request_id=body.request_id, rating=body.rating,
        comment=body.comment, query=body.query,
    )
    logger.info(json.dumps({"event": "feedback", **record}, ensure_ascii=False))
    try:
        await run_in_threadpool(feedback_store.store().save, record)
    except Exception as e:
        logger.error("Could not store feedback %s: %s", record["id"], e)
        raise HTTPException(status_code=503, detail="Feedback could not be saved")
    return {"status": "ok", "id": record["id"]}


@router.get("/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    try:
        await run_in_threadpool(rag_service.refresh)
        stats = rag_service.get_index_stats()
        return IndexStatsResponse(**{k: v for k, v in stats.items()
                                     if k in IndexStatsResponse.model_fields})
    except Exception as e:
        logger.error("Error getting index stats: %s", e)
        return IndexStatsResponse(status="error", document_count=0, error=str(e))


@router.get("/health")
async def health_check():
    try:
        stats = rag_service.get_index_stats()
        return {"status": "healthy" if rag_service.ready else "degraded",
                "rag_status": stats.get("status", "unknown"),
                "document_count": stats.get("document_count", 0),
                "index_version": stats.get("index_version"),
                "model": settings.OPENAI_LLM_MODEL}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
