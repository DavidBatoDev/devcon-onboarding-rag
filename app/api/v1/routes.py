# backend/app/api/v1/routes.py
import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

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
from app.core.limiter import limiter
from app.services.llamaindex_service import rag_service
from app.services.rag_pipeline import legacy_answer_text

logger = logging.getLogger(__name__)
router = APIRouter()

# Cap how many queries hit the LLM/embedding APIs concurrently so bursts of
# traffic don't overwhelm the provider (or our rate limits).
_query_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUERIES)

FEEDBACK_FILE = Path("app/storage/feedback/feedback.jsonl")


def require_admin(x_admin_token: Optional[str] = Header(None)):
    """Gate for local-admin endpoints (Drive rebuild, index clear). These need
    Google credentials that only exist locally; the gate keeps them inert on
    the public deployment (ADMIN_TOKEN unset -> always 403)."""
    if not settings.ADMIN_TOKEN or x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin token required")


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


@router.post("/ask", response_model=AskResponse)
@limiter.limit(settings.RATE_LIMIT_ASK)
async def ask_question(request: Request, body: AskRequest):
    """Ask a question. `structured=true` returns v2 metadata (sources,
    confidence, timings); otherwise the sources footer is appended to the
    answer text for legacy clients."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not rag_service.ready:
        raise HTTPException(status_code=503, detail="RAG index is empty or not initialized")

    try:
        async with _query_semaphore:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    rag_service.query, body.query, _history(body),
                    body.top_k, body.temperature, body.nocache,
                ),
                timeout=settings.QUERY_TIMEOUT_SECONDS,
            )
        if body.structured:
            return AskResponse(answer=result.answer,
                               metadata=AskMetadata(**result.metadata_dict()))
        return AskResponse(answer=legacy_answer_text(result))
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504,
                            detail="The request took too long to process. Please try again.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing query: %s", e)
        raise HTTPException(status_code=500, detail="Error processing query")


@router.post("/ask/stream")
@limiter.limit(settings.RATE_LIMIT_ASK)
async def ask_question_stream(request: Request, body: AskRequest):
    """Streaming ask. Content negotiation:
    - Accept: application/x-ndjson  -> typed NDJSON events
      {"type":"meta"|"sources"|"token"|"done"|"error", ...} one per line
    - anything else                 -> legacy plain-text token stream
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not rag_service.ready:
        raise HTTPException(status_code=503, detail="RAG index is empty or not initialized")

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
                # Legacy footer from structured sources (no Beta footer in v2).
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

@router.post("/retrieve", response_model=RetrieveResponse)
@limiter.limit(settings.RATE_LIMIT_RETRIEVE)
async def retrieve_debug(request: Request, body: RetrieveRequest):
    """Retrieval inspector (powers the RAG Playground): hybrid candidates with
    per-leg scores (cosine / BM25 rank / RRF) and the confidence-gate tier."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not rag_service.ready:
        raise HTTPException(status_code=503, detail="RAG index is empty or not initialized")
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


@router.post("/feedback")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Thumbs feedback. Durable copy = the structured log line (Cloud Logging);
    the JSONL file is a local-dev convenience (Cloud Run FS is ephemeral)."""
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "request_id": body.request_id,
        "rating": body.rating,
        "comment": body.comment,
        "query": body.query,
    }
    logger.info(json.dumps({"event": "feedback", **record}))
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Could not append feedback file: %s", e)
    return {"status": "ok"}


@router.get("/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    try:
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
        return {
            "status": "healthy",
            "rag_status": stats.get("status", "unknown"),
            "document_count": stats.get("document_count", 0),
            "rebuild_status": rebuild_state["last_status"],
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e),
                "rebuild_status": rebuild_state.get("last_status", "unknown")}


# ------------------------------------------------- admin (local-only, gated)
# These endpoints require Google Drive credentials and X-Admin-Token. On the
# public deployment ADMIN_TOKEN is unset so they always 403.

class FetchFilesRequest(BaseModel):
    folder_id: Optional[str] = None


class DriveFileInfo(BaseModel):
    id: str
    name: str
    mime_type: str
    size: Optional[int] = None
    modified_time: Optional[str] = None
    status: str = "available"


class FetchFilesResponse(BaseModel):
    status: str
    message: str
    total_files: int = 0
    supported_files: int = 0
    files: List[DriveFileInfo] = []
    processing_time: float = 0


class RebuildRequest(BaseModel):
    folder_id: Optional[str] = None
    batch_size: int = 25
    file_ids: Optional[List[str]] = None
    incremental: bool = False


class RebuildResponse(BaseModel):
    status: str
    message: str
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    processing_time: float = 0
    files_details: List[Dict[str, Any]] = []


rebuild_state = {
    "in_progress": False,
    "last_started": None,
    "last_completed": None,
    "last_duration": None,
    "last_status": None,
}
rebuild_lock = threading.Lock()

DEFAULT_FOLDER_ID = "1eocL8T8BH6EwnP5siOtDz3FG2CqGHveS"


@router.post("/files/fetch", response_model=FetchFilesResponse)
async def fetch_drive_files(request: FetchFilesRequest,
                            x_admin_token: Optional[str] = Header(None)):
    """List Google Drive files without processing them (admin, local-only)."""
    require_admin(x_admin_token)
    from app.services.drive_loader import get_drive_files

    try:
        start_time = time.time()
        folder_id = request.folder_id or DEFAULT_FOLDER_ID
        files = get_drive_files(folder_id)
        drive_files = [
            DriveFileInfo(
                id=f.get("id", ""),
                name=f.get("name", "Unknown"),
                mime_type=f.get("mimeType", "Unknown"),
                size=int(f["size"]) if f.get("size") else None,
                modified_time=f.get("modifiedTime", ""),
            )
            for f in files
        ]
        return FetchFilesResponse(
            status="success",
            message=f"Successfully fetched {len(files)} files from Google Drive",
            total_files=len(files),
            supported_files=len(files),
            files=drive_files,
            processing_time=time.time() - start_time,
        )
    except Exception as e:
        logger.exception("Error fetching files from Google Drive: %s", e)
        raise HTTPException(status_code=500, detail=f"Error fetching files: {e}")


@router.get("/files/list")
async def list_drive_files(folder_id: Optional[str] = None,
                           x_admin_token: Optional[str] = Header(None)):
    request = FetchFilesRequest(folder_id=folder_id)
    return await fetch_drive_files(request, x_admin_token)


@router.get("/rebuild/status")
async def get_rebuild_status():
    return rebuild_state


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_index(request: RebuildRequest,
                        x_admin_token: Optional[str] = Header(None)):
    """Rebuild the index from Google Drive (admin, local-only). Produces
    v2-format chunks via the shared section-aware chunker."""
    require_admin(x_admin_token)
    from app.services.document_processor import document_processor
    from app.services.drive_loader import get_drive_files

    with rebuild_lock:
        if rebuild_state["in_progress"]:
            return RebuildResponse(status="already_running",
                                   message="Rebuild is already in progress")
        rebuild_state.update(in_progress=True, last_started=str(datetime.now()),
                             last_completed=None, last_duration=None,
                             last_status="in_progress")

    start_time = time.time()
    try:
        folder_id = request.folder_id or DEFAULT_FOLDER_ID
        all_files = get_drive_files(folder_id)
        files = ([f for f in all_files if f.get("id") in request.file_ids]
                 if request.file_ids else all_files)

        if not files:
            rebuild_state["last_status"] = "completed_no_files"
            return RebuildResponse(status="completed", message="No files found to process",
                                   processing_time=time.time() - start_time)

        if request.incremental:
            summary = rag_service.incremental_update(files)
            rebuild_state["last_status"] = "completed_incremental"
            return RebuildResponse(
                status="completed",
                message=(f"Incremental update: {summary.get('added', 0)} added, "
                         f"{summary.get('changed', 0)} changed, {summary.get('skipped', 0)} skipped, "
                         f"{summary.get('deleted', 0)} deleted"),
                processed_files=summary.get("added", 0) + summary.get("changed", 0),
                failed_files=summary.get("failed", 0),
                processing_time=time.time() - start_time,
            )

        documents = document_processor.process_drive_files(files)
        if not documents:
            rebuild_state["last_status"] = "completed_no_documents"
            return RebuildResponse(status="completed",
                                   message="No documents were successfully processed",
                                   failed_files=len(files),
                                   processing_time=time.time() - start_time)

        success = rag_service.rebuild_index_from_scratch(documents)
        rebuild_state["last_status"] = "completed_success" if success else "failed_rebuild"
        stats = rag_service.get_index_stats()
        return RebuildResponse(
            status="completed" if success else "error",
            message=(f"Successfully rebuilt index with {len(documents)} documents"
                     if success else "Failed to rebuild index"),
            processed_files=len(documents),
            failed_files=len(files) - len(documents),
            total_chunks=stats.get("document_count", 0),
            processing_time=time.time() - start_time,
        )
    except Exception as e:
        logger.exception("Error during index rebuild: %s", e)
        rebuild_state["last_status"] = f"error: {e}"
        raise HTTPException(status_code=500, detail=f"Error during rebuild: {e}")
    finally:
        with rebuild_lock:
            rebuild_state.update(in_progress=False, last_completed=str(datetime.now()),
                                 last_duration=time.time() - start_time)


@router.delete("/index")
async def clear_index(x_admin_token: Optional[str] = Header(None)):
    """Clear the configured collection (admin only — this would otherwise let
    anyone wipe the baked index on the public deployment)."""
    require_admin(x_admin_token)
    success = rag_service.clear_index()
    if success:
        return {"status": "success", "message": "Index cleared successfully"}
    return {"status": "error", "message": "Failed to clear index"}
