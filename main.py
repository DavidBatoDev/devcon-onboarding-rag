"""DEBBIE API entrypoint.

Runs unchanged in two places:
- locally:      python main.py  (or uvicorn main:app)
- AWS Lambda:   behind API Gateway via Lambda Web Adapter (run.sh starts uvicorn;
                AWS_LWA_INVOKE_MODE=response_stream keeps NDJSON token streaming).
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import timedelta

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routes import router as v1_router
from app.core.config import ON_LAMBDA, settings
from app.services.rag_service import rag_service


class JsonFormatter(logging.Formatter):
    """One JSON object per line -> queryable with CloudWatch Logs Insights.
    Messages that are already JSON (pipeline events) are merged, not nested."""

    def format(self, record):
        base = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                "logger": record.name}
        msg = record.getMessage()
        try:
            parsed = json.loads(msg) if msg.startswith("{") else None
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            base.update(parsed)
        else:
            base["message"] = msg
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def _setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if ON_LAMBDA else
                         logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    for noisy in ("httpx", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger("debbie")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    try:
        rag_service.startup()  # cold start: load the index once
    except Exception as e:     # keep serving /status; /ask returns 503 until it loads
        logger.exception("Index load failed at startup: %s", e)
    logger.info(json.dumps({"event": "startup", "model": settings.OPENAI_LLM_MODEL,
                            "index_version": rag_service.store.version if rag_service.ready else None,
                            "startup_ms": int((time.time() - app.state.start_time) * 1000)}))
    yield


app = FastAPI(title="DEBBIE RAG API", lifespan=lifespan)


@app.get("/")
def root_check():
    return {"status": "ok", "message": "DEVCON RAG API"}


@app.get("/health")
def health_check():
    uptime = time.time() - app.state.start_time
    return {"status": "ok", "uptime_seconds": round(uptime, 1),
            "uptime_human": str(timedelta(seconds=int(uptime)))}


# CORS — restricted to the deployed frontend, its Vercel previews, and local dev.
# Extra origins via CORS_EXTRA_ORIGINS (comma-separated). On AWS this middleware
# is the only CORS layer: its headers and the OPTIONS preflight pass through
# API Gateway's STREAM proxy unchanged (verified in the Phase 0 spike).
_origins = [
    "https://devcon-onboarding-web.vercel.app",
    "http://localhost:8080",
    "http://localhost:8081",
    "http://localhost:5173",
]
_origins += [o.strip() for o in os.environ.get("CORS_EXTRA_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://devcon-onboarding-web[a-z0-9-]*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

app.include_router(v1_router, prefix="/api/v1")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, timeout_keep_alive=300)
