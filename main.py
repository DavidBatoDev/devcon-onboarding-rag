# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from app.api.v1.routes import router as v1_router
from app.core.limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import os
import uvicorn
import time
from contextlib import asynccontextmanager
from datetime import timedelta

# Lifespan events for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting DEVCON RAG API...")
    app.state.start_time = time.time()
    yield
    print("🔌 Shutting down...")

app = FastAPI(lifespan=lifespan)

# Rate limiting (SlowAPI). The limiter is applied per-route via decorators
# (see app/api/v1/routes.py); here we register it and its 429 handler.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
def root_check():
    return {"status": "ok", "message": "DEVCON RAG API"}

@app.get("/health")
def health_check():
    uptime = time.time() - app.state.start_time
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "uptime_human": str(timedelta(seconds=int(uptime)))
    }

# Configure CORS — restricted to the deployed frontend + local dev.
# Extra origins can be added via CORS_EXTRA_ORIGINS (comma-separated).
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
    # Vercel preview deployments live on *.vercel.app subdomains
    allow_origin_regex=r"https://devcon-onboarding-web[a-z0-9-]*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Admin-Token"],
)

# Register API routes
app.include_router(v1_router, prefix="/api/v1")

if __name__ == "__main__":
    # Use Render's port or default to 8000 locally
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=300
    )
