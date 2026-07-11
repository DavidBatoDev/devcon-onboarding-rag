# backend/app/core/config.py
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from the server/ directory (config.py is at server/app/core/config.py,
# so parents[2] == server/). This makes config load reliably regardless of CWD.
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Google Drive Configuration - Method 3: Individual Variables
    GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
    GOOGLE_CLIENT_EMAIL = os.getenv("GOOGLE_CLIENT_EMAIL")
    GOOGLE_PRIVATE_KEY_ID = os.getenv("GOOGLE_PRIVATE_KEY_ID")
    GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY")
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    
    # Google Drive Configuration - Fallback methods (keep for compatibility)
    GOOGLE_SERVICE_ACCOUNT_JSON_DATA = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_DATA")
    GOOGLE_SERVICE_ACCOUNT_JSON_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_FILE") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    
    # RAG Configuration
    RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "app/storage/rag_index")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    
    # LLM Configuration (OpenAI)
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
    OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
    OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "app/storage/chroma_db")
    # v2 collection: section-aware 512-token chunks, cosine space. The legacy
    # l2 collection (devcon_documents_openai_1536) is kept in the same store as
    # a rollback lever — flip this env var to revert without a redeploy.
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "devcon_chunks_v2")

    # Embedding Configuration (OpenAI)
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    # Performance Settings
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

    # --- Rate limiting & resilience ---
    RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "120/minute")
    RATE_LIMIT_ASK = os.getenv("RATE_LIMIT_ASK", "20/minute")
    MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "8"))
    QUERY_TIMEOUT_SECONDS = float(os.getenv("QUERY_TIMEOUT_SECONDS", "60"))
    DRIVE_MAX_RETRIES = int(os.getenv("DRIVE_MAX_RETRIES", "5"))

    # --- Smarter retrieval (v2 pipeline) ---
    # Retrieve a wide candidate set (vector + BM25, RRF-fused), gate on the best
    # raw cosine similarity, then keep TOP_K_RETRIEVAL for synthesis.
    RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "20"))
    # Pre-fusion floor on the vector leg (cosine similarity).
    SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.25"))
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
    # Confidence gate tiers (calibrated via scripts/inspect_retrieval.py + eval):
    #   >= STRONG -> grounded; STRONG..WEAK -> grounded but hedged; < WEAK -> honest refusal.
    CONFIDENCE_STRONG = float(os.getenv("CONFIDENCE_STRONG", "0.50"))
    CONFIDENCE_WEAK = float(os.getenv("CONFIDENCE_WEAK", "0.35"))
    # LLM rerank: "off" | "nonstream" | "always". Default off — it adds seconds of
    # latency; the eval harness decides whether it earns its keep.
    RERANK_MODE = os.getenv("RERANK_MODE", "off")
    ENABLE_QUERY_CONDENSE = os.getenv("ENABLE_QUERY_CONDENSE", "true").lower() == "true"
    CONDENSE_HISTORY_TURNS = int(os.getenv("CONDENSE_HISTORY_TURNS", "6"))
    PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "v2")
    KB_VERSION = os.getenv("KB_VERSION", "2025-06")
    # Admin gate for the local-only Drive endpoints (unset -> endpoints 403).
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
    RATE_LIMIT_RETRIEVE = os.getenv("RATE_LIMIT_RETRIEVE", "10/minute")

    # --- Caching ---
    ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
    QUERY_CACHE_SIZE = int(os.getenv("QUERY_CACHE_SIZE", "256"))
    QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "900"))  # seconds

    # API Configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    VITE_API_URL = os.getenv("VITE_API_URL", "http://localhost:8000/api/v1/ask")
    
    @property
    def google_service_account_configured(self) -> bool:
        """Check if Google service account is properly configured using any method"""
        # Method 3: Individual variables
        method3_vars = all([
            self.GOOGLE_PROJECT_ID,
            self.GOOGLE_CLIENT_EMAIL,
            self.GOOGLE_PRIVATE_KEY_ID,
            self.GOOGLE_PRIVATE_KEY,
            self.GOOGLE_CLIENT_ID
        ])
        
        # Method 1: JSON data
        method1_configured = bool(self.GOOGLE_SERVICE_ACCOUNT_JSON_DATA)
        
        # Method 2: JSON file
        method2_configured = bool(self.GOOGLE_SERVICE_ACCOUNT_JSON_FILE and 
                                 os.path.exists(self.GOOGLE_SERVICE_ACCOUNT_JSON_FILE))
        
        return method3_vars or method1_configured or method2_configured


settings = Settings()