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
    # New collection name: OpenAI embeddings are 1536-dim and incompatible with the
    # old 1024-dim BGE collection, so a fresh collection is required.
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "devcon_documents_openai_1536")

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

    # --- Smarter retrieval ---
    # Retrieve a wide candidate set, then rerank/filter down to TOP_K_RETRIEVAL.
    RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "20"))
    SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.3"))
    ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
    ENABLE_QUERY_CONDENSE = os.getenv("ENABLE_QUERY_CONDENSE", "true").lower() == "true"
    CONDENSE_HISTORY_TURNS = int(os.getenv("CONDENSE_HISTORY_TURNS", "6"))
    ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true"
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))

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