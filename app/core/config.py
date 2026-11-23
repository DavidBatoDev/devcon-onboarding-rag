"""Runtime configuration.

Locally: values come from server/.env. On Lambda (AWS_LAMBDA_FUNCTION_NAME is
set): plain settings come from the function's environment and secrets are read
once at cold start from SSM Parameter Store (SecureString) via the function's
IAM role — no secret is ever in the environment, the image, or Terraform state.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=SERVER_ROOT / ".env")

logger = logging.getLogger(__name__)

ON_LAMBDA = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))


def _ssm_secret(param_name: str) -> str:
    import boto3  # provided by the Lambda runtime; only needed on AWS

    resp = boto3.client("ssm").get_parameter(Name=param_name, WithDecryption=True)
    return resp["Parameter"]["Value"]


def _secret(env_name: str) -> str:
    """Secret from <NAME>_PARAM (SSM parameter name) on AWS, else from the env/.env."""
    param = os.getenv(f"{env_name}_PARAM")
    if param:
        try:
            return _ssm_secret(param)
        except Exception as e:  # surfaced on first use, not as an import crash
            logger.error("Could not read SSM parameter %s: %s", param, e)
            return ""
    return os.getenv(env_name, "")


class Settings:
    # --- Secrets
    OPENAI_API_KEY = _secret("OPENAI_API_KEY")

    # --- Models (OpenAI)
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    # Condense (follow-up -> standalone query) model; defaults to the answer model.
    CONDENSE_MODEL = os.getenv("CONDENSE_MODEL", "") or OPENAI_LLM_MODEL
    # Reasoning effort for reasoning models (gpt-6-luna: "none" keeps `temperature`
    # usable and adds no reasoning tokens). Empty = don't send the parameter
    # (required for non-reasoning models like gpt-4o-mini, which reject it).
    OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "")
    # Which OpenAI endpoint generates text: "chat" (Chat Completions — the endpoint
    # the v2 baseline was measured on, so the platform swap changes no model call)
    # or "responses" (Responses API).
    OPENAI_API_STYLE = os.getenv("OPENAI_API_STYLE", "chat")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
    OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
    OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
    # Eval only: the judge is pinned so it doesn't change with the answer model.
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

    # --- Knowledge-base index (see app/services/index_format.py)
    # S3 on AWS; a local directory in development.
    INDEX_BUCKET = os.getenv("INDEX_BUCKET", "")
    INDEX_PREFIX = os.getenv("INDEX_PREFIX", "index/")
    INDEX_LOCAL_DIR = os.getenv("INDEX_LOCAL_DIR", str(SERVER_ROOT / "app" / "storage" / "index"))
    # A warm instance re-checks index/current.json at most this often.
    INDEX_REFRESH_SECONDS = int(os.getenv("INDEX_REFRESH_SECONDS", "300"))

    # --- Retrieval (v2 pipeline)
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "20"))
    SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.25"))
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
    # Confidence gate (calibrated on the v2 vectors; the export keeps them identical):
    #   >= STRONG -> grounded; STRONG..WEAK -> grounded but hedged; < WEAK -> honest refusal.
    CONFIDENCE_STRONG = float(os.getenv("CONFIDENCE_STRONG", "0.50"))
    CONFIDENCE_WEAK = float(os.getenv("CONFIDENCE_WEAK", "0.35"))
    ENABLE_QUERY_CONDENSE = os.getenv("ENABLE_QUERY_CONDENSE", "true").lower() == "true"
    CONDENSE_HISTORY_TURNS = int(os.getenv("CONDENSE_HISTORY_TURNS", "6"))
    PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "v3")

    # --- Query cache (per instance)
    ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
    QUERY_CACHE_SIZE = int(os.getenv("QUERY_CACHE_SIZE", "256"))
    QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "900"))

    # --- Rate limits (requests per minute per client IP) and concurrency
    RATE_LIMIT_ASK = int(os.getenv("RATE_LIMIT_ASK_PER_MIN", "20"))
    RATE_LIMIT_RETRIEVE = int(os.getenv("RATE_LIMIT_RETRIEVE_PER_MIN", "10"))
    RATE_LIMIT_DEFAULT = int(os.getenv("RATE_LIMIT_DEFAULT_PER_MIN", "120"))
    # DynamoDB table for shared per-IP counters; unset -> in-process counters (dev/tests).
    RATELIMIT_TABLE = os.getenv("RATELIMIT_TABLE", "")
    MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "8"))
    QUERY_TIMEOUT_SECONDS = float(os.getenv("QUERY_TIMEOUT_SECONDS", "60"))

    # --- Feedback: DynamoDB table on AWS; unset -> local JSONL file.
    FEEDBACK_TABLE = os.getenv("FEEDBACK_TABLE", "")
    FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", str(SERVER_ROOT / "app" / "storage" / "feedback" / "feedback.jsonl"))


settings = Settings()
