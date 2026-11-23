"""Per-client-IP rate limiting, shared across Lambda instances via DynamoDB.

Layering on AWS: API Gateway stage/method throttles (global) -> this per-IP
fixed-window counter -> Lambda reserved concurrency (absolute ceiling).

Counter key: "<scope>#<hashed ip>#<minute>", atomic `ADD count :1`, expired by
DynamoDB TTL a couple of minutes later. The IP is hashed so the table never
holds a raw address. If DynamoDB is unavailable the limiter fails OPEN (logged):
availability wins, and reserved concurrency still caps spend.

Client IP on Lambda = `identity.sourceIp` from the `x-amzn-request-context`
header that Lambda Web Adapter builds from the API Gateway event — never
X-Forwarded-For, which the client controls.
"""

import hashlib
import json
import logging
import threading
import time
from typing import Dict, Tuple

from fastapi import Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.core.config import ON_LAMBDA, settings

logger = logging.getLogger(__name__)
WINDOW_SECONDS = 60


def client_ip(request: Request) -> str:
    if ON_LAMBDA:
        raw = request.headers.get("x-amzn-request-context")
        if raw:
            try:
                ip = (json.loads(raw).get("identity") or {}).get("sourceIp")
                if ip:
                    return ip
            except ValueError:
                pass
    return request.client.host if request.client else "unknown"


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(f"debbie|{ip}".encode()).hexdigest()[:20]


class MemoryCounter:
    """Single-process counter for local dev and tests."""

    def __init__(self):
        self._counts: Dict[str, Tuple[int, float]] = {}
        self._lock = threading.Lock()

    def incr(self, key: str, expires_at: int) -> int:
        now = time.time()
        with self._lock:
            if len(self._counts) > 10_000:
                self._counts = {k: v for k, v in self._counts.items() if v[1] > now}
            count = self._counts.get(key, (0, 0))[0] + 1
            self._counts[key] = (count, expires_at)
            return count


class DynamoCounter:
    def __init__(self, table: str, client=None):
        self.table = table
        if client is None:
            import boto3
            client = boto3.client("dynamodb")
        self.client = client

    def incr(self, key: str, expires_at: int) -> int:
        resp = self.client.update_item(
            TableName=self.table,
            Key={"pk": {"S": key}},
            UpdateExpression="ADD #c :one SET #t = if_not_exists(#t, :ttl)",
            ExpressionAttributeNames={"#c": "count", "#t": "ttl"},
            ExpressionAttributeValues={":one": {"N": "1"}, ":ttl": {"N": str(expires_at)}},
            ReturnValues="UPDATED_NEW",
        )
        return int(resp["Attributes"]["count"]["N"])


_counter = None


def counter():
    global _counter
    if _counter is None:
        _counter = DynamoCounter(settings.RATELIMIT_TABLE) if settings.RATELIMIT_TABLE else MemoryCounter()
    return _counter


def set_counter(c) -> None:
    """Test hook."""
    global _counter
    _counter = c


def limit(scope: str, setting_name: str):
    """FastAPI dependency: `dependencies=[limit("ask", "RATE_LIMIT_ASK")]`."""

    async def check(request: Request):
        per_minute = getattr(settings, setting_name)
        now = time.time()
        window = int(now // WINDOW_SECONDS)
        key = f"{scope}#{_hash_ip(client_ip(request))}#{window}"
        try:
            count = await run_in_threadpool(counter().incr, key,
                                            (window + 2) * WINDOW_SECONDS + 60)
        except Exception as e:
            logger.error("Rate limiter unavailable, allowing request: %s", e)
            return
        if count > per_minute:
            retry = WINDOW_SECONDS - int(now % WINDOW_SECONDS)
            raise HTTPException(status_code=429,
                                detail=f"Rate limit exceeded: {per_minute} per 1 minute",
                                headers={"Retry-After": str(retry)})

    return Depends(check)
