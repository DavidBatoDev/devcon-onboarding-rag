"""Thumbs feedback storage: DynamoDB on AWS (durable, survives deploys), a JSONL
file locally. No IP address or other client identifier is stored."""

import json
import logging
import uuid
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class JsonlFeedback:
    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class DynamoFeedback:
    def __init__(self, table: str, client=None):
        self.table = table
        if client is None:
            import boto3
            client = boto3.client("dynamodb")
        self.client = client

    def save(self, record: dict) -> None:
        item = {k: {"S": str(v)} for k, v in record.items() if v not in (None, "")}
        self.client.put_item(TableName=self.table, Item=item)


_store = None


def store():
    global _store
    if _store is None:
        _store = DynamoFeedback(settings.FEEDBACK_TABLE) if settings.FEEDBACK_TABLE \
            else JsonlFeedback(settings.FEEDBACK_FILE)
    return _store


def set_store(s) -> None:
    """Test hook."""
    global _store
    _store = s


def new_record(**fields) -> dict:
    return {"id": uuid.uuid4().hex, **fields}
