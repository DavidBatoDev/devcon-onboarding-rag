# backend/app/core/limiter.py
"""Shared SlowAPI rate limiter.

Kept in its own module so both ``main.py`` (for registration) and the API
routes (for the ``@limiter.limit`` decorator) can import it without a circular
dependency.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
)
