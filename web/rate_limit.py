"""Minimal in-memory per-IP rate limiter -- no external dependency.

Deliberately simple: a single-process/single-container deployment doesn't need
a shared store like Redis. This is meant to blunt casual/accidental abuse of a
public CPU-inference endpoint, not to be a robust production rate limiter.
"""

import os
import time
from collections import defaultdict, deque

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 10

# Unset/unrecognized APP_ENV defaults to rate-limiting (the safe, prod-like
# behavior) -- only an explicit "development" skips it.
RATE_LIMIT_ENABLED = os.environ.get("APP_ENV", "production").strip().lower() != "development"

_requests: dict[str, deque] = defaultdict(deque)


def is_rate_limited(client_ip: str) -> bool:
    if not RATE_LIMIT_ENABLED:
        return False

    now = time.monotonic()
    bucket = _requests[client_ip]
    while bucket and now - bucket[0] > WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= MAX_REQUESTS_PER_WINDOW:
        return True
    bucket.append(now)
    return False
