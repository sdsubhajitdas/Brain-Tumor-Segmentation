"""Minimal in-memory per-IP rate limiter -- no external dependency.

Deliberately simple: a single-process/single-container deployment doesn't need
a shared store like Redis. This is meant to blunt casual/accidental abuse of a
public CPU-inference endpoint, not to be a robust production rate limiter.
"""

import time
from collections import defaultdict, deque

WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 10

_requests: dict[str, deque] = defaultdict(deque)


def is_rate_limited(client_ip: str) -> bool:
    now = time.monotonic()
    bucket = _requests[client_ip]
    while bucket and now - bucket[0] > WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= MAX_REQUESTS_PER_WINDOW:
        return True
    bucket.append(now)
    return False
