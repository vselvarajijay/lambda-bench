"""Independent, repeatable random streams derived from one run seed."""

from __future__ import annotations

import hashlib
import random


def stream_seed(seed: int, name: str) -> int:
    """Return a stable integer seed for a named stream."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def random_stream(seed: int, name: str) -> random.Random:
    """Create a named random stream that other streams cannot disturb."""
    return random.Random(stream_seed(seed, name))
