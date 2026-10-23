"""Order-independent pseudo-random draws for reproducible ABM experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from typing import TypeVar


T = TypeVar("T")


class RandomStreams:
    """Create deterministic draws keyed by seed, stream, time, and entity IDs.

    The class is stateless: requesting one draw never advances or changes another
    draw. This prevents policy branches from shifting unrelated random events.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)

    def uniform(self, stream: str, time: int, *keys: object) -> float:
        if not stream:
            raise ValueError("stream must not be empty")
        payload = json.dumps(
            [self.seed, stream, int(time), *keys],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        integer = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "big"
        )
        return integer / 2**64

    def ordered(
        self,
        stream: str,
        time: int,
        items: Iterable[T],
        key: Callable[[T], object],
    ) -> list[T]:
        return sorted(
            items,
            key=lambda item: (
                self.uniform(stream, time, key(item)),
                str(key(item)),
            ),
        )
