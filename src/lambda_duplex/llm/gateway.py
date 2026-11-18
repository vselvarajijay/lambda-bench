"""Single route for model calls, with recording and exact replay."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

GatewayMode = Literal["off", "record", "replay", "cache"]


class LLMDisabledError(RuntimeError):
    """A component requested a model call while calls are disabled."""


class ReplayMissError(RuntimeError):
    """Replay requested a call that was not recorded."""


@dataclass(frozen=True)
class LLMRequest:
    """A complete model request."""

    model: str
    messages: tuple[Mapping[str, str], ...]
    purpose: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True)
class LLMResponse:
    """A complete model response."""

    text: str
    model_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Backend that performs a model call."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class FakeProvider:
    """Deterministic provider for tests."""

    def __init__(self, respond: Callable[[LLMRequest, int], str]) -> None:
        self._respond = respond
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        text = self._respond(request, self.calls)
        self.calls += 1
        return LLMResponse(text, "fake-1")


class LLMGateway:
    """Record calls, replay calls, serve them from a cache, or reject them."""

    def __init__(
        self,
        mode: GatewayMode = "off",
        provider: LLMProvider | None = None,
        entries: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        if mode in ("record", "cache") and provider is None:
            raise ValueError(f"{mode} mode requires a provider")
        self.mode = mode
        self._provider = provider
        self._entries = [dict(entry) for entry in entries]
        self._lookup = {(entry["key"], entry["occurrence"]): entry for entry in self._entries}
        self._next: Counter[str] = Counter()
        self._free: defaultdict[str, list[int]] = defaultdict(list)

    @property
    def entries(self) -> Sequence[Mapping[str, Any]]:
        return tuple(self._entries)

    def _reserve(self, key: str) -> int:
        """Claim an occurrence before awaiting, so concurrent calls stay distinct."""
        if self._free[key]:
            return heapq.heappop(self._free[key])
        occurrence = self._next[key]
        self._next[key] += 1
        return occurrence

    async def _resolve(self, request: LLMRequest, key: str, occurrence: int) -> LLMResponse:
        if self.mode in ("replay", "cache"):
            cached = self._lookup.get((key, occurrence))
            if cached is not None:
                return LLMResponse(**cached["response"])
            if self.mode == "replay":
                raise ReplayMissError(f"no recorded response for {request.purpose}")
        assert self._provider is not None
        response = await self._provider.complete(request)
        entry = {
            "key": key,
            "occurrence": occurrence,
            "request": asdict(request),
            "response": asdict(response),
        }
        stored = json.loads(json.dumps(entry, allow_nan=False))
        self._entries.append(stored)
        self._lookup[(key, occurrence)] = stored
        return response

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self.mode == "off":
            raise LLMDisabledError(f"LLM calls are disabled ({request.purpose})")
        key = request.key()
        occurrence = self._reserve(key)
        try:
            return await self._resolve(request, key, occurrence)
        except BaseException:
            heapq.heappush(self._free[key], occurrence)
            raise
