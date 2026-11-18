"""Scripted model adapter for tests and smoke runs."""

from __future__ import annotations

from collections.abc import Mapping

from lambda_duplex.adapters.base import (
    Capabilities,
    MediaFrame,
    ModelContext,
    ModelIdentity,
    SessionConfig,
    SessionInfo,
    SessionSummary,
    StepResult,
)


class MockAdapter:
    """Reply from a tick script, or echo text heard on the current tick."""

    capabilities = Capabilities(
        clock_modes=frozenset({"lockstep"}),
        exposes_text_stream=True,
        deterministic_seed=True,
    )

    def __init__(self, script: Mapping[int, str] | None = None, echo: bool = True) -> None:
        self._script = dict(script or {})
        self._echo = echo
        self._steps = 0

    def describe(self) -> ModelIdentity:
        return ModelIdentity(name="mock", version="1", settings={"echo": self._echo})

    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo:
        self._steps = 0
        return SessionInfo(session_id=f"mock-{config.seed}")

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        self._steps += 1
        text = self._script.get(tick)
        if text is None and self._echo and frame.text:
            text = f"echo: {frame.text}"
        return StepResult(output=MediaFrame(tick=tick, text=text), compute_ms=0.0)

    async def stop(self) -> SessionSummary:
        return SessionSummary(steps=self._steps)
