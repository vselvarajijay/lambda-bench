"""Scripted model adapter for tests and smoke runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

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
from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE, PlayoutBuffer


class SpeechRenderer(Protocol):
    """Turns text into 16-bit mono PCM."""

    def render(self, text: str) -> bytes: ...


class MockAdapter:
    """Reply from a tick script, or echo text heard on the current tick."""

    capabilities = Capabilities(
        clock_modes=frozenset({"lockstep"}),
        exposes_text_stream=True,
        deterministic_seed=True,
    )

    def __init__(
        self,
        script: Mapping[int, str] | None = None,
        echo: bool = True,
        speech: SpeechRenderer | None = None,
    ) -> None:
        self._script = dict(script or {})
        self._echo = echo
        self._speech = speech
        self._steps = 0
        self._playout: PlayoutBuffer | None = None

    def describe(self) -> ModelIdentity:
        return ModelIdentity(name="mock", version="1", settings={"echo": self._echo})

    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo:
        self._steps = 0
        self._playout = None
        if self._speech is not None:
            samples = INTERNAL_SAMPLE_RATE * config.tick_ms // 1000
            self._playout = PlayoutBuffer(samples)
        return SessionInfo(session_id=f"mock-{config.seed}")

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        self._steps += 1
        text = self._script.get(tick)
        if text is None and self._echo and frame.text:
            text = f"echo: {frame.text}"
        pcm = b""
        if self._playout is not None and self._speech is not None:
            if text:
                self._playout.push(self._speech.render(text))
            pcm = self._playout.pop_tick()
        rate = None if self._playout is None else self._playout.sample_rate
        return StepResult(MediaFrame(tick, pcm, rate, text), compute_ms=0.0)

    async def stop(self) -> SessionSummary:
        return SessionSummary(steps=self._steps)
