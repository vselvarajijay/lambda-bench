"""Scripted model adapter for tests and smoke runs."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping
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
from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE, PlayoutBuffer, sample_count


class SpeechRenderer(Protocol):
    """Turns text into 16-bit mono PCM."""

    def render(self, text: str) -> bytes: ...


class MockAdapter:
    """Reply from a tick script, or echo text heard on the current tick.

    In audio mode, text is released on the tick its final sample plays. Ticks listed in
    ``truncate_at`` drop unplayed audio and its unreleased text.
    """

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
        truncate_at: Collection[int] = (),
    ) -> None:
        self._script = dict(script or {})
        self._echo = echo
        self._speech = speech
        self._truncate_at = frozenset(truncate_at)
        self._steps = 0
        self._playout: PlayoutBuffer | None = None
        self._pending_text: deque[tuple[int, str]] = deque()
        self._pushed = 0
        self._played = 0

    def describe(self) -> ModelIdentity:
        return ModelIdentity(name="mock", version="1", settings={"echo": self._echo})

    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo:
        self._steps = 0
        self._playout = None
        self._pending_text.clear()
        self._pushed = 0
        self._played = 0
        if self._speech is not None:
            samples = INTERNAL_SAMPLE_RATE * config.tick_ms // 1000
            self._playout = PlayoutBuffer(samples)
        return SessionInfo(session_id=f"mock-{config.seed}")

    def _step_audio(self, playout: PlayoutBuffer, text: str | None, tick: int) -> StepResult:
        assert self._speech is not None
        truncated_ms = 0
        if tick in self._truncate_at:
            truncated_ms = playout.truncate()
            self._pending_text.clear()
            self._pushed = self._played
        if text:
            pcm = self._speech.render(text)
            playout.push(pcm)
            self._pushed += sample_count(pcm)
            self._pending_text.append((self._pushed, text))
        output = playout.pop_tick()
        self._played = min(self._played + playout.samples_per_tick, self._pushed)
        released: list[str] = []
        while self._pending_text and self._pending_text[0][0] <= self._played:
            released.append(self._pending_text.popleft()[1])
        frame = MediaFrame(tick, output, playout.sample_rate, " ".join(released) or None)
        return StepResult(frame, truncated_ms=truncated_ms, compute_ms=0.0)

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        self._steps += 1
        text = self._script.get(tick)
        if text is None and self._echo and frame.text:
            text = f"echo: {frame.text}"
        if self._playout is not None:
            return self._step_audio(self._playout, text, tick)
        return StepResult(MediaFrame(tick, text=text), compute_ms=0.0)

    async def stop(self) -> SessionSummary:
        return SessionSummary(steps=self._steps)
