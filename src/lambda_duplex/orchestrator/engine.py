"""Tick-based orchestrator for full-duplex conversations."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Protocol

from lambda_duplex.adapters.base import (
    ClockMode,
    MediaFrame,
    ModelAdapter,
    ModelContext,
    SessionConfig,
)
from lambda_duplex.orchestrator.clock import DEFAULT_TICK_MS, MediaClock

RunStatus = Literal["completed", "adapter_error", "adapter_timeout", "counterpart_error"]
EndedBy = Literal["counterpart", "tick_limit", "error"]
DrainedEvent = tuple[str, str, dict[str, Any]]


@dataclass(frozen=True)
class Event:
    """One ordered event in a run."""

    seq: int
    tick: int
    time_ms: int
    source: str
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    origin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Run limits and clock settings."""

    max_ticks: int
    seed: int = 0
    tick_ms: int = DEFAULT_TICK_MS
    clock_mode: ClockMode = "lockstep"
    step_timeout_s: float = 30.0
    audio_sample_rate: int | None = None
    trust_model_text: bool = True


@dataclass(frozen=True)
class RunOutcome:
    """Final state of a conversation."""

    status: RunStatus
    ticks: int
    events: tuple[Event, ...]
    error: str | None = None
    ended_by: EndedBy = "counterpart"


class Counterpart(Protocol):
    """Simulated speaker controlled by the orchestrator."""

    async def start(self, seed: int) -> None: ...

    async def hear(self, frame: MediaFrame, tick: int) -> None: ...

    async def render(self, tick: int) -> MediaFrame: ...

    def is_finished(self) -> bool: ...

    def drain_events(self) -> list[DrainedEvent]: ...

    async def finish(self, pending: MediaFrame | None, tick: int) -> None: ...

    async def stop(self) -> None: ...


class TickRecorder(Protocol):
    """Receives the media exchanged during each tick."""

    def record(self, tick: int, incoming: MediaFrame, output: MediaFrame) -> None: ...


class _CounterpartFailure(Exception):
    pass


class Orchestrator:
    """Run one model and one counterpart through a shared media clock."""

    def __init__(
        self,
        adapter: ModelAdapter,
        counterpart: Counterpart,
        config: OrchestratorConfig,
        context: ModelContext | None = None,
        recorder: TickRecorder | None = None,
    ) -> None:
        if config.max_ticks < 0:
            raise ValueError("max_ticks must not be negative")
        if config.step_timeout_s <= 0:
            raise ValueError("step_timeout_s must be positive")
        if config.clock_mode == "realtime" and not config.trust_model_text:
            raise ValueError("realtime runs must use injected model text")
        if config.clock_mode not in adapter.capabilities.clock_modes:
            raise ValueError(f"adapter does not support {config.clock_mode} mode")
        self._adapter = adapter
        self._counterpart = counterpart
        self._config = config
        self._context = context or ModelContext()
        self._clock = MediaClock(config.tick_ms)
        self._recorder = recorder
        self._tick_bytes = (
            None
            if config.audio_sample_rate is None
            else self._clock.samples_per_tick(config.audio_sample_rate) * 2
        )
        self._events: list[Event] = []

    def _append(
        self, source: str, event_type: str, data: dict[str, Any], origin: str | None = None
    ) -> None:
        self._events.append(
            Event(
                seq=len(self._events),
                tick=self._clock.tick,
                time_ms=self._clock.now_ms,
                source=source,
                type=event_type,
                data=data,
                origin=origin,
            )
        )

    def _emit(self, source: str, event_type: str, /, **data: Any) -> None:
        self._append(source, event_type, data)

    def _check_audio(self, frame: MediaFrame, owner: str) -> None:
        if self._tick_bytes is None:
            return
        if len(frame.pcm) != self._tick_bytes:
            raise ValueError(
                f"{owner} returned {len(frame.pcm)} audio bytes, expected {self._tick_bytes}"
            )
        if frame.sample_rate != self._config.audio_sample_rate:
            raise ValueError(
                f"{owner} returned sample rate {frame.sample_rate}, "
                f"expected {self._config.audio_sample_rate}"
            )
        if frame.clean_pcm is not None and len(frame.clean_pcm) != self._tick_bytes:
            raise ValueError(f"{owner} returned clean audio with the wrong length")

    def _drain_suite_events(self) -> None:
        for event_type, origin, data in self._counterpart.drain_events():
            source = "counterpart" if origin == "core" else "suite"
            self._append(source, event_type, data, None if origin == "core" else origin)

    async def _counterpart_turn(self, pending: MediaFrame | None, tick: int) -> MediaFrame | None:
        """Deliver model output, then return counterpart media unless it has finished."""
        try:
            if pending is not None:
                if not self._config.trust_model_text:
                    pending = replace(pending, text=None)
                await self._counterpart.hear(pending, tick)
                if pending.text:
                    self._emit("counterpart", "heard", text=pending.text, from_tick=pending.tick)
            self._drain_suite_events()
            if self._counterpart.is_finished():
                self._emit("counterpart", "finished")
                return None
            incoming = await self._counterpart.render(tick)
            self._check_audio(incoming, "counterpart")
            if incoming.text:
                self._emit("counterpart", "spoke", text=incoming.text)
            self._drain_suite_events()
            return incoming
        except Exception as error:
            raise _CounterpartFailure(str(error) or type(error).__name__) from error

    async def _exchange(self) -> EndedBy:
        pending: MediaFrame | None = None
        while self._clock.tick < self._config.max_ticks:
            tick = self._clock.tick
            incoming = await self._counterpart_turn(pending, tick)
            if incoming is None:
                await self._finish_counterpart(None)
                return "counterpart"
            model_frame = incoming
            if self._tick_bytes is not None:
                model_frame = replace(incoming, text=None, clean_pcm=None)
            result = await asyncio.wait_for(
                self._adapter.step(model_frame, tick), self._config.step_timeout_s
            )
            if result.output.tick != tick:
                raise ValueError(f"adapter returned tick {result.output.tick} during tick {tick}")
            self._check_audio(result.output, "adapter")
            if self._recorder is not None:
                self._recorder.record(tick, incoming, result.output)
            if result.truncated_ms > 0:
                self._emit("model", "truncated", ms=result.truncated_ms)
            if result.output.text:
                self._emit("model", "spoke", text=result.output.text)
            pending = result.output
            self._clock.advance()
        await self._finish_counterpart(pending)
        return "tick_limit"

    async def _finish_counterpart(self, pending: MediaFrame | None) -> None:
        try:
            if pending is not None and not self._config.trust_model_text:
                pending = replace(pending, text=None)
            await self._counterpart.finish(pending, self._clock.tick)
            self._drain_suite_events()
        except Exception as error:
            raise _CounterpartFailure(str(error) or type(error).__name__) from error

    async def _stop_participants(self, model_started: bool, counterpart_started: bool) -> None:
        if model_started:
            try:
                summary = await asyncio.wait_for(self._adapter.stop(), self._config.step_timeout_s)
                self._emit("model", "stopped", steps=summary.steps)
            except Exception as error:
                self._emit("model", "stop_failed", error=str(error) or type(error).__name__)
        if counterpart_started:
            try:
                await self._counterpart.stop()
                self._emit("counterpart", "stopped")
            except Exception as error:
                self._emit("counterpart", "stop_failed", error=str(error) or type(error).__name__)

    async def run(self) -> RunOutcome:
        """Run the conversation and always attempt to stop started participants."""
        status: RunStatus = "completed"
        ended_by: EndedBy = "error"
        error: str | None = None
        model_started = False
        counterpart_started = False
        config = self._config
        session = SessionConfig(config.seed, config.tick_ms, config.clock_mode)
        try:
            try:
                await self._counterpart.start(config.seed)
                counterpart_started = True
            except Exception as failure:
                raise _CounterpartFailure(str(failure) or type(failure).__name__) from failure
            self._emit("counterpart", "started")
            info = await asyncio.wait_for(
                self._adapter.start(self._context, session), config.step_timeout_s
            )
            model_started = True
            self._emit("model", "started", session_id=info.session_id)
            ended_by = await self._exchange()
        except _CounterpartFailure as failure:
            status, error = "counterpart_error", str(failure)
        except TimeoutError:
            status, error = "adapter_timeout", f"timed out after {config.step_timeout_s:g} s"
        except Exception as failure:
            status, error = "adapter_error", str(failure) or type(failure).__name__
        if error is not None:
            self._emit("orchestrator", "failed", status=status, error=error)
        await self._stop_participants(model_started, counterpart_started)
        self._emit("orchestrator", "ended", status=status, ended_by=ended_by)
        return RunOutcome(status, self._clock.tick, tuple(self._events), error, ended_by)
