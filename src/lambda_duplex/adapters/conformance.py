"""Behavior checks for model adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

from lambda_duplex.adapters.base import (
    ClockMode,
    MediaFrame,
    ModelAdapter,
    ModelContext,
    SessionConfig,
    StepResult,
)
from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE, sample_count, silence
from lambda_duplex.orchestrator.clock import DEFAULT_TICK_MS, MediaClock

AdapterFactory = Callable[[], ModelAdapter]
PROBE_CONTEXT = ModelContext("Answer the speaker briefly.", "Answer briefly.")


@dataclass(frozen=True)
class ConformanceCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ConformanceSettings:
    ticks: int = 30
    seed: int = 7
    tick_ms: int = DEFAULT_TICK_MS
    step_timeout_s: float = 5.0


class _Failure(Exception):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _Failure(message)


class _Probe:
    def __init__(self, factory: AdapterFactory, settings: ConformanceSettings) -> None:
        self.factory = factory
        self.settings = settings
        self.samples = MediaClock(settings.tick_ms).samples_per_tick(INTERNAL_SAMPLE_RATE)
        self.silence = silence(self.samples)
        self.speech = self._speech_frames()

    def _speech_frames(self) -> list[bytes]:
        from lambda_duplex.audio.synthetic import SyntheticSpeech

        pcm = SyntheticSpeech(self.settings.seed).render("please tell me about this today")
        size = self.samples * 2
        pcm += bytes(-len(pcm) % size)
        return [pcm[index : index + size] for index in range(0, len(pcm), size)]

    def config(self, adapter: ModelAdapter) -> SessionConfig:
        modes = adapter.capabilities.clock_modes
        _require(bool(modes), "adapter declares no clock mode")
        mode: ClockMode = "lockstep" if "lockstep" in modes else "realtime"
        return SessionConfig(self.settings.seed, self.settings.tick_ms, mode)

    def input_frame(self, tick: int, pcm: bytes) -> MediaFrame:
        return MediaFrame(tick, pcm, INTERNAL_SAMPLE_RATE)

    async def step(self, adapter: ModelAdapter, tick: int, pcm: bytes) -> StepResult:
        try:
            result = await asyncio.wait_for(
                adapter.step(self.input_frame(tick, pcm), tick), self.settings.step_timeout_s
            )
        except TimeoutError as error:
            raise _Failure(f"tick {tick} timed out") from error
        output = result.output
        _require(output.tick == tick, f"tick {tick} returned tick {output.tick}")
        _require(output.sample_rate == INTERNAL_SAMPLE_RATE, f"tick {tick} has the wrong rate")
        _require(
            sample_count(output.pcm) == self.samples, f"tick {tick} returned the wrong frame size"
        )
        _require(result.truncated_ms >= 0, f"tick {tick} reported negative truncation")
        return result

    async def conversation(self, adapter: ModelAdapter) -> list[StepResult]:
        await adapter.start(PROBE_CONTEXT, self.config(adapter))
        results = []
        try:
            for tick in range(self.settings.ticks):
                pcm = self.speech[tick] if tick < len(self.speech) else self.silence
                results.append(await self.step(adapter, tick, pcm))
        finally:
            await adapter.stop()
        return results


async def _lifecycle(probe: _Probe) -> None:
    adapter = probe.factory()
    identity = adapter.describe()
    _require(bool(identity.name and identity.version), "identity needs a name and version")
    info = await adapter.start(PROBE_CONTEXT, probe.config(adapter))
    _require(bool(info.session_id), "session id is empty")
    await probe.step(adapter, 0, probe.silence)
    summary = await adapter.stop()
    _require(summary.steps == 1, f"stop reported {summary.steps} steps after 1 step")


async def _frames(probe: _Probe) -> None:
    await probe.conversation(probe.factory())


async def _silence(probe: _Probe) -> None:
    adapter = probe.factory()
    await adapter.start(PROBE_CONTEXT, probe.config(adapter))
    try:
        for tick in range(probe.settings.ticks):
            await probe.step(adapter, tick, probe.silence)
    finally:
        await adapter.stop()


async def _interruption(probe: _Probe) -> None:
    adapter = probe.factory()
    await adapter.start(PROBE_CONTEXT, probe.config(adapter))
    try:
        heard = probe.speech[: probe.settings.ticks // 3]
        tick = 0
        for pcm in [*heard, *[probe.silence] * 8, *heard]:
            if tick >= probe.settings.ticks:
                break
            await probe.step(adapter, tick, pcm)
            tick += 1
    finally:
        await adapter.stop()


async def _restart(probe: _Probe) -> None:
    adapter = probe.factory()
    await probe.conversation(adapter)
    await probe.conversation(adapter)


async def _capabilities(probe: _Probe) -> None:
    adapter = probe.factory()
    capabilities = adapter.capabilities
    _require(
        capabilities.clock_modes <= {"lockstep", "realtime"}, "adapter declares an unknown mode"
    )
    if capabilities.native_frame_ms is not None:
        _require(capabilities.native_frame_ms > 0, "native frame must be positive")
        _require(
            probe.settings.tick_ms % capabilities.native_frame_ms == 0,
            "tick is not a multiple of the native frame",
        )
    if capabilities.context_injection:
        _require(callable(getattr(adapter, "inject_context", None)), "inject_context is missing")
    results = await probe.conversation(adapter)
    spoke = any(any(result.output.pcm) for result in results)
    text = any(result.output.text for result in results)
    if capabilities.exposes_text_stream and spoke:
        _require(text, "adapter declares a text stream but returned no text")
    if not capabilities.exposes_text_stream:
        _require(not text, "adapter returned text without declaring a text stream")
    if capabilities.deterministic_seed:
        repeated = await probe.conversation(probe.factory())
        _require(
            [asdict(item.output) for item in results] == [asdict(item.output) for item in repeated],
            "same seed produced different output",
        )


CHECKS: tuple[tuple[str, Callable[[_Probe], Awaitable[None]]], ...] = (
    ("lifecycle", _lifecycle),
    ("frames", _frames),
    ("silence", _silence),
    ("interruption", _interruption),
    ("restart", _restart),
    ("capabilities", _capabilities),
)


async def check_adapter(
    factory: AdapterFactory, settings: ConformanceSettings | None = None
) -> list[ConformanceCheck]:
    """Run every check with a fresh adapter."""
    probe = _Probe(factory, settings or ConformanceSettings())
    results = []
    for name, check in CHECKS:
        try:
            await check(probe)
        except _Failure as error:
            results.append(ConformanceCheck(name, False, str(error)))
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            results.append(ConformanceCheck(name, False, detail))
        else:
            results.append(ConformanceCheck(name, True))
    return results


def report_rows(results: list[ConformanceCheck]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]
