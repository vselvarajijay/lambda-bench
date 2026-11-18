from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import ClassVar

import pytest

from lambda_duplex import cli
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
from lambda_duplex.adapters.conformance import (
    ConformanceCheck,
    ConformanceSettings,
    check_adapter,
)
from lambda_duplex.adapters.mock import MockAdapter
from lambda_duplex.audio.pcm import silence
from lambda_duplex.audio.synthetic import SyntheticSpeech

TICK_SAMPLES = 3840
SETTINGS = ConformanceSettings(step_timeout_s=0.02)


def good() -> MockAdapter:
    return MockAdapter({2: "hello from the mock model"}, False, SyntheticSpeech(7))


def run(factory: Callable[[], object]) -> dict[str, ConformanceCheck]:
    results = asyncio.run(check_adapter(factory, SETTINGS))  # type: ignore[arg-type]
    return {result.name: result for result in results}


class BaseFake:
    capabilities: ClassVar[Capabilities] = Capabilities(clock_modes=frozenset({"lockstep"}))

    def __init__(self) -> None:
        self.steps = 0

    def describe(self) -> ModelIdentity:
        return ModelIdentity("fake", "1")

    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo:
        self.steps = 0
        return SessionInfo("session")

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        self.steps += 1
        return StepResult(MediaFrame(tick, silence(TICK_SAMPLES), 24_000))

    async def stop(self) -> SessionSummary:
        return SessionSummary(self.steps)


class ShortFrame(BaseFake):
    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        return StepResult(MediaFrame(tick, b"\0\0", 24_000))


class WrongTick(BaseFake):
    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        return StepResult(MediaFrame(tick + 1, silence(TICK_SAMPLES), 24_000))


class WrongRate(BaseFake):
    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        return StepResult(MediaFrame(tick, silence(TICK_SAMPLES), 16_000))


class Slow(BaseFake):
    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        await asyncio.sleep(1)
        return await super().step(frame, tick)


class WrongSteps(BaseFake):
    async def stop(self) -> SessionSummary:
        return SessionSummary(99)


class EmptySession(BaseFake):
    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo:
        return SessionInfo("")


class FailsOnSilence(BaseFake):
    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        if not any(frame.pcm):
            raise RuntimeError("silence is unsupported")
        return await super().step(frame, tick)


class CannotRestart(BaseFake):
    started = False

    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo:
        if self.started:
            raise RuntimeError("already used")
        self.started = True
        return await super().start(context, config)


class NegativeTruncation(BaseFake):
    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        result = await super().step(frame, tick)
        return StepResult(result.output, truncated_ms=-1)


class UndeclaredText(BaseFake):
    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        return StepResult(MediaFrame(tick, silence(TICK_SAMPLES), 24_000, "hello"))


class MissingText(BaseFake):
    capabilities = Capabilities(clock_modes=frozenset({"lockstep"}), exposes_text_stream=True)

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        return StepResult(MediaFrame(tick, b"\x01\x00" * TICK_SAMPLES, 24_000))


class FalseDeterminism(BaseFake):
    capabilities = Capabilities(clock_modes=frozenset({"lockstep"}), deterministic_seed=True)
    counter = 0

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        FalseDeterminism.counter += 1
        pcm = FalseDeterminism.counter.to_bytes(2, "little") * TICK_SAMPLES
        return StepResult(MediaFrame(tick, pcm, 24_000))


class FalseInjection(BaseFake):
    capabilities = Capabilities(clock_modes=frozenset({"lockstep"}), context_injection=True)


class BadNativeFrame(BaseFake):
    capabilities = Capabilities(clock_modes=frozenset({"lockstep"}), native_frame_ms=100)


class NoMode(BaseFake):
    capabilities = Capabilities(clock_modes=frozenset())


def test_mock_adapter_passes_every_check() -> None:
    results = run(good)

    assert set(results) == {
        "lifecycle",
        "frames",
        "silence",
        "interruption",
        "restart",
        "capabilities",
    }
    assert all(result.passed for result in results.values())


@pytest.mark.parametrize(
    ("factory", "check", "message"),
    [
        (ShortFrame, "frames", "frame size"),
        (WrongTick, "frames", "returned tick"),
        (WrongRate, "frames", "wrong rate"),
        (Slow, "frames", "timed out"),
        (WrongSteps, "lifecycle", "reported 99"),
        (EmptySession, "lifecycle", "session id"),
        (FailsOnSilence, "silence", "silence is unsupported"),
        (CannotRestart, "restart", "already used"),
        (NegativeTruncation, "interruption", "negative truncation"),
        (UndeclaredText, "capabilities", "without declaring"),
        (MissingText, "capabilities", "returned no text"),
        (FalseDeterminism, "capabilities", "same seed"),
        (FalseInjection, "capabilities", "inject_context"),
        (BadNativeFrame, "capabilities", "native frame"),
        (NoMode, "lifecycle", "no clock mode"),
    ],
)
def test_broken_adapters_fail_specific_checks(
    factory: Callable[[], object], check: str, message: str
) -> None:
    result = run(factory)[check]

    assert result.passed is False
    assert message in result.detail


def test_cli_reports_mock_success(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["adapter-check", "mock"]) == 0
    assert "ok   capabilities" in capsys.readouterr().out

    assert cli.main(["adapter-check", "mock", "--json"]) == 0
    assert all(item["passed"] for item in json.loads(capsys.readouterr().out))


def test_cli_rejects_unknown_adapter(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["adapter-check", "missing"]) == 2
    assert "unknown adapter" in capsys.readouterr().err
