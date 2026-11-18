from __future__ import annotations

import asyncio

import pytest

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
from lambda_duplex.adapters.mock import MockAdapter
from lambda_duplex.orchestrator import MediaClock, Orchestrator, OrchestratorConfig
from lambda_duplex.orchestrator.scripted import ScriptedCounterpart


class FaultyAdapter(MockAdapter):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.stopped = False

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        if tick == 1 and self.mode == "error":
            raise RuntimeError("model crashed")
        if tick == 1 and self.mode == "hang":
            await asyncio.sleep(60)
        if tick == 1 and self.mode == "wrong_tick":
            return StepResult(MediaFrame(tick=9))
        return await super().step(frame, tick)

    async def stop(self) -> SessionSummary:
        self.stopped = True
        return await super().stop()


class FailingCounterpart(ScriptedCounterpart):
    async def render(self, tick: int) -> MediaFrame:
        raise RuntimeError("policy crashed")


class RealtimeOnlyAdapter:
    capabilities = Capabilities(clock_modes=frozenset({"realtime"}))

    def describe(self) -> ModelIdentity:
        return ModelIdentity("realtime", "1")

    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo:
        return SessionInfo("x")

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        return StepResult(MediaFrame(tick))

    async def stop(self) -> SessionSummary:
        return SessionSummary(0)


def run(adapter: MockAdapter, counterpart: ScriptedCounterpart, ticks: int = 3):  # type: ignore[no-untyped-def]
    config = OrchestratorConfig(max_ticks=ticks, step_timeout_s=0.01)
    return asyncio.run(Orchestrator(adapter, counterpart, config).run())


def test_clock_uses_integer_media_time() -> None:
    clock = MediaClock(160)
    clock.advance()
    clock.advance()

    assert clock.now_ms == 320
    assert clock.samples_per_tick(24_000) == 3840
    assert clock.samples_per_tick(16_000) == 2560
    assert clock.samples_per_tick(8_000) == 1280


def test_clock_rejects_fractional_samples() -> None:
    with pytest.raises(ValueError, match="whole number"):
        MediaClock(1).samples_per_tick(22_050 * 1)


def test_tick_order_and_event_sequence() -> None:
    outcome = run(MockAdapter(), ScriptedCounterpart({0: "hello"}), ticks=2)

    assert outcome.status == "completed"
    assert [(event.source, event.type, event.tick) for event in outcome.events] == [
        ("counterpart", "started", 0),
        ("model", "started", 0),
        ("counterpart", "spoke", 0),
        ("model", "spoke", 0),
        ("counterpart", "heard", 1),
        ("model", "stopped", 2),
        ("counterpart", "stopped", 2),
        ("orchestrator", "ended", 2),
    ]
    assert [event.seq for event in outcome.events] == list(range(8))


def test_model_output_is_audible_one_tick_later() -> None:
    counterpart = ScriptedCounterpart({})
    run(MockAdapter({0: "first", 1: "second"}), counterpart, ticks=3)

    assert [(tick, frame.tick, frame.text) for tick, frame in counterpart.heard] == [
        (1, 0, "first"),
        (2, 1, "second"),
    ]


def test_counterpart_can_finish_early() -> None:
    outcome = run(MockAdapter(), ScriptedCounterpart({}, finish_after=1), ticks=10)

    assert outcome.status == "completed"
    assert outcome.ticks == 2
    assert any(event.type == "finished" for event in outcome.events)


@pytest.mark.parametrize(
    ("mode", "status"),
    [("error", "adapter_error"), ("hang", "adapter_timeout"), ("wrong_tick", "adapter_error")],
)
def test_adapter_failures_stop_cleanly(mode: str, status: str) -> None:
    adapter = FaultyAdapter(mode)

    outcome = run(adapter, ScriptedCounterpart({}))

    assert outcome.status == status
    assert outcome.ticks == 1
    assert adapter.stopped
    assert outcome.events[-1].type == "ended"
    assert outcome.events[-1].data == {"status": status, "ended_by": "error"}
    assert outcome.ended_by == "error"


def test_counterpart_failure_is_not_blamed_on_model() -> None:
    adapter = FaultyAdapter("none")

    outcome = run(adapter, FailingCounterpart({}))

    assert outcome.status == "counterpart_error"
    assert outcome.error == "policy crashed"
    assert adapter.stopped


def test_unsupported_clock_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="lockstep"):
        Orchestrator(RealtimeOnlyAdapter(), ScriptedCounterpart({}), OrchestratorConfig(1))
