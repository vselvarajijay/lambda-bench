"""Run suite scenarios through the orchestrator."""

from __future__ import annotations

import copy
import json
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from lambda_duplex.adapters.base import MediaFrame, ModelAdapter
from lambda_duplex.adapters.mock import MockAdapter
from lambda_duplex.llm import LLMGateway
from lambda_duplex.orchestrator import (
    DrainedEvent,
    MediaClock,
    Orchestrator,
    OrchestratorConfig,
    RunOutcome,
)
from lambda_duplex.suites.base import (
    CheckResult,
    ConversationView,
    CounterpartPolicy,
    DeliveredSpan,
    MockScripted,
    ModelUtterance,
    PassDecision,
    PolicyOutput,
    RunContext,
    RunResult,
    Scenario,
    SpeechSegment,
    Suite,
    SuiteEvent,
    Trajectory,
)
from lambda_duplex.suites.loader import public_scenario


def _stored_data(data: Any) -> dict[str, Any]:
    """Copy event data into the JSON form used by stored trajectories."""
    stored: dict[str, Any] = json.loads(json.dumps(dict(data), allow_nan=False))
    return stored


class PolicyCounterpart:
    """Adapt a suite policy to the orchestrator's counterpart contract."""

    def __init__(self, policy: CounterpartPolicy, tick_ms: int) -> None:
        self._policy = policy
        self._clock = MediaClock(tick_ms)
        self._queue: deque[SpeechSegment] = deque()
        self._events: list[DrainedEvent] = []
        self._utterances: list[ModelUtterance] = []
        self._delivered: list[DeliveredSpan] = []
        self._ending = False
        self._next_segment_id = 0
        self._hold_until_tick = 0

    def _view(self) -> ConversationView:
        return ConversationView(
            self._clock.tick,
            self._clock.now_ms,
            tuple(self._utterances),
            tuple(self._delivered),
        )

    def _record(self, events: Iterable[SuiteEvent], origin: str) -> None:
        self._events.extend((event.type, origin, _stored_data(event.data)) for event in events)

    def _accept(self, output: PolicyOutput | None) -> None:
        if output is None:
            return
        if output.preempt:
            self._queue.clear()
            self._hold_until_tick = self._clock.tick
        self._queue.extend(output.segments)
        self._record(output.events, "policy")
        self._ending = self._ending or output.end_conversation

    async def start(self, seed: int) -> None:
        return None

    async def hear(self, frame: MediaFrame, tick: int) -> None:
        self._clock.tick = tick
        if not frame.text:
            return
        utterance = ModelUtterance(frame.text, frame.tick)
        self._utterances.append(utterance)
        if not self._ending:
            self._accept(await self._policy.on_model_utterance(utterance, self._view()))

    async def render(self, tick: int) -> MediaFrame:
        self._clock.tick = tick
        if not self._ending:
            self._accept(self._policy.on_tick(self._view()))
        if tick < self._hold_until_tick or not self._queue:
            return MediaFrame(tick=tick)
        segment = self._queue.popleft()
        span = DeliveredSpan(self._next_segment_id, segment.text, segment.tags, tick, tick)
        self._next_segment_id += 1
        self._delivered.append(span)
        self._hold_until_tick = tick + 1 + math.ceil(segment.pause_after_ms / self._clock.tick_ms)
        self._record(self._policy.on_delivered(span), "delivery")
        return MediaFrame(tick=tick, text=segment.text)

    def is_finished(self) -> bool:
        if self._queue or self._clock.tick < self._hold_until_tick:
            return False
        return self._ending or self._policy.is_finished(self._view())

    def drain_events(self) -> list[DrainedEvent]:
        events, self._events = self._events, []
        return events

    async def stop(self) -> None:
        return None


@dataclass(frozen=True)
class ScenarioRun:
    """Conversation outcome and suite scoring."""

    outcome: RunOutcome
    trajectory: Trajectory
    result: RunResult
    decision: PassDecision


def decide_pass(suite: Suite, result: RunResult) -> PassDecision:
    """Apply core validity rules, then the suite pass definition."""
    if result.status != "completed":
        return PassDecision(False, (f"run {result.status}",))
    if result.ended_by != "counterpart":
        return PassDecision(False, (f"run ended by {result.ended_by}",))
    return suite.is_pass(result)


def score_trajectory(suite: Suite, trajectory: Trajectory) -> tuple[RunResult, PassDecision]:
    """Compute suite checks and the pass decision from stored evidence."""
    clean = trajectory.status == "completed" and trajectory.ended_by == "counterpart"
    checks: list[CheckResult] = []
    for check in suite.checks():
        try:
            checks.append(check.compute(trajectory))
        except Exception as error:
            if clean:
                raise
            detail = f"not computed: {error}"
            checks.append(CheckResult(check.id, check.kind, None, None, detail))
    result = RunResult(
        trajectory.scenario.id, trajectory.status, trajectory.ended_by, tuple(checks)
    )
    return result, decide_pass(suite, result)


async def run_scenario(
    suite: Suite,
    scenario: Scenario,
    adapter: ModelAdapter,
    *,
    seed: int = 0,
    max_ticks: int = 200,
    condition: str | None = None,
    llm: LLMGateway | None = None,
) -> ScenarioRun:
    """Run and score one scenario."""
    context = RunContext(
        seed, condition or suite.manifest.default_condition, llm=llm or LLMGateway()
    )
    config = OrchestratorConfig(max_ticks=max_ticks, seed=seed)
    policy = suite.create_policy(copy.deepcopy(scenario), context)
    counterpart = PolicyCounterpart(policy, config.tick_ms)
    model_context = suite.model_context(public_scenario(scenario), adapter.capabilities)
    outcome = await Orchestrator(adapter, counterpart, config, model_context).run()
    trajectory = Trajectory(
        scenario, outcome.status, outcome.ended_by, outcome.ticks, config.tick_ms, outcome.events
    )
    result, decision = score_trajectory(suite, trajectory)
    return ScenarioRun(outcome, trajectory, result, decision)


def mock_for(suite: Suite, scenario: Scenario, behavior: str = "good") -> MockAdapter:
    """Create a scripted model for a suite that provides mock behaviors."""
    if not isinstance(suite, MockScripted):
        raise ValueError(f"suite {suite.manifest.name} does not provide mock behaviors")
    scripts = suite.mock_scripts(replace(scenario))
    if behavior not in scripts:
        available = ", ".join(sorted(scripts)) or "none"
        raise ValueError(f"unknown mock behavior {behavior!r}; available: {available}")
    return MockAdapter(scripts[behavior], echo=False)
