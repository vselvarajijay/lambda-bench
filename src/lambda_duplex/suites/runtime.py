"""Run suite scenarios through the orchestrator."""

from __future__ import annotations

import copy
import json
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from lambda_duplex.adapters.base import MediaFrame, ModelAdapter
from lambda_duplex.adapters.mock import MockAdapter, SpeechRenderer
from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE, PlayoutBuffer
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


class TrackStore(Protocol):
    """Recorder that can write its tracks with a run."""

    def record(self, tick: int, incoming: MediaFrame, output: MediaFrame) -> None: ...

    def tracks(self) -> dict[str, bytes]: ...

    def write(self, folder: Path) -> dict[str, Any]: ...


@dataclass
class _Playback:
    segment: SpeechSegment
    segment_id: int
    start_tick: int
    audio: PlayoutBuffer | None
    last_played_tick: int | None = None


class PolicyCounterpart:
    """Adapt a suite policy to the orchestrator's counterpart contract."""

    def __init__(
        self, policy: CounterpartPolicy, tick_ms: int, speech: SpeechRenderer | None = None
    ) -> None:
        self._policy = policy
        self._clock = MediaClock(tick_ms)
        self._speech = speech
        self._samples_per_tick = self._clock.samples_per_tick(INTERNAL_SAMPLE_RATE)
        self._queue: deque[SpeechSegment] = deque()
        self._events: list[DrainedEvent] = []
        self._utterances: list[ModelUtterance] = []
        self._delivered: list[DeliveredSpan] = []
        self._playing: _Playback | None = None
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

    def _deliver(self, playback: _Playback, end_tick: int, interrupted: bool) -> None:
        segment = playback.segment
        tags = () if interrupted else segment.tags
        span = DeliveredSpan(
            playback.segment_id, segment.text, tags, playback.start_tick, end_tick, interrupted
        )
        self._delivered.append(span)
        self._playing = None
        pause_ticks = math.ceil(segment.pause_after_ms / self._clock.tick_ms)
        self._hold_until_tick = end_tick + 1 + (0 if interrupted else pause_ticks)
        self._record(self._policy.on_delivered(span), "delivery")

    def _accept(self, output: PolicyOutput | None) -> None:
        if output is None:
            return
        if output.preempt:
            self._queue.clear()
            self._hold_until_tick = self._clock.tick
            if self._playing is not None:
                self._deliver(self._playing, self._clock.tick - 1, interrupted=True)
                self._hold_until_tick = self._clock.tick
        self._queue.extend(output.segments)
        self._record(output.events, "policy")
        self._ending = self._ending or output.end_conversation

    def _start_next(self, tick: int) -> _Playback:
        segment = self._queue.popleft()
        audio = None
        if self._speech is not None:
            audio = PlayoutBuffer(self._samples_per_tick)
            audio.push(self._speech.render(segment.text))
        playback = _Playback(segment, self._next_segment_id, tick, audio)
        self._next_segment_id += 1
        self._playing = playback
        return playback

    def _frame(self, tick: int, pcm: bytes, text: str | None) -> MediaFrame:
        rate = None if self._speech is None else INTERNAL_SAMPLE_RATE
        return MediaFrame(tick, pcm, rate, text, pcm if self._speech is not None else None)

    async def start(self, seed: int) -> None:
        return None

    async def hear(self, frame: MediaFrame, tick: int) -> None:
        self._clock.tick = tick
        playing = self._playing
        if playing is not None and playing.last_played_tick is not None:
            self._deliver(playing, playing.last_played_tick, interrupted=False)
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
        silent = bytes(self._samples_per_tick * 2) if self._speech is not None else b""
        text = None
        if self._playing is None:
            if tick < self._hold_until_tick or not self._queue:
                return self._frame(tick, silent, None)
            text = self._start_next(tick).segment.text
        playback = self._playing
        assert playback is not None
        if playback.audio is None:
            self._deliver(playback, tick, interrupted=False)
            return self._frame(tick, b"", text)
        pcm = playback.audio.pop_tick()
        if playback.audio.pending_samples == 0:
            playback.last_played_tick = tick
        return self._frame(tick, pcm, text)

    def is_finished(self) -> bool:
        if self._queue or self._playing is not None or self._clock.tick < self._hold_until_tick:
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
    tracks: TrackStore | None = None


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
    speech: SpeechRenderer | None = None,
) -> ScenarioRun:
    """Run and score one scenario."""
    context = RunContext(
        seed, condition or suite.manifest.default_condition, llm=llm or LLMGateway()
    )
    rate = None if speech is None else INTERNAL_SAMPLE_RATE
    config = OrchestratorConfig(max_ticks=max_ticks, seed=seed, audio_sample_rate=rate)
    policy = suite.create_policy(copy.deepcopy(scenario), context)
    counterpart = PolicyCounterpart(policy, config.tick_ms, speech)
    tracks: TrackStore | None = None
    if speech is not None:
        from lambda_duplex.audio.tracks import TrackRecorder

        samples = MediaClock(config.tick_ms).samples_per_tick(INTERNAL_SAMPLE_RATE)
        tracks = TrackRecorder(INTERNAL_SAMPLE_RATE, samples)
    model_context = suite.model_context(public_scenario(scenario), adapter.capabilities)
    outcome = await Orchestrator(adapter, counterpart, config, model_context, tracks).run()
    trajectory = Trajectory(
        scenario, outcome.status, outcome.ended_by, outcome.ticks, config.tick_ms, outcome.events
    )
    result, decision = score_trajectory(suite, trajectory)
    return ScenarioRun(outcome, trajectory, result, decision, tracks)


def mock_for(
    suite: Suite,
    scenario: Scenario,
    behavior: str = "good",
    speech: SpeechRenderer | None = None,
) -> MockAdapter:
    """Create a scripted model for a suite that provides mock behaviors."""
    if not isinstance(suite, MockScripted):
        raise ValueError(f"suite {suite.manifest.name} does not provide mock behaviors")
    scripts = suite.mock_scripts(replace(scenario))
    if behavior not in scripts:
        available = ", ".join(sorted(scripts)) or "none"
        raise ValueError(f"unknown mock behavior {behavior!r}; available: {available}")
    return MockAdapter(scripts[behavior], echo=False, speech=speech)
