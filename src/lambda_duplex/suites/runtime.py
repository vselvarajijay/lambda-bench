"""Run suite scenarios through the orchestrator."""

from __future__ import annotations

import array
import copy
import json
import math
import sys
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from lambda_duplex.adapters.base import MediaFrame, ModelAdapter
from lambda_duplex.adapters.mock import MockAdapter, SpeechRenderer
from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE, PlayoutBuffer, sample_count
from lambda_duplex.counterpart.turn_taking import (
    INJECTED_TAG,
    INTERRUPT_TEXT,
    OVERLAY_TEXT,
    PROMPT_TEXT,
    LLMInterruptDecider,
    SpeechGate,
    TurnAction,
    TurnController,
    TurnPersona,
    pcm_samples,
)
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
from lambda_duplex.trajectory.seeds import stream_seed

CALIBRATION_TEXT = "The quick brown fox jumps over the lazy dog near the quiet river."


def _stored_data(data: Any) -> dict[str, Any]:
    """Copy event data into the JSON form used by stored trajectories."""
    stored: dict[str, Any] = json.loads(json.dumps(dict(data), allow_nan=False))
    return stored


class AudioEffects(Protocol):
    """Streaming degradation applied to counterpart audio."""

    def process(self, pcm: bytes) -> bytes: ...

    def metadata(self) -> dict[str, Any]: ...


class TrackStore(Protocol):
    """Recorder that can write its tracks with a run."""

    def record(self, tick: int, incoming: MediaFrame, output: MediaFrame) -> None: ...

    def tracks(self) -> dict[str, bytes]: ...

    def write(self, folder: Path) -> dict[str, Any]: ...


@dataclass
class _Overlay:
    kind: str
    start_tick: int
    audio: PlayoutBuffer
    duration_ms: float


def _mix_pcm(first: bytes, second: bytes) -> bytes:
    left = pcm_samples(first)
    right = pcm_samples(second)
    mixed = array.array(
        "h", (max(-32768, min(32767, a + b)) for a, b in zip(left, right, strict=True))
    )
    if sys.byteorder == "big":
        mixed.byteswap()
    return mixed.tobytes()


@dataclass
class _Playback:
    segment: SpeechSegment
    segment_id: int
    start_tick: int
    audio: PlayoutBuffer | None
    duration_ms: float
    last_played_tick: int | None = None


class PolicyCounterpart:
    """Adapt a suite policy to the orchestrator's counterpart contract."""

    def __init__(
        self,
        policy: CounterpartPolicy,
        tick_ms: int,
        speech: SpeechRenderer | None = None,
        effects: AudioEffects | None = None,
        turns: TurnController | None = None,
    ) -> None:
        self._policy = policy
        self._clock = MediaClock(tick_ms)
        self._speech = speech
        self._effects = effects
        self._turns = turns
        self._overlays: list[_Overlay] = []
        self._speech_gate = SpeechGate() if turns is not None else None
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
        injected = INJECTED_TAG in segment.tags
        credit_tags = tuple(tag for tag in segment.tags if tag != INJECTED_TAG)
        tags = () if interrupted else credit_tags
        span = DeliveredSpan(
            playback.segment_id, segment.text, tags, playback.start_tick, end_tick, interrupted
        )
        self._delivered.append(span)
        self._playing = None
        pause_ticks = math.ceil(segment.pause_after_ms / self._clock.tick_ms)
        self._hold_until_tick = end_tick + 1 + (0 if interrupted else pause_ticks)
        self._record(self._policy.on_delivered(span), "delivery")
        played_ms = (end_tick - playback.start_tick + 1) * self._clock.tick_ms
        details: dict[str, Any] = {
            "segment_id": span.segment_id,
            "start_tick": span.start_tick,
            "end_tick": span.end_tick,
            "duration_ms": played_ms if interrupted else playback.duration_ms,
            "interrupted": interrupted,
        }
        if injected:
            details["injected"] = True
        self._events.append(("segment_delivered", "core", details))
        if self._turns is not None:
            self._turns.speech_delivered()

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
        duration_ms = float(self._clock.tick_ms)
        if self._speech is not None:
            pcm = self._speech.render(segment.text)
            audio = PlayoutBuffer(self._samples_per_tick)
            audio.push(pcm)
            duration_ms = sample_count(pcm) * 1000 / INTERNAL_SAMPLE_RATE
        playback = _Playback(segment, self._next_segment_id, tick, audio, duration_ms)
        self._next_segment_id += 1
        self._playing = playback
        return playback

    def _frame(self, tick: int, pcm: bytes, text: str | None) -> MediaFrame:
        rate = None if self._speech is None else INTERNAL_SAMPLE_RATE
        clean = pcm if self._speech is not None else None
        heard = pcm if self._effects is None else self._effects.process(pcm)
        return MediaFrame(tick, heard, rate, text, clean)

    async def start(self, seed: int) -> None:
        return None

    async def hear(self, frame: MediaFrame, tick: int) -> None:
        self._clock.tick = tick
        playing = self._playing
        if playing is not None and playing.last_played_tick is not None:
            self._deliver(playing, playing.last_played_tick, interrupted=False)
        if self._turns is not None:
            assert self._speech_gate is not None
            self._turns.observe_model(tick, self._speech_gate.update(frame.pcm), frame.text)
        if not frame.text:
            return
        utterance = ModelUtterance(frame.text, frame.tick)
        self._utterances.append(utterance)
        if not self._ending:
            self._accept(await self._policy.on_model_utterance(utterance, self._view()))

    def _core_event(self, name: str, **data: Any) -> None:
        self._events.append((name, "core", _stored_data(data)))

    def _start_overlay(self, kind: str, tick: int) -> None:
        assert self._speech is not None
        pcm = self._speech.render(OVERLAY_TEXT[kind])
        audio = PlayoutBuffer(self._samples_per_tick)
        audio.push(pcm)
        duration_ms = sample_count(pcm) * 1000 / INTERNAL_SAMPLE_RATE
        self._overlays.append(_Overlay(kind, tick, audio, duration_ms))
        start_ms = tick * self._clock.tick_ms
        self._core_event(
            "counterpart_event", kind=kind, start_ms=start_ms, end_ms=start_ms + duration_ms
        )

    def _apply_turn_action(self, action: TurnAction, tick: int) -> None:
        self._core_event("turn_action", action=action.kind, reason=action.reason)
        if action.kind == "yield":
            if self._playing is not None:
                self._deliver(self._playing, tick - 1, interrupted=True)
                self._hold_until_tick = tick
        elif action.kind == "interrupt":
            if not self._queue:
                self._queue.append(SpeechSegment(INTERRUPT_TEXT, (INJECTED_TAG,)))
            self._hold_until_tick = tick
        elif action.kind == "prompt":
            self._queue.append(SpeechSegment(PROMPT_TEXT))
        else:
            self._start_overlay(action.kind, tick)

    async def _run_turn_policy(self, tick: int) -> None:
        if self._turns is None or self._ending:
            return
        actions = await self._turns.decide(tick, self._playing is not None, bool(self._queue))
        for action in actions:
            self._apply_turn_action(action, tick)

    def _next_speech(self, tick: int, silent: bytes) -> tuple[bytes, str | None]:
        text = None
        may_start = self._turns is None or self._ending or self._turns.may_start_speech()
        if self._playing is None:
            if tick < self._hold_until_tick or not self._queue or not may_start:
                return silent, None
            text = self._start_next(tick).segment.text
        playback = self._playing
        assert playback is not None
        if playback.audio is None:
            self._deliver(playback, tick, interrupted=False)
            return b"", text
        pcm = playback.audio.pop_tick()
        if playback.audio.pending_samples == 0:
            playback.last_played_tick = tick
        return pcm, text

    def _mix_overlays(self, pcm: bytes) -> bytes:
        for overlay in list(self._overlays):
            pcm = _mix_pcm(pcm, overlay.audio.pop_tick())
            if overlay.audio.pending_samples == 0:
                self._overlays.remove(overlay)
        return pcm

    async def render(self, tick: int) -> MediaFrame:
        self._clock.tick = tick
        if not self._ending:
            self._accept(self._policy.on_tick(self._view()))
        await self._run_turn_policy(tick)
        silent = bytes(self._samples_per_tick * 2) if self._speech is not None else b""
        pcm, text = self._next_speech(tick, silent)
        return self._frame(tick, self._mix_overlays(pcm), text)

    def is_finished(self) -> bool:
        busy = self._queue or self._playing is not None or self._overlays
        if busy or self._clock.tick < self._hold_until_tick:
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
    effects: dict[str, Any] | None = None
    condition: str = "clean"
    condition_details: dict[str, Any] | None = None


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
    noise_dir: Path | None = None,
) -> ScenarioRun:
    """Run and score one scenario."""
    selected = condition or ("clean" if speech is None else suite.manifest.default_condition)
    if speech is None and selected != "clean":
        raise ValueError("text mode supports only the clean condition")
    context = RunContext(seed, selected, llm=llm or LLMGateway())
    rate = None if speech is None else INTERNAL_SAMPLE_RATE
    config = OrchestratorConfig(max_ticks=max_ticks, seed=seed, audio_sample_rate=rate)
    policy = suite.create_policy(copy.deepcopy(scenario), context)
    tracks: TrackStore | None = None
    effects: AudioEffects | None = None
    condition_details: dict[str, Any] | None = None
    turns: TurnController | None = None
    if speech is not None:
        from lambda_duplex.audio.tracks import TrackRecorder
        from lambda_duplex.conditions import condition_record, get_condition
        from lambda_duplex.counterpart.effects import EffectChain, EffectConfig, active_rms
        from lambda_duplex.counterpart.effects.chain import NOMINAL_SPEECH_RMS
        from lambda_duplex.counterpart.effects.noise_assets import load_noise_bed

        samples = MediaClock(config.tick_ms).samples_per_tick(INTERNAL_SAMPLE_RATE)
        tracks = TrackRecorder(INTERNAL_SAMPLE_RATE, samples)
        audio_condition = get_condition(selected)
        condition_details = condition_record(audio_condition)
        effect_config = audio_condition.effects
        if effect_config != EffectConfig():
            effect_seed = stream_seed(seed, f"effects:{scenario.id}")
            bed, source = None, None
            if noise_dir is not None and effect_config.noise is not None:
                noise = load_noise_bed(noise_dir, effect_seed)
                bed, source = noise.samples, noise.record
            reference = NOMINAL_SPEECH_RMS
            if effect_config.noise is not None:
                reference = active_rms(speech.render(CALIBRATION_TEXT))
            effects = EffectChain(effect_config, effect_seed, samples, bed, source, reference)
        if audio_condition.turn_taking == "free":
            public_persona = scenario.data.get("persona", {})
            hidden_persona = scenario.hidden.get("persona", {})
            if not isinstance(public_persona, Mapping) or not isinstance(hidden_persona, Mapping):
                raise ValueError("persona must be a mapping")
            values = {**public_persona, **hidden_persona}
            persona = TurnPersona.from_scenario(values, audio_condition.speech_inserts)
            decider = None
            if values.get("turn_decider_model"):
                note = str(values.get("id", ""))
                decider = LLMInterruptDecider(context.llm, str(values["turn_decider_model"]), note)
            turn_seed = stream_seed(seed, f"turns:{scenario.id}")
            turns = TurnController(persona, turn_seed, config.tick_ms, decider)
    counterpart = PolicyCounterpart(policy, config.tick_ms, speech, effects, turns)
    model_context = suite.model_context(public_scenario(scenario), adapter.capabilities)
    outcome = await Orchestrator(adapter, counterpart, config, model_context, tracks).run()
    trajectory = Trajectory(
        scenario, outcome.status, outcome.ended_by, outcome.ticks, config.tick_ms, outcome.events
    )
    result, decision = score_trajectory(suite, trajectory)
    metadata = None if effects is None else effects.metadata()
    return ScenarioRun(
        outcome,
        trajectory,
        result,
        decision,
        tracks,
        metadata,
        context.condition,
        condition_details,
    )


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
