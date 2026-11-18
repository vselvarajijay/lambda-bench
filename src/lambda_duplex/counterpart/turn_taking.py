"""Rules that decide how the simulated speaker takes turns."""

from __future__ import annotations

import array
import json
import math
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal, Protocol

from lambda_duplex.llm import LLMGateway, LLMRequest
from lambda_duplex.trajectory.seeds import random_stream

ActionKind = Literal["yield", "interrupt", "prompt", "backchannel", "vocal_tic", "non_directed"]
OVERLAY_TEXT = {
    "backchannel": "mm hm",
    "vocal_tic": "um",
    "non_directed": "hold on one second",
}
INTERRUPT_TEXT = "Sorry, can I jump in?"
PROMPT_TEXT = "Are you still there?"
INJECTED_TAG = "_injected_interrupt"
MODEL_TURN_GAP_MS = 500
GATE_FRAME_SAMPLES = 480
GATE_WINDOW_FRAMES = 50
GATE_MIN_FRAMES = 24
GATE_MIN_RMS = 328.0
GATE_MIN_RANGE = 2.0


def pcm_samples(pcm: bytes) -> array.array[int]:
    """Return signed 16-bit samples from little-endian PCM."""
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


class SpeechGate:
    """Detect speech-like model audio without treating steady noise as speech."""

    def __init__(self) -> None:
        self._levels: deque[float] = deque(maxlen=GATE_WINDOW_FRAMES)

    def update(self, pcm: bytes) -> bool:
        samples = pcm_samples(pcm)
        current: list[float] = []
        for start in range(0, len(samples) - GATE_FRAME_SAMPLES + 1, GATE_FRAME_SAMPLES):
            frame = samples[start : start + GATE_FRAME_SAMPLES]
            current.append(math.sqrt(sum(value * value for value in frame) / len(frame)))
        self._levels.extend(current)
        if not current or sum(current) / len(current) < GATE_MIN_RMS:
            return False
        if len(self._levels) < GATE_MIN_FRAMES:
            return True
        ordered = sorted(self._levels)
        low = max(ordered[len(ordered) // 10], 1.0)
        high = ordered[len(ordered) * 9 // 10]
        return high / low >= GATE_MIN_RANGE


def _number(
    values: Mapping[str, Any],
    name: str,
    default: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f"{name} is out of range")
    return number


def _whole_ms(values: Mapping[str, Any], name: str, default: int) -> int:
    number = _number(values, name, default, 0)
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number of milliseconds")
    return int(number)


@dataclass(frozen=True)
class TurnPersona:
    """Speaker tendencies, using milliseconds for durations."""

    talk_limit_ms: int = 9000
    interruptiveness: float = 0.2
    silence_tolerance_ms: int = 8000
    decision_interval_ms: int = 2000
    yield_when_interrupted_ms: int = 1000
    yield_when_interrupting_ms: int = 5000
    backchannel_min_ms: int = 3000
    backchannel_max_ms: int = 12000
    backchannels_per_second: float = 0.1
    speech_inserts_per_minute: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{item.name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} is out of range")
        if self.interruptiveness > 1:
            raise ValueError("interruptiveness must be between 0 and 1")
        if self.decision_interval_ms <= 0:
            raise ValueError("decision_interval_ms must be positive")
        if self.backchannel_min_ms > self.backchannel_max_ms:
            raise ValueError("backchannel_min_ms must not exceed backchannel_max_ms")

    @classmethod
    def from_scenario(cls, values: Mapping[str, Any], inserts: bool) -> TurnPersona:
        """Build settings from scenario persona fields."""
        patience = _number(values, "patience", 0.5, 0, 1)
        return cls(
            talk_limit_ms=round(3000 + patience * 12_000),
            interruptiveness=_number(values, "interruptiveness", 0.2, 0, 1),
            silence_tolerance_ms=_whole_ms(values, "silence_tolerance_ms", 8000),
            yield_when_interrupted_ms=_whole_ms(values, "yield_when_interrupted_ms", 1000),
            yield_when_interrupting_ms=_whole_ms(values, "yield_when_interrupting_ms", 5000),
            speech_inserts_per_minute=2.0 if inserts else 0.0,
        )


@dataclass(frozen=True)
class TurnAction:
    kind: ActionKind
    reason: str


class InterruptDecider(Protocol):
    """Optional decision source for whether to interrupt the model."""

    async def should_interrupt(self, model_text: str, run_ms: int) -> bool: ...


class LLMInterruptDecider:
    """Ask a recorded LLM whether the speaker should interrupt."""

    def __init__(self, llm: LLMGateway, model: str, persona_note: str = "") -> None:
        self._llm = llm
        self._model = model
        self._persona_note = persona_note

    async def should_interrupt(self, model_text: str, run_ms: int) -> bool:
        prompt = {
            "persona": self._persona_note,
            "other_speaker_recent_text": model_text,
            "other_speaker_talking_ms": run_ms,
            "instruction": 'Return JSON with one boolean field, "interrupt".',
        }
        request = LLMRequest(
            self._model,
            ({"role": "user", "content": json.dumps(prompt, sort_keys=True)},),
            "turn_taking",
        )
        response = await self._llm.complete(request)
        decision = json.loads(response.text)
        value = decision.get("interrupt") if isinstance(decision, dict) else None
        if not isinstance(value, bool):
            raise ValueError("turn-taking response must contain boolean interrupt")
        return value


class TurnController:
    """Deterministic free-form turn-taking state for one conversation."""

    def __init__(
        self,
        persona: TurnPersona,
        seed: int,
        tick_ms: int,
        decider: InterruptDecider | None = None,
    ) -> None:
        self.persona = persona
        self.tick_ms = tick_ms
        self._decider = decider
        self._interrupt_rng = random_stream(seed, "turns:interrupt")
        self._backchannel_rng = random_stream(seed, "turns:backchannel")
        self._insert_rng = random_stream(seed, "turns:insert")
        self._gap_ticks = math.ceil(MODEL_TURN_GAP_MS / tick_ms)
        self._last_model_speech: int | None = None
        self._model_start: int | None = None
        self._model_active = False
        self._last_activity = 0
        self._overlap_start: int | None = None
        self._last_backchannel: int | None = None
        self._last_decision_slot = -1
        self._prompted_since_activity = False
        self._interrupt_blocked = False
        self._awaiting_model = False
        self.interrupting = False
        self.latest_model_text = ""

    def observe_model(self, tick: int, speaking: bool, text: str | None = None) -> None:
        """Record model audio heard on this tick, bridging normal speech pauses."""
        if text:
            self.latest_model_text = text
        if speaking:
            if not self._model_active:
                self._model_start = tick
                self._last_backchannel = None
            self._model_active = True
            self._last_model_speech = tick
            self._last_activity = tick
            self._prompted_since_activity = False
            self._awaiting_model = False
        elif self._last_model_speech is None or tick - self._last_model_speech >= self._gap_ticks:
            self._model_active = False
            self._model_start = None
            self._overlap_start = None
            self._interrupt_blocked = False

    def speech_delivered(self) -> None:
        """Record that a counterpart segment finished or was cut."""
        self.interrupting = False
        self._awaiting_model = True

    @property
    def model_speaking(self) -> bool:
        return self._model_active

    def may_start_speech(self) -> bool:
        return not self._model_active or self.interrupting

    def needs_decision_text(self, tick: int) -> bool:
        """Return whether an interrupt decision on this tick needs current model text."""
        if self._decider is None or self._model_start is None or self.interrupting:
            return False
        if self._interrupt_blocked or self._last_model_speech != tick:
            return False
        slot = self._ms(tick) // self.persona.decision_interval_ms
        run_ms = self._ms(tick - self._model_start + 1)
        return slot != self._last_decision_slot and run_ms >= self.persona.talk_limit_ms

    def _ms(self, ticks: int) -> int:
        return ticks * self.tick_ms

    def _chance(self, per_second: float) -> float:
        return min(1.0, per_second * self.tick_ms / 1000)

    def _decision_due(self, tick: int) -> bool:
        slot = self._ms(tick) // self.persona.decision_interval_ms
        if slot == self._last_decision_slot:
            return False
        self._last_decision_slot = slot
        return True

    async def _wants_interrupt(self, run_ms: int) -> bool:
        if self._decider is not None:
            return await self._decider.should_interrupt(self.latest_model_text, run_ms)
        return self._interrupt_rng.random() < self.persona.interruptiveness

    def _yield_action(self, tick: int, playing: bool) -> TurnAction | None:
        if not (playing and self._model_active):
            self._overlap_start = None
            return None
        if self._overlap_start is None:
            if self._last_model_speech != tick:
                return None
            self._overlap_start = tick
        assert self._last_model_speech is not None
        limit = self.persona.yield_when_interrupted_ms
        if self.interrupting:
            limit = self.persona.yield_when_interrupting_ms
        if self._ms(self._last_model_speech - self._overlap_start + 1) < limit:
            return None
        self._overlap_start = None
        self.interrupting = False
        self._interrupt_blocked = True
        return TurnAction("yield", "overlap_limit")

    def _backchannel_action(self, tick: int, run_ms: int) -> TurnAction | None:
        persona = self.persona
        reference = self._last_backchannel
        since = run_ms if reference is None else self._ms(tick - reference)
        if since < persona.backchannel_min_ms:
            return None
        chance = self._chance(persona.backchannels_per_second)
        if since < persona.backchannel_max_ms and self._backchannel_rng.random() >= chance:
            return None
        self._last_backchannel = tick
        return TurnAction("backchannel", "listener_feedback")

    def _insert_action(self, playing: bool) -> TurnAction | None:
        rate = self.persona.speech_inserts_per_minute / 60
        if rate <= 0 or self._insert_rng.random() >= self._chance(rate):
            return None
        if playing:
            return TurnAction("vocal_tic", "speech_insert")
        if self._model_active:
            kind: ActionKind = "vocal_tic" if self._insert_rng.random() < 0.5 else "non_directed"
            return TurnAction(kind, "speech_insert")
        if self._awaiting_model:
            return None
        return TurnAction("non_directed", "speech_insert")

    async def decide(self, tick: int, playing: bool, pending: bool) -> list[TurnAction]:
        """Return actions for this tick."""
        actions: list[TurnAction] = []
        if playing or pending:
            self._last_activity = tick
        yielded = self._yield_action(tick, playing)
        if yielded is not None:
            actions.append(yielded)
            playing = False
        due = self._decision_due(tick)
        model_audible = self._last_model_speech == tick
        if model_audible and self._model_start is not None and not playing:
            run_ms = self._ms(tick - self._model_start + 1)
            may_interrupt = due and not self.interrupting and not self._interrupt_blocked
            if may_interrupt and run_ms >= self.persona.talk_limit_ms:
                if await self._wants_interrupt(run_ms):
                    self.interrupting = True
                    actions.append(TurnAction("interrupt", "model_talk_limit"))
            backchannel = self._backchannel_action(tick, run_ms)
            if backchannel is not None:
                actions.append(backchannel)
        elif not self._model_active and not model_audible:
            silent_ms = self._ms(tick - self._last_activity)
            if (
                not playing
                and not pending
                and not self._prompted_since_activity
                and silent_ms >= self.persona.silence_tolerance_ms
            ):
                self._prompted_since_activity = True
                actions.append(TurnAction("prompt", "silence_tolerance"))
        insert = self._insert_action(playing)
        if insert is not None:
            actions.append(insert)
        return actions
