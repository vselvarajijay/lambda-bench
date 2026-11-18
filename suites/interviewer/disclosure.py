"""Deterministic disclosure engine for interviewer scenarios."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lambda_duplex.suites.base import Scenario

from .unlock import Atom, Clause, Node, atom_dependency, clauses, event_ref

EVENT_TIMING_PREDICATES = {"elapsed_since", "no_interruption_since"}


@dataclass(frozen=True)
class UtteranceSignals:
    """Classification of one model utterance."""

    asked_about: frozenset[str] = frozenset()
    followed_up_on: frozenset[str] = frozenset()
    open_ended: bool = False
    not_leading: bool = False
    acknowledged: frozenset[Node] = frozenset()

    @classmethod
    def from_mapping(cls, values: Any, scenario: Scenario) -> UtteranceSignals:
        """Validate signals against the scenario's topics, facts, and moments."""
        if not isinstance(values, Mapping):
            raise ValueError("utterance signals must be a mapping")
        allowed = {"asked_about", "followed_up_on", "open_ended", "not_leading", "acknowledged"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown utterance signals: {sorted(map(str, unknown))}")
        for name in ("open_ended", "not_leading"):
            if not isinstance(values.get(name, False), bool):
                raise ValueError(f"{name} must be true or false")
        topics = {item["id"] for item in scenario.data["brief"]["required_topics"]}
        facts = {item["id"] for item in scenario.hidden["hidden_story"]["facts"]}
        moments = {item["id"] for item in scenario.hidden["scripted_moments"]}
        asked = _identifier_list(values.get("asked_about", ()), "asked_about", topics)
        followed = _identifier_list(values.get("followed_up_on", ()), "followed_up_on", facts)
        raw_acknowledged = values.get("acknowledged", ())
        if not isinstance(raw_acknowledged, list | tuple):
            raise ValueError("acknowledged must be a list")
        acknowledged = []
        for item in raw_acknowledged:
            node = event_ref(item)
            if node is None:
                raise ValueError("acknowledged items must be event references")
            kind, target = node
            if target not in (facts if kind == "fact" else moments):
                raise ValueError(f"unknown {kind} {target}")
            acknowledged.append(node)
        return cls(
            frozenset(asked),
            frozenset(followed),
            values.get("open_ended", False),
            values.get("not_leading", False),
            frozenset(acknowledged),
        )


def _identifier_list(value: Any, name: str, known: set[str]) -> list[str]:
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of identifiers")
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown {name}: {unknown}")
    return list(value)


@dataclass(frozen=True)
class Unlock:
    fact_id: str
    time_ms: int
    source: str
    reasons: tuple[str, ...]
    utterance_id: str


@dataclass(frozen=True)
class TrailoffRequest:
    """The next reply must end with this trail-off before its timer starts."""

    request_id: int
    fact_id: str
    source_facts: tuple[str, ...]
    min_ms: int
    utterance_id: str


@dataclass(frozen=True)
class UtteranceResult:
    unlocks: tuple[Unlock, ...]
    trailoffs: tuple[TrailoffRequest, ...]


@dataclass(frozen=True)
class _Candidate:
    clause: Clause
    atom: Atom
    min_ms: int


@dataclass
class _PendingTrailoff:
    request_id: int
    fact_id: str
    candidates: tuple[_Candidate, ...]
    signals: UtteranceSignals
    utterance_id: str
    started_ms: int | None = None


def _reason(atom: Atom, negated: bool = False) -> str:
    name, encoded = atom
    prefix = "not " if negated else ""
    return f"{prefix}{name}:{encoded}"


def _silence_atoms(clause: Clause) -> list[Atom]:
    return [atom for atom in clause.positive if atom[0] == "silence_after_trailoff"]


@dataclass
class DisclosureEngine:
    """Apply unlock rules without depending on wall time or language models."""

    scenario: Scenario
    disclosed: dict[str, int] = field(default_factory=dict)
    unlocked: dict[str, Unlock] = field(default_factory=dict)
    moments: dict[str, int] = field(default_factory=dict)
    interruptions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        facts = self.scenario.hidden["hidden_story"]["facts"]
        self._order = [fact["id"] for fact in facts]
        self._clauses = {fact["id"]: clauses(fact["unlock"]) for fact in facts}
        self._moments = {item["id"]: item for item in self.scenario.hidden["scripted_moments"]}
        self._pending: dict[int, _PendingTrailoff] = {}
        self._next_request_id = 1
        self._last_ms = 0
        self._speech_start_ms: int | None = None

    def _advance(self, now_ms: int) -> None:
        if now_ms < self._last_ms:
            raise ValueError("time must not move backward")
        self._last_ms = now_ms

    def _event_time(self, node: Node | None) -> int | None:
        if node is None:
            return None
        kind, target = node
        return self.disclosed.get(target) if kind == "fact" else self.moments.get(target)

    def _atom_holds(
        self, atom: Atom, signals: UtteranceSignals, now_ms: int, silence: Atom | None
    ) -> bool:
        name, encoded = atom
        value = json.loads(encoded)
        if name == "disclosed":
            return value in self.disclosed
        if name == "followed_up_on":
            return value in self.disclosed and value in signals.followed_up_on
        if name == "asked_about":
            return value in signals.asked_about
        if name == "open_ended":
            return signals.open_ended
        if name == "not_leading":
            return signals.not_leading
        if name == "silence_after_trailoff":
            return atom == silence
        dependency = atom_dependency(atom)
        event_ms = self._event_time(dependency)
        if event_ms is None:
            return False
        if name == "acknowledged":
            return dependency in signals.acknowledged
        if name == "elapsed_since":
            return now_ms - event_ms >= round(float(value["min_s"]) * 1000)
        if name == "no_interruption_since":
            return not any(item >= event_ms for item in self.interruptions)
        raise ValueError(f"unknown predicate {name}")

    def _negative_allows(
        self, atom: Atom, signals: UtteranceSignals, now_ms: int, silence: Atom | None
    ) -> bool:
        if atom[0] in EVENT_TIMING_PREDICATES and self._event_time(atom_dependency(atom)) is None:
            return False
        return not self._atom_holds(atom, signals, now_ms, silence)

    def _clause_holds(
        self,
        clause: Clause,
        signals: UtteranceSignals,
        now_ms: int,
        silence: Atom | None = None,
    ) -> bool:
        return all(
            self._atom_holds(atom, signals, now_ms, silence) for atom in clause.positive
        ) and (
            all(self._negative_allows(atom, signals, now_ms, silence) for atom in clause.negative)
        )

    def _unlock(
        self, fact_id: str, clause: Clause, now_ms: int, source: str, utterance_id: str
    ) -> Unlock:
        reasons = (
            *sorted(_reason(atom) for atom in clause.positive),
            *sorted(_reason(atom, True) for atom in clause.negative),
        )
        unlock = Unlock(fact_id, now_ms, source, reasons, utterance_id)
        self.unlocked[fact_id] = unlock
        for request_id, pending in list(self._pending.items()):
            if pending.fact_id == fact_id:
                del self._pending[request_id]
        return unlock

    def _candidate(
        self, clause: Clause, signals: UtteranceSignals, now_ms: int
    ) -> _Candidate | None:
        silence = _silence_atoms(clause)
        if len(silence) != 1:
            return None
        atom = silence[0]
        if self._event_time(atom_dependency(atom)) is None:
            return None
        if not self._clause_holds(clause, signals, now_ms, atom):
            return None
        return _Candidate(clause, atom, round(float(json.loads(atom[1])["min_s"]) * 1000))

    def _arm_trailoff(
        self, fact_id: str, signals: UtteranceSignals, now_ms: int, utterance_id: str
    ) -> TrailoffRequest | None:
        options = [
            candidate
            for clause in self._clauses[fact_id]
            if (candidate := self._candidate(clause, signals, now_ms)) is not None
        ]
        if not options:
            return None
        ordered = tuple(
            sorted(options, key=lambda item: (item.min_ms, sorted(item.clause.positive)))
        )
        request_id = self._next_request_id
        self._next_request_id += 1
        self._pending[request_id] = _PendingTrailoff(
            request_id, fact_id, ordered, signals, utterance_id
        )
        sources = tuple(dict.fromkeys(str(json.loads(item.atom[1])["fact"]) for item in ordered))
        return TrailoffRequest(request_id, fact_id, sources, ordered[0].min_ms, utterance_id)

    def on_utterance(
        self,
        signals: UtteranceSignals,
        now_ms: int,
        utterance_id: str,
        onset_ms: int | None = None,
    ) -> UtteranceResult:
        """Evaluate one model utterance after settling silence before its onset."""
        self._advance(now_ms)
        start_ms = now_ms if onset_ms is None else onset_ms
        if start_ms > now_ms:
            raise ValueError("utterance onset cannot be after its evaluation time")
        unlocks = list(self._settle(start_ms))
        self._pending.clear()
        self._speech_start_ms = None
        trailoffs: list[TrailoffRequest] = []
        for fact_id in self._order:
            if fact_id in self.unlocked:
                continue
            matched = next(
                (
                    clause
                    for clause in self._clauses[fact_id]
                    if self._clause_holds(clause, signals, now_ms)
                ),
                None,
            )
            if matched is not None:
                unlocks.append(self._unlock(fact_id, matched, now_ms, "utterance", utterance_id))
                continue
            request = self._arm_trailoff(fact_id, signals, now_ms, utterance_id)
            if request is not None:
                trailoffs.append(request)
        return UtteranceResult(tuple(unlocks), tuple(trailoffs))

    def on_trailoff_played(self, request_id: int, now_ms: int) -> bool:
        """Start a requested silence timer. Return false for a cancelled request."""
        self._advance(now_ms)
        pending = self._pending.get(request_id)
        if pending is None or pending.started_ms is not None:
            return False
        pending.started_ms = now_ms
        return True

    def on_substantive_onset(self, now_ms: int) -> tuple[Unlock, ...]:
        """Settle timers that already matured, then cancel all remaining timers."""
        self._advance(now_ms)
        unlocks = self._settle(now_ms)
        self._pending.clear()
        self._speech_start_ms = None
        return unlocks

    def on_model_speech_start(self, start_ms: int) -> None:
        """Hold silence timers while newly detected model speech is classified."""
        if self._speech_start_ms is None:
            self._speech_start_ms = start_ms

    def on_model_speech_end(
        self, start_ms: int, end_ms: int, substantive: bool
    ) -> tuple[Unlock, ...]:
        """Resolve held timers once a model turn can be classified."""
        if end_ms < start_ms:
            raise ValueError("model speech cannot end before it starts")
        self._advance(end_ms)
        self._speech_start_ms = None
        if not substantive:
            return ()
        unlocks = self._settle(start_ms)
        self._pending.clear()
        return unlocks

    def next_deadline(self) -> int | None:
        """Return the next time a running silence timer can complete."""
        deadlines = [
            pending.started_ms + candidate.min_ms
            for pending in self._pending.values()
            if pending.started_ms is not None
            for candidate in pending.candidates
        ]
        return min(deadlines, default=None)

    def on_tick(self, now_ms: int) -> tuple[Unlock, ...]:
        """Unlock facts whose silence condition has completed."""
        self._advance(now_ms)
        limit = now_ms if self._speech_start_ms is None else min(now_ms, self._speech_start_ms)
        return self._settle(limit)

    def _settle(self, at_ms: int) -> tuple[Unlock, ...]:
        unlocks: list[Unlock] = []
        for request_id, pending in list(self._pending.items()):
            if request_id not in self._pending or pending.started_ms is None:
                continue
            elapsed = at_ms - pending.started_ms
            waiting: list[_Candidate] = []
            matched: _Candidate | None = None
            for candidate in pending.candidates:
                if elapsed < candidate.min_ms:
                    waiting.append(candidate)
                elif self._clause_holds(candidate.clause, pending.signals, at_ms, candidate.atom):
                    matched = candidate
                    break
            if matched is not None:
                unlocks.append(
                    self._unlock(
                        pending.fact_id,
                        matched.clause,
                        at_ms,
                        "silence",
                        pending.utterance_id,
                    )
                )
            elif waiting:
                pending.candidates = tuple(waiting)
            else:
                del self._pending[request_id]
        return tuple(unlocks)

    def on_delivered(self, fact_id: str, now_ms: int) -> None:
        """Record a fact only after its speech has been delivered."""
        self._advance(now_ms)
        if fact_id not in self.unlocked:
            raise ValueError(f"fact {fact_id} was not unlocked")
        self.disclosed.setdefault(fact_id, now_ms)

    def _moment_trigger_met(self, moment: Mapping[str, Any], now_ms: int) -> bool:
        trigger = moment["trigger"]
        if "on_disclosed" in trigger:
            return trigger["on_disclosed"] in self.disclosed
        if "after_moment" in trigger:
            return trigger["after_moment"] in self.moments
        event_ms = self._event_time(event_ref(trigger["after_event"]))
        if event_ms is None:
            return False
        return now_ms - event_ms >= round(float(trigger["min_elapsed_s"]) * 1000)

    def on_pause_started(self, now_ms: int, pause_ms: int) -> None:
        """Exclude a scripted pause that starts now from running silence timers."""
        self._advance(now_ms)
        if pause_ms < 0:
            raise ValueError("pause_ms must not be negative")
        for pending in self._pending.values():
            if pending.started_ms is not None:
                pending.started_ms += pause_ms

    def on_moment(self, moment_id: str, now_ms: int) -> None:
        """Record a completed scripted moment."""
        self._advance(now_ms)
        moment = self._moments.get(moment_id)
        if moment is None:
            raise ValueError(f"unknown moment {moment_id}")
        if not self._moment_trigger_met(moment, now_ms):
            raise ValueError(f"moment {moment_id} has not been triggered")
        self.moments.setdefault(moment_id, now_ms)

    def on_interruption(self, now_ms: int) -> None:
        self._advance(now_ms)
        self.interruptions.append(now_ms)
