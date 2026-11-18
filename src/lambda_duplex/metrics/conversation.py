"""Conversation metrics computed from speech timelines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from lambda_duplex.metrics.reliability import quantile

MODEL_TURN_MERGE_MS = 500.0
BACKCHANNEL_MAX_MS = 1000.0
BACKCHANNEL_MAX_WORDS = 3
MIN_SUPPORT = 10


@dataclass(frozen=True)
class Interval:
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class CounterpartTurn:
    start_ms: float
    end_ms: float
    pauses: tuple[Interval, ...] = ()
    scripted_pauses: tuple[Interval, ...] = ()
    eligible: bool = True

    @property
    def speech_ms(self) -> float:
        silence = sum(item.duration_ms for item in (*self.pauses, *self.scripted_pauses))
        return self.end_ms - self.start_ms - silence


@dataclass(frozen=True)
class ModelSegment:
    start_ms: float
    end_ms: float
    words: int | None = None


@dataclass(frozen=True)
class ModelTurn:
    start_ms: float
    end_ms: float
    speech_ms: float
    words: int | None
    backchannel: bool


@dataclass(frozen=True)
class ConversationTimeline:
    duration_ms: float
    counterpart: tuple[CounterpartTurn, ...]
    model: tuple[ModelSegment, ...]
    goal_ms: float | None = None
    talk_target: tuple[float, float] | None = None


@dataclass(frozen=True)
class MetricValue:
    """A metric and the number of events supporting it."""

    value: float | bool | None
    count: int
    sufficient: bool


def _check_intervals(items: Iterable[Any], duration_ms: float, owner: str) -> None:
    previous = 0.0
    for item in items:
        if item.start_ms < previous or item.end_ms < item.start_ms or item.end_ms > duration_ms:
            raise ValueError(f"{owner} intervals must be ordered and within the conversation")
        previous = item.end_ms


def validate_timeline(timeline: ConversationTimeline) -> None:
    """Reject timelines that cannot produce meaningful metrics."""
    if timeline.duration_ms < 0:
        raise ValueError("duration_ms must not be negative")
    _check_intervals(timeline.counterpart, timeline.duration_ms, "counterpart")
    _check_intervals(timeline.model, timeline.duration_ms, "model")
    for turn in timeline.counterpart:
        pauses = sorted((*turn.pauses, *turn.scripted_pauses), key=lambda item: item.start_ms)
        previous = turn.start_ms
        for pause in pauses:
            if pause.start_ms < previous or pause.end_ms > turn.end_ms:
                raise ValueError("pauses must be ordered inside their counterpart turn")
            if pause.end_ms < pause.start_ms:
                raise ValueError("pause end must not precede its start")
            previous = pause.end_ms
    if timeline.goal_ms is not None and not 0 <= timeline.goal_ms <= timeline.duration_ms:
        raise ValueError("goal_ms must be within the conversation")
    if timeline.talk_target is not None:
        low, high = timeline.talk_target
        if not 0 <= low <= high <= 1:
            raise ValueError("talk_target must be an ordered range from 0 to 1")


def model_turns(segments: Sequence[ModelSegment]) -> list[ModelTurn]:
    """Merge model speech separated by less than 500 ms and label backchannels."""
    groups: list[list[ModelSegment]] = []
    for segment in segments:
        if groups and segment.start_ms - groups[-1][-1].end_ms < MODEL_TURN_MERGE_MS:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    turns = []
    for group in groups:
        speech_ms = sum(item.end_ms - item.start_ms for item in group)
        counts = [item.words for item in group]
        words = None if any(count is None for count in counts) else sum(c or 0 for c in counts)
        short = speech_ms < BACKCHANNEL_MAX_MS
        brief = words is None or words <= BACKCHANNEL_MAX_WORDS
        turns.append(
            ModelTurn(group[0].start_ms, group[-1].end_ms, speech_ms, words, short and brief)
        )
    return turns


def _metric(value: float | bool | None, count: int) -> MetricValue:
    return MetricValue(value, count, count >= MIN_SUPPORT)


def _ratio(numerator: int, denominator: int) -> MetricValue:
    return _metric(numerator / denominator if denominator else None, denominator)


def _union_ms(intervals: Iterable[Interval]) -> float:
    total = 0.0
    current: Interval | None = None
    for item in sorted(intervals, key=lambda value: value.start_ms):
        if current is None or item.start_ms > current.end_ms:
            total += 0.0 if current is None else current.duration_ms
            current = item
        elif item.end_ms > current.end_ms:
            current = Interval(current.start_ms, item.end_ms)
    return total + (0.0 if current is None else current.duration_ms)


def _counterpart_speech(turn: CounterpartTurn) -> list[Interval]:
    parts: list[Interval] = []
    start = turn.start_ms
    pauses = sorted((*turn.pauses, *turn.scripted_pauses), key=lambda item: item.start_ms)
    for pause in pauses:
        parts.append(Interval(start, pause.start_ms))
        start = pause.end_ms
    parts.append(Interval(start, turn.end_ms))
    return [part for part in parts if part.duration_ms > 0]


def timing_metrics(timeline: ConversationTimeline) -> dict[str, MetricValue]:
    """Compute response, balance, dead-air, and efficiency metrics."""
    validate_timeline(timeline)
    turns = model_turns(timeline.model)
    substantive = [turn for turn in turns if not turn.backchannel]
    latencies: list[float] = []
    answered = early = eligible = 0
    for index, counterpart in enumerate(timeline.counterpart):
        if not counterpart.eligible:
            continue
        eligible += 1
        following = timeline.counterpart[index + 1 :]
        limit = following[0].start_ms if following else timeline.duration_ms
        onset = next(
            (
                turn.start_ms
                for turn in substantive
                if counterpart.start_ms <= turn.start_ms
                and (turn.start_ms < limit or (not following and turn.start_ms <= limit))
            ),
            None,
        )
        if onset is None:
            continue
        answered += 1
        if onset < counterpart.end_ms:
            early += 1
        else:
            latencies.append(onset - counterpart.end_ms)
    ordered = sorted(latencies)
    count = len(ordered)
    metrics = {
        "response_latency_mean_ms": _metric(sum(ordered) / count if count else None, count),
        "response_latency_p50_ms": _metric(quantile(ordered, 0.5) if count else None, count),
        "response_latency_p90_ms": _metric(quantile(ordered, 0.9) if count else None, count),
        "response_rate": _ratio(answered, eligible),
        "early_start_rate": _ratio(early, eligible),
    }
    model_speech = [Interval(item.start_ms, item.end_ms) for item in timeline.model]
    counterpart_speech = [
        part for turn in timeline.counterpart for part in _counterpart_speech(turn)
    ]
    model_ms = _union_ms(model_speech)
    counterpart_ms = _union_ms(counterpart_speech)
    speaking = model_ms + counterpart_ms
    ratio = model_ms / speaking if speaking else None
    metrics["talk_time_ratio"] = _metric(ratio, len(turns) + len(timeline.counterpart))
    if timeline.talk_target is not None:
        low, high = timeline.talk_target
        inside = None if ratio is None else low <= ratio <= high
        metrics["talk_time_in_target"] = _metric(inside, len(turns) + len(timeline.counterpart))
    scripted = [pause for turn in timeline.counterpart for pause in turn.scripted_pauses]
    scripted_ms = sum(item.duration_ms for item in scripted)
    active_ms = timeline.duration_ms - scripted_ms
    occupied = _union_ms([*model_speech, *counterpart_speech])
    overlap_scripted = _union_ms([*scripted, *model_speech]) - _union_ms(model_speech)
    silent_ms = timeline.duration_ms - occupied - overlap_scripted
    dead_air = silent_ms / active_ms if active_ms > 0 else None
    metrics["dead_air_ratio"] = _metric(dead_air, 1 if active_ms > 0 else 0)
    reached = timeline.goal_ms is not None
    end_ms = timeline.goal_ms if timeline.goal_ms is not None else timeline.duration_ms
    metrics["goal_reached"] = _metric(reached, 1)
    metrics["time_to_goal_ms"] = _metric(end_ms, 1)
    metrics["model_turns_to_goal"] = _metric(
        sum(turn.start_ms <= end_ms for turn in substantive), 1
    )
    return metrics


def metrics_to_dict(metrics: Mapping[str, MetricValue]) -> dict[str, dict[str, Any]]:
    return {name: asdict(value) for name, value in sorted(metrics.items())}


def _intervals(values: Iterable[Mapping[str, Any]]) -> tuple[Interval, ...]:
    return tuple(Interval(float(v["start_ms"]), float(v["end_ms"])) for v in values)


def timeline_from_dict(data: Mapping[str, Any]) -> ConversationTimeline:
    """Build a metric timeline from JSON data."""
    target = data.get("talk_target")
    goal = data.get("goal_ms")
    return ConversationTimeline(
        duration_ms=float(data["duration_ms"]),
        counterpart=tuple(
            CounterpartTurn(
                float(turn["start_ms"]),
                float(turn["end_ms"]),
                _intervals(turn.get("pauses", ())),
                _intervals(turn.get("scripted_pauses", ())),
                bool(turn.get("eligible", True)),
            )
            for turn in data["counterpart"]
        ),
        model=tuple(
            ModelSegment(float(item["start_ms"]), float(item["end_ms"]), item.get("words"))
            for item in data["model"]
        ),
        goal_ms=None if goal is None else float(goal),
        talk_target=None if target is None else (float(target[0]), float(target[1])),
    )
