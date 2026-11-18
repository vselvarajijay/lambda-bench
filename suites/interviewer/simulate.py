"""Run a scripted interview through the disclosure engine."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from lambda_duplex.suites.base import Scenario

from .disclosure import DisclosureEngine, TrailoffRequest, Unlock, UtteranceSignals
from .validation import PAUSE_PATTERN

DELIVERY_MS = 500
ACTIONS = {"utterance", "wait_s", "moment", "onset", "interruption"}
OPTIONAL: dict[str, set[str]] = {
    "utterance": set(),
    "wait_s": set(),
    "moment": set(),
    "onset": set(),
    "interruption": set(),
}


def _milliseconds(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    try:
        milliseconds = float(value) * 1000
    except OverflowError as error:
        raise ValueError(f"{name} is too large") from error
    if not math.isfinite(milliseconds) or milliseconds < 0:
        raise ValueError(f"{name} must be finite and not negative")
    return round(milliseconds)


def _action(step: Any, index: int) -> str:
    if not isinstance(step, Mapping):
        raise ValueError(f"step {index} must be a mapping")
    kinds = ACTIONS & set(step)
    if len(kinds) != 1:
        raise ValueError(f"step {index} must contain exactly one action")
    action = next(iter(kinds))
    unknown = set(step) - {action, "after_s"} - OPTIONAL[action]
    if unknown:
        raise ValueError(f"step {index} has unknown keys: {sorted(map(str, unknown))}")
    if action in {"onset", "interruption"} and step[action] is not True:
        raise ValueError(f"step {index}: {action} must be true")
    return action


class _Simulation:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.engine = DisclosureEngine(scenario)
        self.now_ms = 0
        self.rows: list[dict[str, Any]] = []

    def moment_pause_ms(self, moment_id: str) -> int:
        for moment in self.scenario.hidden["scripted_moments"]:
            script = moment.get("script")
            if moment["id"] == moment_id and isinstance(script, str):
                match = PAUSE_PATTERN.search(script)
                if match is not None:
                    return _milliseconds(float(match.group(1)), "pause")
        return 0

    def deliver(self, unlocks: Sequence[Unlock]) -> None:
        for unlock in unlocks:
            self.rows.append(
                {"event": "unlocked", **asdict(unlock), "reasons": list(unlock.reasons)}
            )
        if not unlocks:
            return
        self.now_ms += DELIVERY_MS
        for unlock in unlocks:
            self.engine.on_delivered(unlock.fact_id, self.now_ms)
            self.rows.append(
                {"event": "disclosed", "fact_id": unlock.fact_id, "time_ms": self.now_ms}
            )

    def advance(self, target_ms: int) -> None:
        while True:
            deadline = self.engine.next_deadline()
            if deadline is None or deadline > target_ms:
                break
            self.now_ms = max(self.now_ms, deadline)
            self.deliver(self.engine.on_tick(self.now_ms))
        self.now_ms = max(self.now_ms, target_ms)

    def play_trailoffs(self, requests: Sequence[TrailoffRequest]) -> None:
        if not requests:
            return
        self.now_ms += DELIVERY_MS
        for request in requests:
            self.rows.append({"event": "trailoff", **asdict(request), "time_ms": self.now_ms})
            self.engine.on_trailoff_played(request.request_id, self.now_ms)


def simulate(scenario: Scenario, steps: Any) -> list[dict[str, Any]]:
    """Return disclosures and reasons for a deterministic interview script."""
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")
    run = _Simulation(scenario)
    for index, step in enumerate(steps):
        action = _action(step, index)
        run.advance(run.now_ms + _milliseconds(step.get("after_s", 1.0), "after_s"))
        if action == "utterance":
            signals = UtteranceSignals.from_mapping(step["utterance"], scenario)
            result = run.engine.on_utterance(signals, run.now_ms, f"u{index}")
            run.deliver(result.unlocks)
            run.play_trailoffs(result.trailoffs)
        elif action == "wait_s":
            run.advance(run.now_ms + _milliseconds(step["wait_s"], "wait_s"))
        elif action == "moment":
            pause_ms = run.moment_pause_ms(str(step["moment"]))
            if pause_ms:
                run.engine.on_pause_started(run.now_ms, pause_ms)
                run.advance(run.now_ms + pause_ms)
            run.engine.on_moment(str(step["moment"]), run.now_ms)
            run.rows.append({"event": "moment", "moment_id": step["moment"], "time_ms": run.now_ms})
        elif action == "onset":
            run.deliver(run.engine.on_substantive_onset(run.now_ms))
            run.rows.append({"event": "onset", "time_ms": run.now_ms})
        else:
            run.engine.on_interruption(run.now_ms)
            run.rows.append({"event": "interruption", "time_ms": run.now_ms})
    return run.rows
