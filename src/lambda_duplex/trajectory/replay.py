"""Verify stored runs by rescoring and replaying them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from lambda_duplex.llm import GatewayMode, LLMGateway
from lambda_duplex.suites.base import Suite
from lambda_duplex.suites.loader import load_suite
from lambda_duplex.suites.runtime import run_scenario, score_trajectory
from lambda_duplex.trajectory.store import StoredRun, TrajectoryError, load_run, result_to_dict


class RecordedModel:
    """Replay the model output stored in a run."""

    capabilities = Capabilities(clock_modes=frozenset({"lockstep"}), deterministic_seed=True)

    def __init__(self, stored: StoredRun) -> None:
        events = stored.trajectory.events
        self._text = {
            event.tick: event.data["text"]
            for event in events
            if event.source == "model" and event.type == "spoke"
        }
        started = [e for e in events if e.source == "model" and e.type == "started"]
        self._session_id = started[0].data["session_id"] if started else "recorded"
        self._identity = stored.manifest["model"]
        self._steps = 0

    def describe(self) -> ModelIdentity:
        return ModelIdentity(self._identity["name"], self._identity["version"])

    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo:
        return SessionInfo(self._session_id)

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        self._steps += 1
        return StepResult(MediaFrame(tick=tick, text=self._text.get(tick)))

    async def stop(self) -> SessionSummary:
        return SessionSummary(self._steps)


@dataclass(frozen=True)
class VerifyReport:
    """Outcome of R1 and R2 checks for one run."""

    path: Path
    rescore_ok: bool
    replay_ok: bool
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.rescore_ok and self.replay_ok


def _first_difference(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> str:
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return f"event {index} differs: stored {left} replayed {right}"
    return f"event count differs: stored {len(expected)} replayed {len(actual)}"


async def verify_run(path: Path, suite: Suite | None = None) -> VerifyReport:
    """Check that a stored run rescores and replays identically."""
    stored = load_run(path)
    if suite is None:
        suite = load_suite(stored.manifest["suite"]["reference"]).suite
    if suite.manifest.version != stored.manifest["suite"]["version"]:
        raise TrajectoryError("suite version does not match the stored run")
    problems: list[str] = []
    expected = result_to_dict(stored.result, stored.decision, stored.trajectory.ticks)
    result, decision = score_trajectory(suite, stored.trajectory)
    rescore_ok = result_to_dict(result, decision, stored.trajectory.ticks) == expected
    if not rescore_ok:
        problems.append("rescored result differs from result.json")
    settings = stored.manifest["run"]
    mode: GatewayMode = "replay" if stored.llm_entries else "off"
    replayed = await run_scenario(
        suite,
        stored.scenario,
        RecordedModel(stored),
        seed=settings["seed"],
        max_ticks=settings["max_ticks"],
        condition=settings["condition"],
        llm=LLMGateway(mode, entries=stored.llm_entries),
    )
    stored_events = [event.to_dict() for event in stored.trajectory.events]
    replay_events = [event.to_dict() for event in replayed.trajectory.events]
    replay_ok = stored_events == replay_events
    if not replay_ok:
        problems.append(_first_difference(stored_events, replay_events))
    return VerifyReport(path, rescore_ok, replay_ok, tuple(problems))
