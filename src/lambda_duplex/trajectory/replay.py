"""Verify stored runs by rescoring and replaying them."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
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
from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE
from lambda_duplex.llm import GatewayMode, LLMGateway
from lambda_duplex.orchestrator import MediaClock
from lambda_duplex.suites.base import Suite
from lambda_duplex.suites.loader import load_suite
from lambda_duplex.suites.runtime import run_scenario, score_trajectory
from lambda_duplex.trajectory.store import StoredRun, TrajectoryError, load_run, result_to_dict


class RecordedModel:
    """Replay the model output stored in a run."""

    capabilities = Capabilities(clock_modes=frozenset({"lockstep"}), deterministic_seed=True)

    def __init__(self, stored: StoredRun, pcm: bytes = b"", samples_per_tick: int = 0) -> None:
        self._pcm = pcm
        self._tick_bytes = samples_per_tick * 2
        events = stored.trajectory.events
        self._text = {
            event.tick: event.data["text"]
            for event in events
            if event.source == "model" and event.type == "spoke"
        }
        started = [e for e in events if e.source == "model" and e.type == "started"]
        self._truncated = {
            event.tick: event.data["ms"]
            for event in events
            if event.source == "model" and event.type == "truncated"
        }
        self._session_id = started[0].data["session_id"] if started else "recorded"
        self._identity = stored.manifest["model"]
        self._steps = 0

    def describe(self) -> ModelIdentity:
        return ModelIdentity(self._identity["name"], self._identity["version"])

    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo:
        return SessionInfo(self._session_id)

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        self._steps += 1
        start = tick * self._tick_bytes
        pcm = self._pcm[start : start + self._tick_bytes]
        rate = INTERNAL_SAMPLE_RATE if self._tick_bytes else None
        frame = MediaFrame(tick, pcm, rate, self._text.get(tick))
        return StepResult(frame, truncated_ms=self._truncated.get(tick, 0))

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


def _stored_tracks(path: Path, audio: Mapping[str, Any]) -> tuple[dict[str, bytes], list[str]]:
    """Read every stored track and compare it with the manifest."""
    from lambda_duplex.audio.pcm import sample_count
    from lambda_duplex.audio.tracks import TRACK_NAMES, read_track

    tracks: dict[str, bytes] = {}
    problems: list[str] = []
    for name in TRACK_NAMES:
        details = audio["tracks"].get(name)
        if details is None:
            raise TrajectoryError(f"manifest has no {name} track")
        file_path = path / "audio" / f"{name}.flac"
        try:
            if details["samples"] == 0:
                if file_path.stat().st_size != 0:
                    problems.append(f"{name} track should be empty")
                tracks[name] = b""
                continue
            pcm, sample_rate = read_track(file_path)
        except (OSError, RuntimeError, ValueError) as error:
            raise TrajectoryError(f"cannot read {file_path}: {error}") from error
        tracks[name] = pcm
        if sample_rate != audio["sample_rate"]:
            problems.append(f"{name} track has sample rate {sample_rate}")
        if sample_count(pcm) != details["samples"]:
            problems.append(f"{name} track has the wrong length")
        if hashlib.sha256(pcm).hexdigest() != details["sha256"]:
            problems.append(f"{name} track differs from manifest")
    return tracks, problems


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
    audio = stored.manifest.get("audio")
    model = RecordedModel(stored)
    speech = None
    if audio is not None:
        from lambda_duplex.audio.synthetic import SyntheticSpeech

        if audio["backend"] != "synthetic":
            raise TrajectoryError(f"cannot replay audio backend {audio['backend']}")
        samples = MediaClock(settings["tick_ms"]).samples_per_tick(audio["sample_rate"])
        stored_tracks, track_problems = _stored_tracks(path, audio)
        problems.extend(track_problems)
        model = RecordedModel(stored, stored_tracks["model"], samples)
        speech = SyntheticSpeech(settings["seed"])
    replayed = await run_scenario(
        suite,
        stored.scenario,
        model,
        seed=settings["seed"],
        max_ticks=settings["max_ticks"],
        condition=settings["condition"],
        llm=LLMGateway(mode, entries=stored.llm_entries),
        speech=speech,
    )
    stored_events = [event.to_dict() for event in stored.trajectory.events]
    replay_events = [event.to_dict() for event in replayed.trajectory.events]
    replay_ok = stored_events == replay_events and not problems
    if stored_events != replay_events:
        problems.append(_first_difference(stored_events, replay_events))
    if audio is not None and replayed.tracks is not None:
        for name, pcm in replayed.tracks.tracks().items():
            if pcm != stored_tracks[name]:
                replay_ok = False
                problems.append(f"{name} audio differs after replay")
    return VerifyReport(path, rescore_ok, replay_ok, tuple(problems))
