"""Atomic storage and loading of conversation runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import jsonschema

from lambda_duplex import __version__
from lambda_duplex.adapters.base import ModelIdentity
from lambda_duplex.orchestrator.engine import Event
from lambda_duplex.suites.base import (
    CheckResult,
    PassDecision,
    RunResult,
    Scenario,
    Suite,
    Trajectory,
)

SCHEMA_VERSION = 1
MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "suite",
        "scenario",
        "model",
        "run",
        "framework_version",
        "created_at",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "suite": {"type": "object", "required": ["name", "version", "reference"]},
        "scenario": {"type": "object", "required": ["id", "version", "sha256"]},
        "model": {"type": "object", "required": ["name", "version", "settings", "reference"]},
        "run": {
            "type": "object",
            "required": ["mode", "seed", "max_ticks", "tick_ms", "condition", "clock_mode"],
        },
    },
}
RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["scenario_id", "status", "ended_by", "ticks", "checks", "decision"],
    "properties": {
        "checks": {"type": "array"},
        "decision": {"type": "object", "required": ["passed", "reasons"]},
    },
}


class AudioTracks(Protocol):
    """Audio writer supplied by audio runs."""

    def write(self, folder: Path) -> dict[str, Any]: ...


class TrajectoryError(RuntimeError):
    """A stored run is missing, invalid, or cannot be written."""


@dataclass(frozen=True)
class RunSpec:
    """Settings needed to describe and replay a run."""

    suite_reference: str
    model_reference: str
    mode: str
    seed: int
    max_ticks: int
    tick_ms: int
    condition: str
    clock_mode: str = "lockstep"
    audio_backend: str | None = None


@dataclass(frozen=True)
class StoredRun:
    """A run loaded from disk."""

    path: Path
    manifest: Mapping[str, Any]
    scenario: Scenario
    trajectory: Trajectory
    result: RunResult
    decision: PassDecision
    llm_entries: tuple[Mapping[str, Any], ...]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    return dict(json.loads(_canonical(asdict(scenario))))


def scenario_hash(scenario: Scenario) -> str:
    return hashlib.sha256(_canonical(scenario_to_dict(scenario)).encode()).hexdigest()


def result_to_dict(result: RunResult, decision: PassDecision, ticks: int) -> dict[str, Any]:
    return {
        "scenario_id": result.scenario_id,
        "status": result.status,
        "ended_by": result.ended_by,
        "ticks": ticks,
        "checks": [asdict(check) for check in result.checks],
        "decision": {"passed": decision.passed, "reasons": list(decision.reasons)},
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")


def write_run(
    path: Path,
    suite: Suite,
    spec: RunSpec,
    model: ModelIdentity,
    trajectory: Trajectory,
    result: RunResult,
    decision: PassDecision,
    llm_entries: Iterable[Mapping[str, Any]] = (),
    tracks: AudioTracks | None = None,
    effects: Mapping[str, Any] | None = None,
    condition: Mapping[str, Any] | None = None,
    transcript: Mapping[str, Any] | None = None,
) -> Path:
    """Write a complete run directory, or leave no destination behind."""
    if path.exists():
        raise TrajectoryError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "framework_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "suite": {
            "name": suite.manifest.name,
            "version": suite.manifest.version,
            "reference": spec.suite_reference,
        },
        "scenario": {
            "id": trajectory.scenario.id,
            "version": trajectory.scenario.version,
            "sha256": scenario_hash(trajectory.scenario),
        },
        "model": {**asdict(model), "reference": spec.model_reference},
        "run": {
            "mode": spec.mode,
            "seed": spec.seed,
            "max_ticks": spec.max_ticks,
            "tick_ms": spec.tick_ms,
            "condition": spec.condition,
            "clock_mode": spec.clock_mode,
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    if effects is not None:
        manifest["effects"] = dict(effects)
    if condition is not None:
        manifest["condition"] = dict(condition)
    if transcript is not None:
        manifest["transcript"] = dict(transcript)
    try:
        if tracks is not None:
            manifest["audio"] = {"backend": spec.audio_backend, **tracks.write(temporary / "audio")}
        _write_json(temporary / "manifest.json", manifest)
        _write_json(temporary / "scenario.json", scenario_to_dict(trajectory.scenario))
        _write_jsonl(temporary / "events.jsonl", (event.to_dict() for event in trajectory.events))
        _write_jsonl(temporary / "llm_cache.jsonl", llm_entries)
        _write_json(temporary / "result.json", result_to_dict(result, decision, trajectory.ticks))
        os.chmod(temporary, 0o755)
        os.replace(temporary, path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrajectoryError(f"cannot read {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line]
    except (OSError, json.JSONDecodeError) as error:
        raise TrajectoryError(f"cannot read {path}: {error}") from error


def _validated(value: Any, schema: Mapping[str, Any], path: Path) -> Any:
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as error:
        raise TrajectoryError(f"invalid {path}: {error.message}") from error
    return value


def load_run(path: Path) -> StoredRun:
    """Load and validate one run directory."""
    manifest = _validated(
        _read_json(path / "manifest.json"), MANIFEST_SCHEMA, path / "manifest.json"
    )
    stored = _validated(_read_json(path / "result.json"), RESULT_SCHEMA, path / "result.json")
    try:
        scenario = Scenario(**_read_json(path / "scenario.json"))
        events = tuple(Event(**row) for row in _read_jsonl(path / "events.jsonl"))
        checks = tuple(CheckResult(**check) for check in stored["checks"])
    except TypeError as error:
        raise TrajectoryError(f"invalid run data in {path}: {error}") from error
    if scenario_hash(scenario) != manifest["scenario"]["sha256"]:
        raise TrajectoryError(f"scenario hash does not match manifest in {path}")
    run = manifest["run"]
    trajectory = Trajectory(
        scenario, stored["status"], stored["ended_by"], stored["ticks"], run["tick_ms"], events
    )
    result = RunResult(stored["scenario_id"], stored["status"], stored["ended_by"], checks)
    decision = PassDecision(stored["decision"]["passed"], tuple(stored["decision"]["reasons"]))
    entries = tuple(_read_jsonl(path / "llm_cache.jsonl"))
    return StoredRun(path, manifest, scenario, trajectory, result, decision, entries)


def _holds_run(path: Path) -> bool:
    """Treat an unreadable directory as a run, so loading it reports the problem."""
    try:
        return (path / "manifest.json").is_file()
    except PermissionError:
        return True


def find_run_dirs(path: Path) -> list[Path]:
    """Return a run directory, or the complete runs directly inside a directory."""
    if (path / "manifest.json").is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        child for child in path.iterdir() if not child.name.startswith(".") and _holds_run(child)
    )
