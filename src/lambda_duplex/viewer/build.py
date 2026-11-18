"""Build a static replay page for one stored run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from lambda_duplex.trajectory.store import StoredRun, TrajectoryError, load_run

TRACK_LABELS = {
    "mix": "Conversation mix",
    "model": "Model",
    "counterpart_degraded": "Counterpart as heard by model",
    "counterpart_clean": "Counterpart clean",
}
EVENT_LABELS = {
    "turn_action": "Turn action",
    "counterpart_event": "Counterpart sound",
    "truncated": "Model audio truncated",
    "model_transcript": "Model transcript",
    "model_transcript_partial": "Partial transcript",
    "failed": "Run failed",
}


class ViewerError(RuntimeError):
    """A replay page could not be built."""


def viewer_source() -> Path:
    """Locate viewer assets in an installed package or source checkout."""
    package_root = Path(__file__).resolve().parents[1]
    for candidate in (package_root / "_viewer", package_root.parents[1] / "viewer"):
        if (candidate / "index.html").is_file():
            return candidate
    raise ViewerError("viewer assets are not available")


def _event_label(source: str, event_type: str, data: dict[str, Any]) -> str:
    if source == "suite":
        return event_type.replace("_", " ")
    label = EVENT_LABELS.get(event_type, event_type.replace("_", " "))
    if event_type == "turn_action":
        return f"{label}: {data['action']}"
    if event_type == "counterpart_event":
        return f"{label}: {data['kind']}"
    if event_type == "failed":
        return f"{label}: {data['status']}"
    return label


def _events(run: StoredRun) -> list[dict[str, Any]]:
    visible = set(EVENT_LABELS)
    return [
        {
            "time_ms": event.time_ms,
            "label": _event_label(event.source, event.type, event.data),
        }
        for event in run.trajectory.events
        if event.source == "suite" or event.type in visible
    ]


def _transcript(
    run: StoredRun, path: Path, model_segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tick_ms = run.trajectory.tick_ms
    transcript_path = path / "transcript.jsonl"
    stored_model = False
    if transcript_path.is_file():
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("text"):
                stored_model = True
                time_ms = row.get("start_ms", row.get("time_ms", 0))
                rows.append({"time_ms": time_ms, "speaker": "Model", "text": row["text"]})
    for event in run.trajectory.events:
        if event.type != "spoke" or not event.data.get("text"):
            if event.type == "model_transcript" and event.data.get("text") and not stored_model:
                rows.append(
                    {
                        "time_ms": event.data["start_tick"] * tick_ms,
                        "speaker": "Model",
                        "text": event.data["text"],
                    }
                )
            continue
        if event.source == "counterpart":
            rows.append(
                {"time_ms": event.time_ms, "speaker": "Counterpart", "text": event.data["text"]}
            )
        elif event.source == "model" and not stored_model:
            starts = [
                item["start_ms"] for item in model_segments if item["start_ms"] <= event.time_ms
            ]
            time_ms = starts[-1] if starts else event.time_ms
            rows.append({"time_ms": time_ms, "speaker": "Model", "text": event.data["text"]})
    return sorted(rows, key=lambda item: (item["time_ms"], item["speaker"]))


def build_viewer(run_path: Path, output: Path) -> Path:
    """Create a self-contained replay folder for an audio run."""
    run = load_run(run_path)
    if run.manifest.get("audio") is None:
        raise TrajectoryError(f"{run_path} has no audio")
    if output.exists() and any(output.iterdir()):
        raise ViewerError(f"{output} is not empty")
    timeline_path = run_path / "timeline.json"
    if timeline_path.is_file():
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    else:
        from lambda_duplex.audio.timeline import build_timeline

        timeline = build_timeline(run_path)
    output.mkdir(parents=True, exist_ok=True)
    source = viewer_source()
    for name in ("index.html", "app.js", "style.css"):
        shutil.copy2(source / name, output / name)
    audio_output = output / "audio"
    audio_output.mkdir()
    tracks = []
    for name, label in TRACK_LABELS.items():
        audio_source = run_path / "audio" / f"{name}.flac"
        if audio_source.is_file() and audio_source.stat().st_size:
            shutil.copy2(audio_source, audio_output / audio_source.name)
            tracks.append({"file": f"audio/{audio_source.name}", "label": label})
    if not tracks:
        raise ViewerError(f"{run_path} has no playable audio")
    data = {
        "scenario_id": run.scenario.id,
        "suite": run.manifest["suite"]["name"],
        "model": run.manifest["model"]["reference"],
        "condition": run.manifest["run"]["condition"],
        "status": run.result.status,
        "duration_ms": timeline["duration_ms"],
        "tracks": tracks,
        "counterpart": timeline["counterpart"]["segments"],
        "model_segments": timeline["model"]["segments"],
        "events": _events(run),
        "transcript": _transcript(run, run_path, timeline["model"]["segments"]),
    }
    data["model_name"] = data.pop("model")
    data["model"] = data.pop("model_segments")
    (output / "run-data.js").write_text(
        "window.RUN_DATA = " + json.dumps(data, sort_keys=True, allow_nan=False) + ";\n",
        encoding="utf-8",
    )
    return output / "index.html"
