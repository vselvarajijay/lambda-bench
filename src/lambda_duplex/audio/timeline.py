"""Speech timelines derived from stored conversation runs."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from lambda_duplex.audio.tracks import read_track, resample_pcm
from lambda_duplex.audio.vad import (
    MODEL_SHA256,
    VAD_SAMPLE_RATE,
    WINDOW_MS,
    SileroVad,
    VadConfig,
    segments_from_probabilities,
)
from lambda_duplex.trajectory.store import StoredRun, TrajectoryError, load_run

TIMELINE_VERSION = 1


def _atomic_bytes(path: Path, data: bytes) -> None:
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(data)
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def counterpart_segments(run: StoredRun) -> list[dict[str, Any]]:
    """Return counterpart speech from the simulator's delivered plan."""
    tick_ms = run.trajectory.tick_ms
    return [
        {
            "segment_id": event.data["segment_id"],
            "start_ms": event.data["start_tick"] * tick_ms,
            "end_ms": event.data["start_tick"] * tick_ms + event.data["duration_ms"],
            "interrupted": event.data["interrupted"],
        }
        for event in run.trajectory.events
        if event.source == "counterpart" and event.type == "segment_delivered"
    ]


def build_timeline(
    path: Path, vad: SileroVad | None = None, config: VadConfig | None = None
) -> dict[str, Any]:
    """Create and store the timeline for one audio run."""
    run = load_run(path)
    audio = run.manifest.get("audio")
    if audio is None:
        raise TrajectoryError(f"{path} has no audio")
    settings = config or VadConfig()
    samples = audio["tracks"]["model"]["samples"]
    duration_ms = samples * 1000 / audio["sample_rate"]
    probabilities = np.zeros(0, dtype=np.float32)
    if samples:
        pcm, rate = read_track(path / "audio" / "model.flac")
        speech = resample_pcm(pcm, rate, VAD_SAMPLE_RATE)
        probabilities = (vad or SileroVad()).probabilities(speech)
    segments = segments_from_probabilities(probabilities, duration_ms, settings)
    timeline = {
        "version": TIMELINE_VERSION,
        "duration_ms": duration_ms,
        "tick_ms": run.trajectory.tick_ms,
        "vad": {
            "model": "silero_vad.onnx",
            "model_sha256": MODEL_SHA256,
            "window_ms": WINDOW_MS,
            "config": settings.to_dict(),
        },
        "model": {
            "source": "vad:model_track",
            "segments": [asdict(segment) for segment in segments],
        },
        "counterpart": {"source": "simulator_plan", "segments": counterpart_segments(run)},
    }
    text = json.dumps(timeline, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path / "timeline.json", text.encode())
    handle, name = tempfile.mkstemp(prefix=".timeline.", suffix=".npz", dir=path)
    os.close(handle)
    try:
        np.savez_compressed(name, model_speech_probability=probabilities)
        os.replace(name, path / "timeline.npz")
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise
    return timeline


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def plot_timeline(timeline: dict[str, Any], probabilities: np.ndarray, output: Path) -> None:
    """Write a dependency-free PNG with speaker bars and model speech probability."""
    width, height = 1000, 150
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    duration = max(float(timeline["duration_ms"]), 1.0)

    def x_position(time_ms: float) -> int:
        return min(width - 1, max(0, round(time_ms / duration * (width - 1))))

    rows = (("counterpart", 10, (38, 110, 180)), ("model", 45, (216, 96, 48)))
    for name, top, color in rows:
        image[top : top + 25, :] = (238, 238, 238)
        for segment in timeline[name]["segments"]:
            start = x_position(segment["start_ms"])
            end = x_position(segment["end_ms"])
            image[top : top + 25, start : max(end, start + 1)] = color
    image[80:140, :] = (248, 248, 248)
    for index, probability in enumerate(probabilities):
        column = x_position(index * float(timeline["vad"]["window_ms"]))
        row = 139 - round(float(probability) * 59)
        image[row:140, column] = (120, 120, 120)
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(output, data)


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path / "timeline.npz") as stored:
        return np.asarray(stored["model_speech_probability"])
