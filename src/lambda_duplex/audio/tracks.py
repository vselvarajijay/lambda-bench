"""Sample-accurate conversation tracks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import soxr  # type: ignore[import-untyped]

from lambda_duplex.adapters.base import MediaFrame
from lambda_duplex.audio.pcm import sample_count, silence

TRACK_NAMES = ("model", "counterpart_clean", "counterpart_degraded", "mix")


def to_array(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").copy()


def resample_pcm(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Resample 16-bit mono PCM."""
    if source_rate == target_rate:
        return pcm
    result = soxr.resample(to_array(pcm), source_rate, target_rate, quality="HQ")
    return np.asarray(result, dtype="<i2").tobytes()


def read_track(path: Path) -> tuple[bytes, int]:
    """Read a mono track and return PCM with its sample rate."""
    data, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    return np.asarray(data, dtype="<i2").tobytes(), int(sample_rate)


class TrackRecorder:
    """Record every tick and write conversation tracks."""

    def __init__(self, sample_rate: int, samples_per_tick: int) -> None:
        self.sample_rate = sample_rate
        self.samples_per_tick = samples_per_tick
        self._clean = bytearray()
        self._degraded = bytearray()
        self._model = bytearray()

    def record(self, tick: int, incoming: MediaFrame, output: MediaFrame) -> None:
        expected = tick * self.samples_per_tick * 2
        if len(self._model) != expected:
            raise ValueError(f"track is not aligned at tick {tick}")
        self._clean.extend(incoming.clean_pcm if incoming.clean_pcm is not None else incoming.pcm)
        self._degraded.extend(incoming.pcm)
        self._model.extend(output.pcm)

    def tracks(self) -> dict[str, bytes]:
        """Return all tracks. Model audio reaches the mix one tick after it is produced."""
        degraded = bytes(self._degraded)
        delayed = (silence(self.samples_per_tick) + bytes(self._model))[: len(degraded)]
        mixed = to_array(degraded).astype(np.int32) + to_array(delayed).astype(np.int32)
        return {
            "model": bytes(self._model),
            "counterpart_clean": bytes(self._clean),
            "counterpart_degraded": degraded,
            "mix": np.clip(mixed, -32768, 32767).astype("<i2").tobytes(),
        }

    def write(self, folder: Path) -> dict[str, Any]:
        """Write FLAC tracks and return their manifest entry."""
        folder.mkdir()
        tracks: dict[str, Any] = {}
        for name, pcm in self.tracks().items():
            path = folder / f"{name}.flac"
            if pcm:
                sf.write(path, to_array(pcm), self.sample_rate, subtype="PCM_16")
            else:
                path.write_bytes(b"")
            tracks[name] = {
                "samples": sample_count(pcm),
                "sha256": hashlib.sha256(pcm).hexdigest(),
            }
        return {"sample_rate": self.sample_rate, "tracks": tracks}
