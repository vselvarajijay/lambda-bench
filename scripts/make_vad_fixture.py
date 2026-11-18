#!/usr/bin/env python3
"""Build the labeled VAD fixture from a public-domain speech recording."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import soxr  # type: ignore[import-untyped]

RATE = 16_000
LAYOUT_MS = [
    (500, 1500),
    (700, 800),
    (600, 300),
    (700, 250),
    (800, 1200),
    (400, 600),
    (900, 200),
    (700, 2000),
    (600, 400),
]
TAIL_MS = 600


def voiced_regions(audio: np.ndarray) -> list[tuple[int, int]]:
    frame = RATE // 100
    count = audio.size // frame
    energy = np.array(
        [np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2)) for i in range(count)]
    )
    active = energy > max(0.02, float(np.percentile(energy, 45)))
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate([*active, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            regions.append((start * frame, index * frame))
            start = None
    return regions


def take(audio: np.ndarray, regions: list[tuple[int, int]], samples: int, skip: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    remaining = samples
    ordered = sorted(regions, key=lambda item: item[0] - item[1])
    index = skip % len(ordered)
    while remaining > 0:
        start, end = ordered[index % len(ordered)]
        start += RATE // 50
        size = min(end - start, remaining)
        if size > 0:
            parts.append(audio[start : start + size])
            remaining -= size
        index += 1
    joined = np.concatenate(parts)
    fade = RATE // 200
    joined[:fade] *= np.linspace(0, 1, fade)
    joined[-fade:] *= np.linspace(1, 0, fade)
    return joined


def main(source: Path, output: Path) -> int:
    audio, rate = sf.read(source, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(soxr.resample(audio, rate, RATE), dtype=np.float32)
    regions = voiced_regions(audio)
    generator = np.random.default_rng(7)
    chunks: list[np.ndarray] = []
    labels: list[dict[str, int]] = []
    position = 0
    for index, (gap_ms, speech_ms) in enumerate(LAYOUT_MS):
        gap = RATE * gap_ms // 1000
        speech = take(audio, regions, RATE * speech_ms // 1000, index)
        chunks.extend((np.zeros(gap, dtype=np.float32), speech))
        position += gap
        labels.append(
            {
                "start_ms": position * 1000 // RATE,
                "end_ms": (position + speech.size) * 1000 // RATE,
            }
        )
        position += speech.size
    chunks.append(np.zeros(RATE * TAIL_MS // 1000, dtype=np.float32))
    clip = np.concatenate(chunks)
    clip += generator.normal(0, 0.0005, clip.size).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, np.clip(clip, -1, 1), RATE, subtype="PCM_16")
    output.with_suffix(".json").write_text(
        json.dumps({"sample_rate": RATE, "segments": labels}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
