from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from lambda_duplex import cli
from lambda_duplex.adapters.mock import MockAdapter
from lambda_duplex.audio.timeline import build_timeline, load_probabilities
from lambda_duplex.audio.tracks import resample_pcm
from lambda_duplex.audio.vad import (
    MODEL_SHA256,
    WINDOW_MS,
    Segment,
    SileroVad,
    VadConfig,
    segments_from_probabilities,
)
from lambda_duplex.suites.loader import load_suite, valid_scenarios
from lambda_duplex.suites.runtime import run_scenario
from lambda_duplex.trajectory.store import RunSpec, TrajectoryError, write_run

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/audio/speech_labeled.flac"
ASSET = ROOT / "src/lambda_duplex/audio/assets/silero_vad.onnx"


@pytest.fixture(scope="module")
def vad() -> SileroVad:
    return SileroVad()


def fixture_audio() -> tuple[bytes, list[dict[str, int]]]:
    data, _ = sf.read(FIXTURE, dtype="int16", always_2d=False)
    labels = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))
    return np.asarray(data, dtype="<i2").tobytes(), labels["segments"]


class FixtureSpeech:
    """Use the first labeled real-speech chunk for every utterance."""

    def __init__(self) -> None:
        pcm, labels = fixture_audio()
        first = labels[0]
        chunk = pcm[first["start_ms"] * 32 : first["end_ms"] * 32]
        self.pcm = resample_pcm(chunk, 16_000, 24_000)

    def render(self, text: str) -> bytes:
        return self.pcm


def stored_run(path: Path) -> Path:
    loaded = load_suite(str(ROOT / "suites/demo"))
    scenario = valid_scenarios(loaded)[0]
    speech = FixtureSpeech()
    adapter = MockAdapter({20: "What is the code word?"}, echo=False, speech=speech)
    run = asyncio.run(run_scenario(loaded.suite, scenario, adapter, max_ticks=50, speech=speech))
    spec = RunSpec("demo", "mock", "audio", 0, 50, 160, "clean", audio_backend="fixture")
    return write_run(
        path / "run",
        loaded.suite,
        spec,
        adapter.describe(),
        run.trajectory,
        run.result,
        run.decision,
        tracks=run.tracks,
    )


def test_bundled_model_matches_recorded_hash() -> None:
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == MODEL_SHA256


def test_labeled_speech_boundaries(vad: SileroVad) -> None:
    pcm, labels = fixture_audio()

    segments = segments_from_probabilities(vad.probabilities(pcm), len(pcm) / 32)

    assert len(segments) == len(labels)
    errors = [
        abs(value)
        for segment, label in zip(segments, labels, strict=True)
        for value in (segment.start_ms - label["start_ms"], segment.end_ms - label["end_ms"])
    ]
    assert float(np.median(errors)) <= 60
    assert max(errors) <= 100


def test_digital_silence_has_no_speech(vad: SileroVad) -> None:
    pcm = bytes(16_000 * 2)

    assert segments_from_probabilities(vad.probabilities(pcm), 1000) == []


def test_probability_count_includes_padded_window(vad: SileroVad) -> None:
    assert vad.probabilities(b"").size == 0
    assert vad.probabilities(bytes(513 * 2)).size == 2


def probabilities_for(pattern: list[tuple[float, float]]) -> np.ndarray:
    values: list[float] = []
    for milliseconds, value in pattern:
        values.extend([value] * round(milliseconds / WINDOW_MS))
    return np.array(values)


def test_short_speech_is_removed() -> None:
    probabilities = probabilities_for([(320, 0), (64, 1), (320, 0)])

    assert segments_from_probabilities(probabilities, 704) == []


def test_short_gap_stays_inside_segment() -> None:
    probabilities = probabilities_for([(320, 1), (128, 0), (320, 1), (320, 0)])

    assert segments_from_probabilities(probabilities, 1088) == [Segment(0, 768)]


def test_long_gap_splits_segments() -> None:
    probabilities = probabilities_for([(320, 1), (320, 0), (320, 1), (320, 0)])

    assert segments_from_probabilities(probabilities, 1280) == [
        Segment(0, 320),
        Segment(640, 960),
    ]


def test_hysteresis_keeps_uncertain_frames_in_speech() -> None:
    probabilities = probabilities_for([(320, 1), (640, 0.4), (320, 1), (320, 0)])

    assert segments_from_probabilities(probabilities, 1600) == [Segment(0, 1280)]


def test_padding_is_clamped_and_overlaps_merge() -> None:
    probabilities = probabilities_for([(320, 1), (224, 0), (320, 1)])
    config = VadConfig(pad_ms=30, min_silence_ms=200)

    assert segments_from_probabilities(probabilities, 800, config) == [
        Segment(0, 350),
        Segment(514, 800),
    ]


def test_timeline_uses_vad_for_model_and_plan_for_counterpart(
    tmp_path: Path, vad: SileroVad
) -> None:
    path = stored_run(tmp_path)

    timeline = build_timeline(path, vad)

    model = timeline["model"]["segments"]
    counterpart = timeline["counterpart"]["segments"]
    assert model[0]["start_ms"] == pytest.approx(3200, abs=100)
    assert model[0]["end_ms"] == pytest.approx(4700, abs=100)
    assert counterpart[0] == {
        "segment_id": 0,
        "start_ms": 0,
        "end_ms": pytest.approx(1500, abs=1),
        "interrupted": False,
        "eligible": True,
    }
    assert timeline["vad"]["model_sha256"] == MODEL_SHA256
    assert load_probabilities(path).size > 0


def test_timeline_is_repeatable(tmp_path: Path, vad: SileroVad) -> None:
    path = stored_run(tmp_path)

    first = build_timeline(path, vad)

    assert build_timeline(path, vad) == first


def test_timeline_cli_writes_png(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = stored_run(tmp_path)
    plot = tmp_path / "plots/timeline.png"

    assert cli.main(["timeline", str(path), "--plot", str(plot)]) == 0

    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "counterpart_segments=" in capsys.readouterr().out


def test_text_run_has_no_timeline(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    arguments = ["run", "--suite", str(ROOT / "suites/demo"), "--model", "mock"]
    assert cli.main([*arguments, "--out", str(out)]) == 0

    with pytest.raises(TrajectoryError, match="no audio"):
        build_timeline(next(out.iterdir()))
