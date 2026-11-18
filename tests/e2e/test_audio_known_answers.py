from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from lambda_duplex import cli
from lambda_duplex.adapters.mock import MockAdapter
from lambda_duplex.audio.timeline import build_timeline
from lambda_duplex.audio.tracks import resample_pcm
from lambda_duplex.audio.vad import SileroVad
from lambda_duplex.suites.loader import load_suite, valid_scenarios
from lambda_duplex.suites.runtime import mock_for, run_scenario
from lambda_duplex.trajectory.store import RunSpec, write_run

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "tests/fixtures/suites/known_answer"
FIXTURE = ROOT / "tests/fixtures/audio/speech_labeled"
VAD_TOLERANCE_MS = 100


class FixtureSpeech:
    """Render named utterances from labeled real-speech chunks."""

    def __init__(self) -> None:
        audio, _ = sf.read(FIXTURE.with_suffix(".flac"), dtype="int16", always_2d=False)
        labels = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))["segments"]
        pcm = np.asarray(audio, dtype="<i2").tobytes()
        self.chunks = {}
        for name, index in {"long": 0, "medium": 1, "backchannel": 2}.items():
            label = labels[index]
            clip = pcm[label["start_ms"] * 32 : label["end_ms"] * 32]
            self.chunks[name] = resample_pcm(clip, 16_000, 24_000)

    def render(self, text: str) -> bytes:
        return self.chunks.get(text, self.chunks["medium"])


@pytest.fixture(scope="module")
def vad() -> SileroVad:
    return SileroVad()


def run_condition(path: Path, condition: str, vad: SileroVad) -> tuple[Path, dict[str, Any]]:
    loaded = load_suite(str(SUITE))
    scenario = valid_scenarios(loaded)[0]
    speech = FixtureSpeech()
    adapter: MockAdapter = mock_for(loaded.suite, scenario, "known_behavior", speech)
    run = asyncio.run(
        run_scenario(
            loaded.suite,
            scenario,
            adapter,
            max_ticks=120,
            condition=condition,
            speech=speech,
        )
    )
    assert run.outcome.status == "completed"
    assert run.outcome.ended_by == "counterpart"
    spec = RunSpec(str(SUITE), "mock:known_behavior", "audio", 0, 120, 160, condition)
    stored = write_run(
        path / condition,
        loaded.suite,
        spec,
        adapter.describe(),
        run.trajectory,
        run.result,
        run.decision,
        tracks=run.tracks,
        effects=run.effects,
        condition=run.condition_details,
    )
    return stored, build_timeline(stored, vad)


def metrics_for(path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    capsys.readouterr()
    assert cli.main(["metrics", str(path)]) == 0
    values = json.loads(capsys.readouterr().out)
    return {name: item["value"] for name, item in values.items()}


@pytest.mark.parametrize("condition", ["clean", "ablate_noise", "ablate_channel"])
def test_pipeline_recovers_known_conversation_metrics(
    tmp_path: Path,
    condition: str,
    vad: SileroVad,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = run_condition(tmp_path, condition, vad)

    metrics = metrics_for(path, capsys)

    assert metrics["response_rate"] == pytest.approx(0.75)
    assert metrics["early_start_rate"] == pytest.approx(0.25)
    assert metrics["response_latency_mean_ms"] == pytest.approx(530, abs=VAD_TOLERANCE_MS)
    assert metrics["interruption_mid_sentence_rate"] == pytest.approx(0.25)
    assert metrics["interruption_benign_rate"] == pytest.approx(0.25)
    assert metrics["interruption_pause_cut_in_rate"] == 0
    assert metrics["yield_rate"] == pytest.approx(1)
    assert metrics["yield_latency_mean_ms"] == pytest.approx(700, abs=VAD_TOLERANCE_MS)
    assert metrics["resume_rate"] == 0
    assert metrics["backchannel_per_minute"] > 0


def test_channel_effects_do_not_change_timeline(tmp_path: Path, vad: SileroVad) -> None:
    _, clean = run_condition(tmp_path, "clean", vad)
    _, noisy = run_condition(tmp_path, "ablate_noise", vad)
    _, phone = run_condition(tmp_path, "ablate_channel", vad)

    assert noisy["counterpart"] == clean["counterpart"]
    assert phone["counterpart"] == clean["counterpart"]
    assert noisy["model"]["segments"] == clean["model"]["segments"]
