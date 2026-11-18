from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lambda_duplex import cli
from lambda_duplex.adapters.base import MediaFrame, StepResult
from lambda_duplex.adapters.mock import MockAdapter
from lambda_duplex.audio.pcm import PlayoutBuffer, sample_count, silence
from lambda_duplex.audio.synthetic import SyntheticSpeech
from lambda_duplex.audio.tracks import read_track, resample_pcm, to_array
from lambda_duplex.suites.loader import load_suite, valid_scenarios
from lambda_duplex.suites.runtime import ScenarioRun, mock_for, run_scenario
from lambda_duplex.trajectory.replay import verify_run
from lambda_duplex.trajectory.store import find_run_dirs, load_run

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "suites/demo"
TICK_SAMPLES = 3840


def audio_run(adapter: MockAdapter | None = None, ticks: int = 40) -> ScenarioRun:
    loaded = load_suite(str(DEMO))
    scenario = valid_scenarios(loaded)[0]
    speech = SyntheticSpeech(1)
    model = adapter or mock_for(loaded.suite, scenario, speech=speech)
    return asyncio.run(
        run_scenario(loaded.suite, scenario, model, seed=1, max_ticks=ticks, speech=speech)
    )


def first_sound(pcm: bytes) -> int:
    return int(np.flatnonzero(to_array(pcm))[0])


def store_demo(out: Path) -> Path:
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]
    assert cli.main([*arguments, "--audio-backend", "synthetic", "--out", str(out)]) == 0
    return find_run_dirs(out)[0]


def test_playout_releases_one_tick_and_pads() -> None:
    buffer = PlayoutBuffer(4)
    buffer.push(b"\x01\x00" * 6)

    assert buffer.pop_tick() == b"\x01\x00" * 4
    assert buffer.pop_tick() == b"\x01\x00" * 2 + silence(2)
    assert buffer.pop_tick() == silence(4)


def test_playout_truncation_reports_dropped_audio() -> None:
    buffer = PlayoutBuffer(100, 1000)
    buffer.push(silence(250))

    assert buffer.truncate() == 250
    assert buffer.pending_samples == 0


def test_odd_pcm_is_rejected() -> None:
    with pytest.raises(ValueError, match="whole"):
        sample_count(b"\x00")


def test_synthetic_speech_is_repeatable_and_text_dependent() -> None:
    speech = SyntheticSpeech(3)

    assert speech.render("hello there") == SyntheticSpeech(3).render("hello there")
    assert speech.render("hello there") != speech.render("hello friend")
    assert speech.render("hello there") != SyntheticSpeech(4).render("hello there")
    assert sample_count(speech.render("hello there")) == 24_000 * 540 // 1000


def test_resampling_preserves_duration() -> None:
    pcm = SyntheticSpeech().render("one two three")

    result = resample_pcm(pcm, 24_000, 16_000)

    assert sample_count(result) == pytest.approx(sample_count(pcm) * 2 / 3, abs=2)
    assert resample_pcm(pcm, 24_000, 24_000) == pcm


def test_tracks_match_the_clock_to_the_sample() -> None:
    run = audio_run()
    assert run.tracks is not None

    tracks = run.tracks.tracks()

    assert run.outcome.status == "completed"
    assert run.decision.passed is True
    assert {sample_count(pcm) for pcm in tracks.values()} == {run.outcome.ticks * TICK_SAMPLES}


def test_model_and_counterpart_audio_are_aligned() -> None:
    adapter = MockAdapter({2: "What is the code word?"}, echo=False, speech=SyntheticSpeech(1))
    run = audio_run(adapter)
    assert run.tracks is not None
    tracks = run.tracks.tracks()

    assert first_sound(tracks["counterpart_clean"]) < TICK_SAMPLES
    assert 2 * TICK_SAMPLES <= first_sound(tracks["model"]) < 3 * TICK_SAMPLES


def test_mix_hears_model_one_tick_later() -> None:
    adapter = MockAdapter({20: "What is the code word?"}, echo=False, speech=SyntheticSpeech(1))
    run = audio_run(adapter, ticks=23)
    assert run.tracks is not None
    tracks = run.tracks.tracks()
    delayed = silence(TICK_SAMPLES) + tracks["model"][: -TICK_SAMPLES * 2]
    expected = to_array(tracks["counterpart_degraded"]).astype(int) + to_array(delayed).astype(int)

    assert np.array_equal(to_array(tracks["mix"]), np.clip(expected, -32768, 32767))
    assert first_sound(delayed) >= 21 * TICK_SAMPLES


def test_multi_tick_speech_is_credited_when_playback_ends() -> None:
    run = audio_run()

    disclosure = run.trajectory.suite_events("code_word_disclosed")[0]
    spoken = [
        event.tick
        for event in run.outcome.events
        if event.source == "counterpart" and event.type == "spoke"
    ]

    duration = sample_count(SyntheticSpeech(1).render("It is lantern."))
    assert disclosure.tick == spoken[-1] + (duration + TICK_SAMPLES - 1) // TICK_SAMPLES - 1
    assert disclosure.data == {"tick": disclosure.tick}


def test_wrong_model_frame_size_is_an_adapter_error() -> None:
    class ShortAdapter(MockAdapter):
        async def step(self, frame: MediaFrame, tick: int) -> StepResult:
            return StepResult(MediaFrame(tick, b"\x00\x00", 24_000))

    run = audio_run(ShortAdapter())

    assert run.outcome.status == "adapter_error"
    assert "audio bytes" in str(run.outcome.error)


def test_flac_tracks_round_trip_and_match_manifest(tmp_path: Path) -> None:
    path = store_demo(tmp_path / "runs")
    audio = load_run(path).manifest["audio"]

    for name, details in audio["tracks"].items():
        pcm = read_track(path / "audio" / f"{name}.flac")
        assert sample_count(pcm) == details["samples"]
        assert hashlib.sha256(pcm).hexdigest() == details["sha256"]


def test_audio_run_replays_exactly(tmp_path: Path) -> None:
    path = store_demo(tmp_path / "runs")

    report = asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))

    assert report.ok, report.problems


def test_changed_manifest_hash_fails_replay(tmp_path: Path) -> None:
    path = store_demo(tmp_path / "runs")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["audio"]["tracks"]["counterpart_clean"]["sha256"] = "0" * 64
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))

    assert report.replay_ok is False
    assert "counterpart_clean audio differs" in report.problems[0]


def test_audio_requires_supported_backend(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]) == 2
    assert "--audio-backend" in capsys.readouterr().err
