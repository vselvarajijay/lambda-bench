from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from lambda_duplex import cli
from lambda_duplex.adapters.base import MediaFrame, StepResult
from lambda_duplex.adapters.mock import MockAdapter
from lambda_duplex.audio.pcm import PlayoutBuffer, sample_count, silence
from lambda_duplex.audio.synthetic import SyntheticSpeech
from lambda_duplex.audio.tracks import read_track, resample_pcm, to_array
from lambda_duplex.orchestrator import Orchestrator, OrchestratorConfig
from lambda_duplex.suites.base import (
    ConversationView,
    DeliveredSpan,
    ModelUtterance,
    PolicyOutput,
    SpeechSegment,
    SuiteEvent,
)
from lambda_duplex.suites.loader import load_suite, valid_scenarios
from lambda_duplex.suites.runtime import PolicyCounterpart, ScenarioRun, mock_for, run_scenario
from lambda_duplex.trajectory.replay import verify_run
from lambda_duplex.trajectory.store import (
    RunSpec,
    TrajectoryError,
    find_run_dirs,
    load_run,
    write_run,
)

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
    assert run.tracks is not None
    spoken = [
        event.tick
        for event in run.outcome.events
        if event.source == "counterpart" and event.type == "spoke"
    ]

    duration = sample_count(SyntheticSpeech(1).render("It is lantern."))
    last_tick = spoken[-1] + (duration + TICK_SAMPLES - 1) // TICK_SAMPLES - 1
    assert disclosure.data == {"tick": last_tick}
    assert disclosure.tick == last_tick + 1


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
        pcm, sample_rate = read_track(path / "audio" / f"{name}.flac")
        assert sample_rate == 24_000
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
    assert any("counterpart_clean" in problem for problem in report.problems)


def test_audio_requires_supported_backend(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]) == 2
    assert "--audio-backend" in capsys.readouterr().err


class RecordingAdapter(MockAdapter):
    def __init__(self, fail_at: int | None = None) -> None:
        super().__init__(echo=False, speech=SyntheticSpeech(1))
        self.frames: list[MediaFrame] = []
        self.fail_at = fail_at

    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        self.frames.append(frame)
        if tick == self.fail_at:
            raise RuntimeError("model crashed")
        return await super().step(frame, tick)


class SegmentPolicy:
    def __init__(self, outputs: dict[int, PolicyOutput]) -> None:
        self.outputs = outputs
        self.spans: list[DeliveredSpan] = []

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        return self.outputs.get(view.tick, PolicyOutput())

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        return None if view.tick else self.outputs.get(0)

    def on_delivered(self, span: DeliveredSpan) -> tuple[SuiteEvent, ...]:
        self.spans.append(span)
        return ()

    def is_finished(self, view: ConversationView) -> bool:
        return False


def verify_problems(path: Path) -> tuple[str, ...]:
    report = asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))
    assert not report.ok
    return report.problems


def test_text_is_released_when_model_audio_finishes() -> None:
    speech = SyntheticSpeech(1)
    text = "What is the code word?"
    adapter = MockAdapter({1: text}, echo=False, speech=speech)
    duration = sample_count(speech.render(text))
    last_tick = 1 + (duration + TICK_SAMPLES - 1) // TICK_SAMPLES - 1

    run = audio_run(adapter)

    spoken = [e for e in run.outcome.events if e.source == "model" and e.type == "spoke"]
    heard = [e for e in run.outcome.events if e.type == "heard"]
    assert [event.tick for event in spoken] == [last_tick]
    assert [event.tick for event in heard] == [last_tick + 1]
    assert run.decision.passed is True


def test_adapter_receives_only_the_audio_it_should_hear() -> None:
    adapter = RecordingAdapter()

    audio_run(adapter, ticks=3)

    assert all(frame.text is None and frame.clean_pcm is None for frame in adapter.frames)
    assert any(any(frame.pcm) for frame in adapter.frames)


def test_truncation_drops_audio_and_text_and_is_recorded() -> None:
    adapter = MockAdapter(
        {1: "What is the code word?"}, echo=False, speech=SyntheticSpeech(1), truncate_at={3}
    )

    run = audio_run(adapter, ticks=14)
    assert run.tracks is not None

    truncated = [event for event in run.outcome.events if event.type == "truncated"]
    assert truncated[0].tick == 3
    assert truncated[0].data["ms"] > 0
    assert not [e for e in run.outcome.events if e.source == "model" and e.type == "spoke"]
    assert not any(run.tracks.tracks()["model"][3 * TICK_SAMPLES * 2 :])
    assert run.decision.passed is False


def test_truncated_run_replays(tmp_path: Path) -> None:
    loaded = load_suite(str(DEMO))
    scenario = valid_scenarios(loaded)[0]
    speech = SyntheticSpeech(0)
    adapter = MockAdapter({1: "What is the code word?"}, False, speech, truncate_at={3})
    run = asyncio.run(run_scenario(loaded.suite, scenario, adapter, max_ticks=14, speech=speech))
    spec = RunSpec("demo", "mock", "audio", 0, 14, 160, "clean", audio_backend="synthetic")

    path = write_run(
        tmp_path / "run",
        loaded.suite,
        spec,
        adapter.describe(),
        run.trajectory,
        run.result,
        run.decision,
        tracks=run.tracks,
        condition=run.condition_details,
    )

    report = asyncio.run(verify_run(path, loaded.suite))
    assert report.ok, report.problems


def test_credit_waits_until_final_audio_reaches_model() -> None:
    speech = SyntheticSpeech(1)
    policy = SegmentPolicy({0: PolicyOutput((SpeechSegment("one two", ("fact",)),))})
    counterpart = PolicyCounterpart(policy, 160, speech)
    ticks = (sample_count(speech.render("one two")) + TICK_SAMPLES - 1) // TICK_SAMPLES

    async def play() -> None:
        for tick in range(ticks):
            if tick:
                await counterpart.hear(MediaFrame(tick - 1), tick)
            await counterpart.render(tick)
        assert policy.spans == []
        assert counterpart.is_finished() is False
        await counterpart.hear(MediaFrame(ticks - 1), ticks)

    asyncio.run(play())

    assert policy.spans[0].tags == ("fact",)
    assert policy.spans[0].end_tick == ticks - 1


def test_failed_final_tick_earns_no_delivery_credit() -> None:
    plain = audio_run(MockAdapter({1: "What is the code word?"}, False, SyntheticSpeech(1)))
    disclosure = plain.trajectory.suite_events("code_word_disclosed")[0]

    class FailingScript(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__(fail_at=disclosure.tick - 1)
            self._script = {1: "What is the code word?"}

    run = audio_run(FailingScript())

    assert run.outcome.status == "adapter_error"
    assert run.trajectory.suite_events("code_word_disclosed") == ()


def test_preempted_span_ends_before_replacement_starts() -> None:
    first = PolicyOutput((SpeechSegment("one two three four five six", ("fact",)),))
    second = PolicyOutput((SpeechSegment("now"),), preempt=True)
    policy = SegmentPolicy({0: first, 3: second})
    counterpart = PolicyCounterpart(policy, 160, SyntheticSpeech(1))

    async def play() -> None:
        for tick in range(4):
            if tick:
                await counterpart.hear(MediaFrame(tick - 1, text="go"), tick)
            await counterpart.render(tick)

    asyncio.run(play())

    assert policy.spans[0].start_tick == 0
    assert policy.spans[0].end_tick == 2
    assert policy.spans[0].interrupted is True
    assert policy.spans[0].tags == ()


@pytest.mark.parametrize("rate", [16_000, 48_000, None])
def test_wrong_model_sample_rate_is_an_adapter_error(rate: int | None) -> None:
    class WrongRate(MockAdapter):
        async def step(self, frame: MediaFrame, tick: int) -> StepResult:
            return StepResult(MediaFrame(tick, silence(TICK_SAMPLES), rate))

    run = audio_run(WrongRate())

    assert run.outcome.status == "adapter_error"
    assert "sample rate" in str(run.outcome.error)


def test_short_clean_audio_is_a_counterpart_error() -> None:
    class ShortClean:
        async def start(self, seed: int) -> None:
            return None

        async def hear(self, frame: MediaFrame, tick: int) -> None:
            return None

        async def render(self, tick: int) -> MediaFrame:
            return MediaFrame(tick, silence(TICK_SAMPLES), 24_000, clean_pcm=b"\0\0")

        def is_finished(self) -> bool:
            return False

        def drain_events(self) -> list[tuple[str, str, dict[str, object]]]:
            return []

        async def stop(self) -> None:
            return None

    config = OrchestratorConfig(max_ticks=1, audio_sample_rate=24_000)
    adapter = MockAdapter(echo=False, speech=SyntheticSpeech())

    outcome = asyncio.run(Orchestrator(adapter, ShortClean(), config).run())

    assert outcome.status == "counterpart_error"


@pytest.mark.parametrize("name", ["mix", "counterpart_clean", "counterpart_degraded", "model"])
def test_changed_track_file_fails_verification(tmp_path: Path, name: str) -> None:
    path = store_demo(tmp_path / "runs")
    file_path = path / "audio" / f"{name}.flac"
    pcm, rate = read_track(file_path)
    sf.write(file_path, np.zeros(sample_count(pcm), dtype="<i2"), rate, subtype="PCM_16")

    assert any(f"{name} track differs" in problem for problem in verify_problems(path))


def test_wrong_track_sample_rate_fails_verification(tmp_path: Path) -> None:
    path = store_demo(tmp_path / "runs")
    file_path = path / "audio" / "model.flac"
    pcm, _ = read_track(file_path)
    sf.write(file_path, to_array(pcm), 8_000, subtype="PCM_16")

    assert any("sample rate 8000" in problem for problem in verify_problems(path))


def test_missing_track_is_reported_by_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = store_demo(tmp_path / "runs")
    (path / "audio" / "mix.flac").unlink()

    with pytest.raises(TrajectoryError, match=r"mix\.flac"):
        asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))
    assert cli.main(["replay", str(path), "--verify", "--suite", str(DEMO)]) == 2
    assert "mix.flac" in capsys.readouterr().err


def test_zero_tick_audio_run_replays(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]
    extra = ["--audio-backend", "synthetic", "--ticks", "0", "--out", str(out)]

    assert cli.main([*arguments, *extra]) == 0
    report = asyncio.run(verify_run(find_run_dirs(out)[0], load_suite(str(DEMO)).suite))

    assert report.ok, report.problems


def test_synthetic_speech_is_finite_at_other_rates() -> None:
    assert sample_count(SyntheticSpeech(sample_rate=44_100).render("hello")) > 0
    assert np.isfinite(SyntheticSpeech(sample_rate=44_100)._word("hello", 0)).all()


def test_audio_run_without_audio_manifest_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    arguments = ["run", "--suite", str(DEMO), "--model", "mock:bad", "--mode", "audio"]
    extra = ["--audio-backend", "synthetic", "--ticks", "20", "--out", str(out)]
    assert cli.main([*arguments, *extra]) == 0
    path = find_run_dirs(out)[0]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    del manifest["audio"]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TrajectoryError, match="no audio manifest"):
        asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))
