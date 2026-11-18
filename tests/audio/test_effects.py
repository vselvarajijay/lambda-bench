from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from lambda_duplex import cli
from lambda_duplex.audio.synthetic import SyntheticSpeech
from lambda_duplex.conditions import CONDITIONS, condition_record, get_condition
from lambda_duplex.counterpart.effects import (
    ChannelConfig,
    EffectChain,
    EffectConfig,
    MuffleConfig,
    NoiseConfig,
    active_rms,
    mulaw_decode,
    mulaw_encode,
)
from lambda_duplex.counterpart.effects.chain import active_noise_bed
from lambda_duplex.counterpart.effects.noise_assets import (
    download_musan_noise,
    load_noise_bed,
)
from lambda_duplex.counterpart.errors import NoiseAssetError
from lambda_duplex.suites.base import Scenario
from lambda_duplex.suites.loader import load_suite, valid_scenarios
from lambda_duplex.suites.runtime import mock_for, run_scenario
from lambda_duplex.trajectory.replay import verify_run
from lambda_duplex.trajectory.store import find_run_dirs, load_run

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "suites/demo"
TICK = 3840
RATE = 24_000


def tone(frequency: float, seconds: float, level: float = 0.1) -> np.ndarray:
    time = np.arange(int(RATE * seconds)) / RATE
    return np.asarray(np.sin(2 * np.pi * frequency * time) * level * 32767)


def process(signal: np.ndarray, chain: EffectChain) -> np.ndarray:
    pcm = np.rint(signal).astype("<i2").tobytes()
    pcm += bytes(-len(pcm) % (TICK * 2))
    output = b"".join(
        chain.process(pcm[index : index + TICK * 2]) for index in range(0, len(pcm), TICK * 2)
    )
    return np.frombuffer(output, dtype="<i2").astype(float)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def test_mulaw_matches_known_g711_values() -> None:
    samples = np.array([0, 1, -1, 8031, -8031, 32767, -32768])

    assert mulaw_encode(samples).tolist() == [255, 255, 127, 160, 32, 128, 0]
    assert mulaw_decode(np.array([255, 127, 128, 0])).tolist() == [0, 0, 32124, -32124]


def test_mulaw_round_trip_error_is_bounded() -> None:
    samples = np.arange(-32000, 32000, 37)

    decoded = mulaw_decode(mulaw_encode(samples)).astype(int)

    assert np.all(np.abs(decoded - samples) <= 512)


def test_no_effects_pass_audio_through() -> None:
    signal = tone(440, 0.48)

    assert np.array_equal(process(signal, EffectChain(EffectConfig(), 1)), np.rint(signal))


@pytest.mark.parametrize("name", sorted(CONDITIONS))
def test_same_seed_gives_identical_bytes(name: str) -> None:
    signal = tone(300, 3)
    config = get_condition(name).effects

    first = EffectChain(config, 7)
    second = EffectChain(config, 7)

    assert np.array_equal(process(signal, first), process(signal, second))
    assert first.metadata() == second.metadata()


def test_different_seed_changes_noise() -> None:
    config = EffectConfig(noise=NoiseConfig(bursts_per_minute=0))
    signal = tone(300, 1)

    assert not np.array_equal(
        process(signal, EffectChain(config, 1)), process(signal, EffectChain(config, 2))
    )


@pytest.mark.parametrize("target", [5.0, 15.0, 25.0])
def test_noise_matches_target_snr(target: float) -> None:
    config = NoiseConfig(snr_db=target, drift_db=0, bursts_per_minute=0)
    signal = tone(300, 8)

    chain = EffectChain(EffectConfig(noise=config), 3, speech_rms=rms(signal) / 32768)
    output = process(signal, chain)
    measured = 20 * np.log10(rms(signal) / rms(output[: signal.size] - np.rint(signal)))

    assert measured == pytest.approx(target, abs=1)


def test_supplied_noise_bed_is_used() -> None:
    bed = np.ones(RATE)
    bed[::2] = -1
    config = EffectConfig(noise=NoiseConfig(drift_db=0, bursts_per_minute=0))
    chain = EffectChain(
        config, 1, noise_bed=bed, noise_source={"type": "musan", "path": "test.wav"}
    )

    output = process(np.zeros(TICK), chain)

    assert chain.metadata()["noise_source"]["path"] == "test.wav"
    assert abs(output[0]) == pytest.approx(abs(output[1]), abs=1)


def test_bursts_are_recorded_and_audible() -> None:
    config = NoiseConfig(snr_db=80, drift_db=0, bursts_per_minute=60, burst_snr_db=(0, 0))
    chain = EffectChain(EffectConfig(noise=config), 4)

    output = process(np.zeros(RATE * 10), chain)

    assert chain.metadata()["bursts"]
    assert np.max(np.abs(output)) > 1000


def test_telephony_passes_voice_band_and_removes_high_frequency() -> None:
    config = EffectConfig(channel=ChannelConfig(drop_rate=0))
    voice = process(tone(1000, 1), EffectChain(config, 1))[2000:-2000]
    high = process(tone(6000, 1), EffectChain(config, 1))[2000:-2000]

    assert 20 * np.log10(rms(voice) / rms(tone(1000, 1))) == pytest.approx(0, abs=3)
    assert 20 * np.log10(rms(high) / rms(tone(6000, 1))) < -25


def test_drop_schedule_matches_output_and_rate() -> None:
    config = ChannelConfig(telephony=False, drop_rate=0.1, drop_burst_ms=100)
    chain = EffectChain(EffectConfig(channel=config), 9)

    output = process(tone(500, 120, 0.3), chain)

    dropped = chain.metadata()["dropped_packets"]
    packets = output.size // 480
    assert len(dropped) / packets == pytest.approx(0.1, abs=0.03)
    runs = np.split(np.array(dropped), np.flatnonzero(np.diff(dropped) != 1) + 1)
    assert np.mean([run.size for run in runs]) * 20 == pytest.approx(100, abs=30)
    for packet in range(packets):
        silent = not np.any(output[packet * 480 : (packet + 1) * 480])
        assert silent == (packet in set(dropped))


def test_muffling_reduces_high_frequencies_when_active() -> None:
    config = MuffleConfig(events_per_minute=1000, duration_ms=5000, cutoff_hz=800)
    chain = EffectChain(EffectConfig(muffle=config), 1)

    output = process(tone(5000, 2), chain)[RATE:]

    assert chain.metadata()["muffles"]
    assert rms(output) < rms(tone(5000, 1)) * 0.1


def test_wrong_tick_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="one tick"):
        EffectChain(EffectConfig(), 1).process(b"\0\0")


def test_condition_presets_match_plan() -> None:
    assert get_condition("clean").effects == EffectConfig()
    realistic = get_condition("realistic")
    assert realistic.voices == "varied"
    assert realistic.turn_taking == "free"
    assert realistic.effects.noise and realistic.effects.muffle and realistic.effects.channel
    channel = get_condition("ablate_channel").effects
    assert channel.channel is not None and channel.noise is None
    assert get_condition("ablate_noise").effects.channel is None
    assert get_condition("ablate_accent").voices == "varied"
    assert get_condition("ablate_turns").turn_taking == "free"
    with pytest.raises(ValueError, match="unknown condition"):
        get_condition("missing")


def run_condition(out: Path, condition: str = "realistic") -> Path:
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]
    extra = ["--audio-backend", "synthetic", "--condition", condition, "--out", str(out)]
    assert cli.main([*arguments, *extra]) == 0
    return find_run_dirs(out)[0]


def test_degraded_audio_reaches_track_while_clean_track_stays_clean(tmp_path: Path) -> None:
    path = run_condition(tmp_path / "runs")
    stored = load_run(path)

    tracks = stored.manifest["audio"]["tracks"]

    assert stored.manifest["run"]["condition"] == "realistic"
    assert stored.manifest["effects"]["config"]["channel"]["telephony"] is True
    assert tracks["counterpart_clean"]["sha256"] != tracks["counterpart_degraded"]["sha256"]
    assert stored.decision.passed is True


def test_condition_run_replays_exactly(tmp_path: Path) -> None:
    report = asyncio.run(verify_run(run_condition(tmp_path / "runs"), load_suite(str(DEMO)).suite))

    assert report.ok, report.problems


def test_changed_effect_metadata_fails_replay(tmp_path: Path) -> None:
    path = run_condition(tmp_path / "runs")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["effects"]["seed"] = 99
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = asyncio.run(verify_run(path, load_suite(str(DEMO)).suite))

    assert report.replay_ok is False
    assert "effects differ" in report.problems[0]


def test_preview_is_repeatable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = ROOT / "tests/fixtures/speech.wav"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    arguments = ["effects", "preview", str(source), "--condition", "realistic", "--seed", "7"]

    assert cli.main([*arguments, "--out", str(first)]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert cli.main([*arguments, "--out", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()
    assert metadata["seed"] == 7
    assert sf.info(first).samplerate == RATE


def make_archive(path: Path) -> str:
    wav = io.BytesIO()
    sf.write(wav, tone(500, 2).astype("<i2"), RATE, format="WAV", subtype="PCM_16")
    entries = {
        "musan/noise/free-sound/a.wav": wav.getvalue(),
        "musan/noise/free-sound/LICENSE": b"terms",
        "musan/README": b"corpus terms",
        "musan/speech/ignored.wav": wav.getvalue(),
        "musan/noise/../../escape.wav": wav.getvalue(),
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path.as_uri()


def test_noise_download_keeps_only_noise_and_attribution(tmp_path: Path) -> None:
    target = tmp_path / "noise"

    manifest = download_musan_noise(target, make_archive(tmp_path / "musan.tar.gz"))

    entry = manifest["files"][0]
    assert [item["path"] for item in manifest["files"]] == ["free-sound/a.wav"]
    assert entry["attribution_files"] == ["free-sound/LICENSE"]
    assert manifest["corpus_files"] == ["_corpus/README"]
    assert not (tmp_path / "escape.wav").exists()
    bed = load_noise_bed(target, 3)
    assert bed.samples.size > 0
    assert bed.record["path"] == "free-sound/a.wav"
    assert bed.record["attribution_files"] == ["free-sound/LICENSE"]


def test_noise_download_refuses_non_empty_destination(tmp_path: Path) -> None:
    target = tmp_path / "noise"
    target.mkdir()
    (target / "keep").write_text("x", encoding="utf-8")

    with pytest.raises(NoiseAssetError, match="not empty"):
        download_musan_noise(target, make_archive(tmp_path / "musan.tar.gz"))


def test_missing_noise_assets_are_reported(tmp_path: Path) -> None:
    with pytest.raises(NoiseAssetError):
        load_noise_bed(tmp_path, 0)


def test_speech_remains_detectable_shape_after_telephony() -> None:
    pcm = SyntheticSpeech().render("hello there friend")
    signal = np.frombuffer(pcm, dtype="<i2").astype(float)
    config = EffectConfig(channel=ChannelConfig(drop_rate=0))

    output = process(signal, EffectChain(config, 1))

    assert rms(output[: signal.size]) > rms(signal) * 0.5


def noise_only(seed: int = 1, speech_rms: float = 0.05) -> EffectChain:
    config = EffectConfig(noise=NoiseConfig(drift_db=0, bursts_per_minute=0))
    return EffectChain(config, seed, speech_rms=speech_rms)


def test_noise_level_does_not_reveal_or_follow_speech() -> None:
    quiet = np.zeros(TICK * 8)
    speech = quiet.copy()
    speech[TICK * 3 : TICK * 5] = tone(300, TICK * 2 / RATE, 0.5)

    silent = process(quiet, noise_only())
    spoken = process(speech, noise_only())

    assert np.array_equal(silent[: TICK * 3], spoken[: TICK * 3])
    assert np.array_equal(silent[TICK * 5 :], spoken[TICK * 5 :])


def test_active_rms_ignores_silence() -> None:
    signal = np.concatenate((tone(300, 1, 0.1), np.zeros(RATE * 9)))

    assert active_rms(signal.astype("<i2").tobytes()) == pytest.approx(0.1 / np.sqrt(2), rel=0.05)


def test_nonstationary_noise_recording_keeps_active_level() -> None:
    generator = np.random.default_rng(1)
    bed = np.zeros(RATE * 120)
    bed[: RATE * 12] = generator.normal(0, 0.1, RATE * 12)
    prepared = active_noise_bed(bed)
    config = EffectConfig(noise=NoiseConfig(snr_db=15, drift_db=0, bursts_per_minute=0))

    for seed in range(6):
        chain = EffectChain(config, seed, noise_bed=prepared, speech_rms=0.1)
        output = process(np.zeros(RATE * 5), chain)
        measured = 20 * np.log10(0.1 * 32768 / rms(output))
        assert measured == pytest.approx(15, abs=1)


@pytest.mark.parametrize("bed", [np.zeros(RATE), np.ones(10)])
def test_unusable_noise_recording_is_rejected(bed: np.ndarray) -> None:
    with pytest.raises(ValueError):
        active_noise_bed(bed)


def test_muffle_transition_does_not_comb_filter() -> None:
    config = MuffleConfig(events_per_minute=1000, duration_ms=480, cutoff_hz=3000, transition_ms=40)
    chain = EffectChain(EffectConfig(muffle=config), 1)
    signal = tone(387.1, 4, 0.2)

    output = process(signal, chain)[200 : signal.size]
    window = RATE // 50
    levels = [rms(output[i : i + window]) for i in range(0, output.size - window, window)]

    assert min(levels) > rms(signal) * 0.8
    muffle = chain.metadata()["muffles"][0]
    assert muffle["duration_ms"] == muffle["ticks"] * 160


def test_telephony_keeps_top_of_voice_band() -> None:
    config = EffectConfig(channel=ChannelConfig(drop_rate=0))

    for frequency in (3300, 3400):
        output = process(tone(frequency, 1), EffectChain(config, 1))[2000:-2000]
        assert 20 * np.log10(rms(output) / rms(tone(frequency, 1))) > -2


def test_metadata_describes_applied_settings() -> None:
    chain = EffectChain(get_condition("realistic").effects, 5, speech_rms=0.07)
    process(np.zeros(TICK), chain)

    metadata = chain.metadata()

    assert metadata["speech_rms"] == 0.07
    assert metadata["latency_samples"] == 31 + 126
    assert isinstance(metadata["noise_offset"], int)
    assert metadata["noise_source"] == {"type": "generated"}


def test_condition_records_are_honest_about_missing_behavior() -> None:
    realistic = condition_record(get_condition("realistic"))
    accent = condition_record(get_condition("ablate_accent"))

    assert realistic["requested"]["turn_taking"] == "free"
    assert realistic["applied"]["turn_taking"] == "free"
    assert realistic["applied"]["speech_inserts"] is True
    assert condition_record(get_condition("clean"))["applied"] == {
        "voices": "single",
        "turn_taking": "scripted",
    }
    assert realistic["unimplemented"] == ["varied_voices"]
    assert accent["unimplemented"] == ["varied_voices"]
    assert condition_record(get_condition("clean"))["unimplemented"] == []


def test_stored_audio_run_records_condition(tmp_path: Path) -> None:
    stored = load_run(run_condition(tmp_path / "runs", "ablate_turns"))

    assert stored.manifest["condition"]["applied"]["turn_taking"] == "free"
    assert stored.manifest["condition"]["unimplemented"] == []


def test_effect_schedule_varies_by_scenario() -> None:
    loaded = load_suite(str(DEMO))
    scenario = valid_scenarios(loaded)[0]
    other = replace(scenario, id="another_scenario")

    def effects(item: Scenario) -> dict[str, object] | None:
        speech = SyntheticSpeech(1)
        adapter = mock_for(loaded.suite, item, speech=speech)
        run = asyncio.run(
            run_scenario(loaded.suite, item, adapter, seed=1, condition="realistic", speech=speech)
        )
        return run.effects

    first = effects(scenario)
    second = effects(other)
    assert first is not None and second is not None
    assert first["seed"] != second["seed"]


def install_noise(tmp_path: Path) -> Path:
    target = tmp_path / "noise"
    download_musan_noise(target, make_archive(tmp_path / "musan.tar.gz"))
    return target


def noise_run(tmp_path: Path, noise: Path) -> Path:
    out = tmp_path / "runs"
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]
    extra = ["--audio-backend", "synthetic", "--condition", "ablate_noise"]
    assert cli.main([*arguments, *extra, "--noise-dir", str(noise), "--out", str(out)]) == 0
    return find_run_dirs(out)[0]


def test_recorded_noise_run_replays_with_attribution(tmp_path: Path) -> None:
    noise = install_noise(tmp_path)
    path = noise_run(tmp_path, noise)

    source = load_run(path).manifest["effects"]["noise_source"]
    report = asyncio.run(verify_run(path, load_suite(str(DEMO)).suite, noise))

    assert source["attribution_files"] == ["free-sound/LICENSE"]
    assert report.ok, report.problems


def test_recorded_noise_replay_requires_assets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = noise_run(tmp_path, install_noise(tmp_path))

    assert cli.main(["replay", str(path), "--verify", "--suite", str(DEMO)]) == 2
    assert "--noise-dir" in capsys.readouterr().err


def test_changed_noise_file_is_rejected(tmp_path: Path) -> None:
    noise = install_noise(tmp_path)
    (noise / "free-sound/a.wav").write_bytes(b"changed")

    with pytest.raises(NoiseAssetError, match="hash"):
        load_noise_bed(noise, 0)


def test_archive_without_attribution_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "musan.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("musan/noise/free-sound/a.wav")
        info.size = 1
        output.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(NoiseAssetError, match="attribution"):
        download_musan_noise(tmp_path / "noise", archive.as_uri())
    assert not (tmp_path / "noise").exists()


def test_duplicate_archive_member_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "musan.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for _ in range(2):
            info = tarfile.TarInfo("musan/noise/free-sound/a.wav")
            info.size = 1
            output.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(NoiseAssetError, match="repeats"):
        download_musan_noise(tmp_path / "noise", archive.as_uri())


def test_truncated_download_can_be_retried(tmp_path: Path) -> None:
    good = tmp_path / "musan.tar.gz"
    uri = make_archive(good)
    broken = tmp_path / "broken.tar.gz"
    broken.write_bytes(good.read_bytes()[:40])
    target = tmp_path / "noise"

    with pytest.raises(NoiseAssetError):
        download_musan_noise(target, broken.as_uri())

    assert not target.exists()
    assert download_musan_noise(target, uri)["files"]


def test_text_mode_rejects_degraded_condition(capsys: pytest.CaptureFixture[str]) -> None:
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--condition", "realistic"]

    assert cli.main(arguments) == 2
    assert "clean condition" in capsys.readouterr().err


def test_unknown_condition_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    arguments = ["run", "--suite", str(DEMO), "--model", "mock", "--mode", "audio"]
    extra = ["--audio-backend", "synthetic", "--condition", "missing"]

    assert cli.main([*arguments, *extra]) == 2
    assert "unknown condition" in capsys.readouterr().err


def wav_bytes(samples: np.ndarray) -> bytes:
    output = io.BytesIO()
    sf.write(output, samples, RATE, format="WAV", subtype="PCM_16")
    return output.getvalue()


def steady_noise(level: float, seconds: int = 20, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, level, RATE * seconds)


def transient_noise(level: float) -> np.ndarray:
    audio = steady_noise(level)
    audio[RATE : RATE + RATE // 2] = np.random.default_rng(2).normal(0, 0.8, RATE // 2)
    return audio


def archive_with(path: Path, recordings: dict[str, np.ndarray]) -> str:
    entries = {f"musan/noise/set/{name}": wav_bytes(audio) for name, audio in recordings.items()}
    entries["musan/noise/set/LICENSE"] = b"terms"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path.as_uri()


@pytest.mark.parametrize("level", [0.05, 0.02])
def test_recording_with_loud_transient_is_unusable(level: float) -> None:
    with pytest.raises(ValueError):
        active_noise_bed(transient_noise(level))


def test_steady_recording_is_usable() -> None:
    assert rms(active_noise_bed(steady_noise(0.05))) == pytest.approx(1)


def test_noise_selection_skips_unusable_recording(tmp_path: Path) -> None:
    uri = archive_with(
        tmp_path / "musan.tar.gz",
        {"a.wav": transient_noise(0.05), "b.wav": steady_noise(0.05)},
    )
    target = tmp_path / "noise"
    download_musan_noise(target, uri)

    bed = load_noise_bed(target, 0)

    assert bed.record["path"] == "set/b.wav"
    assert bed.record["skipped_recordings"] == 1


def test_all_unusable_recordings_are_reported(tmp_path: Path) -> None:
    uri = archive_with(tmp_path / "musan.tar.gz", {"a.wav": transient_noise(0.05)})
    target = tmp_path / "noise"
    download_musan_noise(target, uri)

    with pytest.raises(NoiseAssetError, match="no usable"):
        load_noise_bed(target, 0)


def test_incomplete_noise_manifest_is_reported(tmp_path: Path) -> None:
    noise = install_noise(tmp_path)
    manifest = json.loads((noise / "manifest.json").read_text(encoding="utf-8"))
    del manifest["files"][0]["attribution_files"]
    (noise / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NoiseAssetError, match="attribution_files"):
        load_noise_bed(noise, 0)


def test_base_install_runs_text_mode_without_audio_packages(tmp_path: Path) -> None:
    runs = str(tmp_path / "runs")
    program = f"""
import sys
for name in ("numpy", "soundfile", "soxr", "onnxruntime"):
    sys.modules[name] = None
from lambda_duplex import cli
assert cli.main(["doctor"]) == 0
arguments = ["run", "--suite", {str(DEMO)!r}, "--model", "mock", "--out", {runs!r}]
assert cli.main(arguments) == 0
assert cli.main(["replay", {runs!r}, "--verify", "--suite", {str(DEMO)!r}]) == 0
assert cli.main(["report", {runs!r}]) == 0
"""

    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
