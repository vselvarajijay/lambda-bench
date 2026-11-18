"""Command-line interface for lambda-duplex."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from lambda_duplex import __version__
from lambda_duplex.adapters.mock import MockAdapter, SpeechRenderer
from lambda_duplex.counterpart.errors import NoiseAssetError
from lambda_duplex.metrics.conversation import (
    ConversationTimeline,
    conversation_metrics,
    metrics_to_dict,
    overlap_metrics,
    timeline_from_dict,
    timing_metrics,
)
from lambda_duplex.metrics.report import build_report, format_report
from lambda_duplex.orchestrator import MediaClock, Orchestrator, OrchestratorConfig, RunOutcome
from lambda_duplex.orchestrator.scripted import ScriptedCounterpart
from lambda_duplex.suites.base import ScriptSimulator
from lambda_duplex.suites.loader import (
    SuiteError,
    list_suites,
    load_scenario_files,
    load_suite,
    load_yaml_file,
    valid_scenarios,
    validate_scenario_data,
)
from lambda_duplex.suites.runtime import ModelTranscriber, mock_for, run_scenario
from lambda_duplex.suites.scaffold import ScaffoldError, create_suite
from lambda_duplex.trajectory.replay import verify_run
from lambda_duplex.trajectory.seeds import stream_seed
from lambda_duplex.trajectory.store import (
    RunSpec,
    TrajectoryError,
    find_run_dirs,
    load_run,
    write_run,
)

MINIMUM_PYTHON = (3, 12)
DEFAULT_SUITE_TICKS = 3000
OPTIONAL_MODULES = {
    "silero_vad": "speech activity timelines",
    "faster_whisper": "model audio transcription",
    "pipecat": "cascaded baseline adapter",
    "openai": "OpenAI Realtime adapter",
    "google.genai": "Gemini Live adapter",
    "websockets": "Grok Voice adapter",
}


@dataclass(frozen=True)
class Check:
    """One environment check reported by ``doctor``."""

    name: str
    ok: bool
    required: bool
    detail: str


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def collect_checks() -> list[Check]:
    """Return required and optional environment checks."""
    python_ok = sys.version_info[:2] >= MINIMUM_PYTHON
    checks = [
        Check(
            name="python",
            ok=python_ok,
            required=True,
            detail=f"{platform.python_version()} (requires 3.12 or newer)",
        )
    ]
    for module, purpose in OPTIONAL_MODULES.items():
        available = _module_available(module)
        state = "installed" if available else "not installed"
        checks.append(Check(module, available, False, f"{state}; used for {purpose}"))
    return checks


def _doctor(as_json: bool) -> int:
    checks = collect_checks()
    healthy = all(check.ok for check in checks if check.required)
    if as_json:
        print(json.dumps({"ok": healthy, "checks": [asdict(check) for check in checks]}, indent=2))
    else:
        for check in checks:
            status = "ok" if check.ok else ("FAIL" if check.required else "missing")
            print(f"{status:7} {check.name}: {check.detail}")
    return 0 if healthy else 1


def _print_events(outcome: RunOutcome) -> None:
    for event in outcome.events:
        details = json.dumps(event.data, sort_keys=True)
        print(f"{event.seq:04} tick={event.tick:<4} {event.source}.{event.type} {details}")


def _stored_suite_reference(reference: str, name: str) -> str:
    """Avoid storing absolute local paths in shareable manifests."""
    return name if Path(reference).is_absolute() else reference


def _speech_backend(args: argparse.Namespace, seed: int) -> SpeechRenderer | None:
    if args.mode != "audio":
        return None
    from lambda_duplex.audio.synthetic import SyntheticSpeech

    return SyntheticSpeech(seed)


def _live_transcriber(args: argparse.Namespace) -> ModelTranscriber | None:
    if args.transcript != "whisper":
        return None
    from lambda_duplex.audio.transcription import FasterWhisperTranscriber, LiveTranscriber

    return LiveTranscriber(FasterWhisperTranscriber(args.whisper_model))


def _transcript_record(
    args: argparse.Namespace, transcriber: ModelTranscriber | None
) -> dict[str, str] | None:
    if args.mode != "audio":
        return None
    if transcriber is None:
        return {"mode": "injected"}
    return {"mode": "whisper", "model": transcriber.name}


def _trial_seed(seed: int, scenario_id: str, trial: int) -> int:
    """Keep trial zero on the requested seed and derive independent later seeds."""
    if trial == 0:
        return seed
    return stream_seed(seed, f"trial:{scenario_id}:{trial}") % (2**31)


def _run_suite(args: argparse.Namespace) -> int:
    if args.k < 1:
        print("--k must be at least 1", file=sys.stderr)
        return 2
    model, _, behavior = args.model.partition(":")
    if model != "mock":
        print("only mock models are available", file=sys.stderr)
        return 2
    if args.mode == "audio" and args.audio_backend != "synthetic":
        print("audio mode currently requires --audio-backend synthetic", file=sys.stderr)
        return 2
    if args.transcript == "whisper" and args.mode != "audio":
        print("--transcript whisper requires audio mode", file=sys.stderr)
        return 2
    try:
        live_transcriber = _live_transcriber(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"cannot load transcription model: {error}", file=sys.stderr)
        return 2
    try:
        loaded = load_suite(args.suite)
        scenarios = valid_scenarios(loaded, args.split)
    except SuiteError as error:
        print(error, file=sys.stderr)
        return 2
    if args.scenario:
        scenarios = [scenario for scenario in scenarios if scenario.id == args.scenario]
    if not scenarios:
        print("no matching scenarios", file=sys.stderr)
        return 2
    exit_code = 0
    suite_ticks = DEFAULT_SUITE_TICKS if args.ticks is None else args.ticks
    behaviors = (behavior or "good").split("+")
    for scenario in scenarios:
        for trial in range(args.k):
            seed = _trial_seed(args.seed, scenario.id, trial)
            speech = _speech_backend(args, seed)
            try:
                adapter = mock_for(
                    loaded.suite, scenario, behaviors[trial % len(behaviors)], speech
                )
            except ValueError as error:
                print(error, file=sys.stderr)
                return 2
            try:
                run = asyncio.run(
                    run_scenario(
                        loaded.suite,
                        scenario,
                        adapter,
                        seed=seed,
                        max_ticks=suite_ticks,
                        condition=args.condition,
                        speech=speech,
                        noise_dir=args.noise_dir,
                        transcriber=live_transcriber,
                    )
                )
            except (NoiseAssetError, ValueError) as error:
                print(error, file=sys.stderr)
                return 2
            if args.out is not None:
                spec = RunSpec(
                    _stored_suite_reference(args.suite, loaded.suite.manifest.name),
                    args.model,
                    args.mode,
                    seed,
                    suite_ticks,
                    run.trajectory.tick_ms,
                    run.condition,
                    audio_backend=args.audio_backend if args.mode == "audio" else None,
                )
                try:
                    write_run(
                        args.out / f"{scenario.id}__seed{args.seed}__trial{trial}",
                        loaded.suite,
                        spec,
                        adapter.describe(),
                        run.trajectory,
                        run.result,
                        run.decision,
                        tracks=run.tracks,
                        effects=run.effects,
                        condition=run.condition_details,
                        transcript=_transcript_record(args, live_transcriber),
                    )
                except TrajectoryError as error:
                    print(error, file=sys.stderr)
                    return 2
            if args.k == 1:
                _print_events(run.outcome)
            passed = str(run.decision.passed).lower()
            print(
                f"scenario={scenario.id} trial={trial} status={run.outcome.status} "
                f"ended_by={run.outcome.ended_by} passed={passed}"
            )
            if run.outcome.status != "completed":
                exit_code = 1
    return exit_code


def _simulate_script(args: argparse.Namespace) -> int:
    try:
        loaded = load_suite(args.suite)
        if not isinstance(loaded.suite, ScriptSimulator) or loaded.root is None:
            raise SuiteError(f"{args.suite} does not support script simulation")
        path = args.scenario if args.scenario.is_file() else loaded.root / args.scenario
        split = path.parent.name
        if split not in {"dev", "test"}:
            raise SuiteError("scenario must be stored in a dev or test folder")
        result = validate_scenario_data(loaded, load_yaml_file(path), split)
        if result.issues or result.scenario is None:
            problems = "; ".join(f"{item.path}: {item.message}" for item in result.issues)
            raise SuiteError(f"invalid scenario: {problems}")
        script = load_yaml_file(args.script)
        rows = loaded.suite.simulate_script(result.scenario, script)
    except (
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
        SuiteError,
        yaml.YAMLError,
    ) as error:
        print(f"cannot simulate: {error}", file=sys.stderr)
        return 2
    print(json.dumps(list(rows), indent=2, sort_keys=True))
    return 0


def _new_suite(args: argparse.Namespace) -> int:
    destination = args.out or Path("suites") / args.name
    try:
        path = create_suite(args.name, destination)
    except ScaffoldError as error:
        print(error, file=sys.stderr)
        return 2
    print(f"created {path}")
    print(f"next: lambda-duplex suites validate {path}")
    return 0


def _apply_transcript_words(path: Path, segments: list[dict[str, Any]]) -> str | None:
    transcript_path = path / "transcript.jsonl"
    if not transcript_path.is_file():
        return None
    positions = {(item["start_ms"], item["end_ms"]): item for item in segments}
    sources: set[str] = set()
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "segment_index" not in row:
            continue
        segment = positions.get((row["start_ms"], row["end_ms"]))
        if segment is None:
            raise ValueError("transcript does not match the speech timeline")
        segment["words"] = row["word_count"]
        sources.add(str(row["source"]))
    return ",".join(sorted(sources)) or None


def _load_metric_timeline(path: Path) -> tuple[ConversationTimeline, str | None]:
    if path.is_file():
        return timeline_from_dict(json.loads(path.read_text(encoding="utf-8"))), None
    run = load_run(path)
    stored = json.loads((path / "timeline.json").read_text(encoding="utf-8"))
    suite = load_suite(run.manifest["suite"]["reference"]).suite
    goal_event = suite.manifest.goal_event
    goals = run.trajectory.suite_events(goal_event) if goal_event else ()
    model_segments = [dict(segment) for segment in stored["model"]["segments"]]
    word_source = _apply_transcript_words(path, model_segments)
    timeline = timeline_from_dict(
        {
            "duration_ms": stored["duration_ms"],
            "counterpart": stored["counterpart"]["segments"],
            "model": model_segments,
            "counterpart_events": stored["counterpart"].get("events", []),
            "goal_ms": goals[0].time_ms if goals else None,
            "talk_target": list(suite.manifest.talk_ratio_target),
        }
    )
    return timeline, word_source


def _metrics(args: argparse.Namespace) -> int:
    try:
        timeline, word_source = _load_metric_timeline(args.path)
        groups = {
            "all": conversation_metrics,
            "timing": timing_metrics,
            "overlap": overlap_metrics,
        }
        metrics = groups[args.group](timeline)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        SuiteError,
        TrajectoryError,
    ) as error:
        print(f"cannot compute metrics: {error}", file=sys.stderr)
        return 2
    output: dict[str, Any] = dict(metrics_to_dict(metrics))
    if word_source is not None:
        output["word_count_source"] = word_source
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _effects(args: argparse.Namespace) -> int:
    import numpy as np
    import soundfile as sf  # type: ignore[import-untyped]

    from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE, silence
    from lambda_duplex.audio.tracks import resample_pcm, to_array
    from lambda_duplex.conditions import get_condition
    from lambda_duplex.counterpart.effects import EffectChain, active_rms

    try:
        condition = get_condition(args.condition)
        audio, rate = sf.read(args.input, dtype="int16", always_2d=False)
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    mono = np.asarray(audio.mean(axis=1) if audio.ndim > 1 else audio, dtype="<i2")
    pcm = resample_pcm(mono.tobytes(), int(rate), INTERNAL_SAMPLE_RATE)
    samples = MediaClock().samples_per_tick(INTERNAL_SAMPLE_RATE)
    pcm += silence(-(len(pcm) // 2) % samples)
    chain = EffectChain(condition.effects, args.seed, samples, speech_rms=active_rms(pcm))
    size = samples * 2
    output = b"".join(chain.process(pcm[i : i + size]) for i in range(0, len(pcm), size))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, to_array(output), INTERNAL_SAMPLE_RATE, subtype="PCM_16")
    print(json.dumps(chain.metadata(), indent=2, sort_keys=True))
    return 0


def _assets(args: argparse.Namespace) -> int:
    from lambda_duplex.counterpart.effects.noise_assets import download_musan_noise

    try:
        manifest = download_musan_noise(args.out)
    except (NoiseAssetError, OSError) as error:
        print(error, file=sys.stderr)
        return 2
    print(f"installed {len(manifest['files'])} noise recordings in {args.out}")
    return 0


def _transcribe(args: argparse.Namespace) -> int:
    paths = find_run_dirs(args.path)
    if not paths:
        print(f"no runs found in {args.path}", file=sys.stderr)
        return 2
    try:
        from lambda_duplex.audio.transcription import FasterWhisperTranscriber, transcribe_run

        transcriber = None
        if args.backend == "whisper":
            transcriber = FasterWhisperTranscriber(args.whisper_model)
        for path in paths:
            rows = transcribe_run(path, transcriber)
            print(f"{path}: transcript_rows={len(rows)}")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"cannot transcribe: {error}", file=sys.stderr)
        return 2
    return 0


def _timeline(args: argparse.Namespace) -> int:
    paths = find_run_dirs(args.path)
    if not paths:
        print(f"no runs found in {args.path}", file=sys.stderr)
        return 2
    if args.plot is not None and len(paths) != 1:
        print("--plot requires exactly one run", file=sys.stderr)
        return 2
    from lambda_duplex.audio.timeline import build_timeline, load_probabilities, plot_timeline
    from lambda_duplex.audio.vad import SileroVad

    vad = SileroVad()
    for path in paths:
        try:
            timeline = build_timeline(path, vad)
        except TrajectoryError as error:
            print(error, file=sys.stderr)
            return 2
        print(
            f"{path}: model_segments={len(timeline['model']['segments'])} "
            f"counterpart_segments={len(timeline['counterpart']['segments'])}"
        )
        if args.plot is not None:
            plot_timeline(timeline, load_probabilities(path), args.plot)
    return 0


def _report(args: argparse.Namespace) -> int:
    paths = find_run_dirs(args.path)
    if not paths:
        print(f"no runs found in {args.path}", file=sys.stderr)
        return 2
    try:
        report = build_report(load_run(path) for path in paths)
    except TrajectoryError as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_report(report))
    return 0


def _replay(args: argparse.Namespace) -> int:
    paths = find_run_dirs(args.path)
    if not paths:
        print(f"no runs found in {args.path}", file=sys.stderr)
        return 2
    suite = load_suite(args.suite).suite if args.suite else None
    healthy = True
    for path in paths:
        try:
            report = asyncio.run(verify_run(path, suite, args.noise_dir))
        except (SuiteError, TrajectoryError) as error:
            print(f"{path}: {error}", file=sys.stderr)
            return 2
        healthy = healthy and report.ok
        rescore = "ok" if report.rescore_ok else "FAILED"
        replay = "ok" if report.replay_ok else "FAILED"
        print(f"{path}: rescore={rescore} replay={replay}")
        for problem in report.problems:
            print(f"  {problem}")
    return 0 if healthy else 1


def _suites(args: argparse.Namespace) -> int:
    if args.suites_command == "list":
        for name in list_suites():
            print(name)
        return 0
    try:
        files = load_scenario_files(load_suite(args.suite), args.split)
    except SuiteError as error:
        print(error, file=sys.stderr)
        return 2
    invalid = 0
    for item in files:
        for issue in item.issues:
            invalid += 1
            print(f"{item.path}: {issue.path}: {issue.message}")
    print(f"validated {len(files)} scenarios, {invalid} issues")
    return 1 if invalid else 0


def _run(args: argparse.Namespace) -> int:
    if args.ticks is not None and args.ticks < 0:
        print("--ticks must not be negative", file=sys.stderr)
        return 2
    if args.suite != "_inline_echo":
        return _run_suite(args)
    if args.model != "mock" or args.mode != "text":
        print("_inline_echo supports only the mock model in text mode", file=sys.stderr)
        return 2
    ticks = 50 if args.ticks is None else args.ticks
    script = {tick: f"line {tick // 10}" for tick in range(0, ticks, 10)}
    orchestrator = Orchestrator(
        MockAdapter(),
        ScriptedCounterpart(script),
        OrchestratorConfig(max_ticks=ticks, seed=args.seed),
    )
    outcome = asyncio.run(orchestrator.run())
    _print_events(outcome)
    print(f"status={outcome.status} ticks={outcome.ticks}")
    return 0 if outcome.status == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="lambda-duplex", description=__doc__)
    parser.add_argument("--version", action="version", version=f"lambda-duplex {__version__}")
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="check the local environment")
    doctor.add_argument("--json", action="store_true", help="print machine-readable output")
    run = commands.add_parser("run", help="run a conversation")
    run.add_argument("--suite", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--mode", choices=("text", "audio"), default="text")
    run.add_argument("--audio-backend", choices=("synthetic",))
    run.add_argument("--condition", help="audio condition; defaults to the suite setting")
    run.add_argument("--noise-dir", type=Path, help="installed noise recordings")
    run.add_argument("--transcript", choices=("injected", "whisper"), default="injected")
    run.add_argument("--whisper-model", default="small.en")
    run.add_argument("--ticks", type=int)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--split", choices=("dev", "test"), default="dev")
    run.add_argument("--scenario")
    run.add_argument("--out", type=Path, help="directory for stored run data")
    run.add_argument("--k", type=int, default=1, help="trials per scenario")
    effects = commands.add_parser("effects", help="preview audio conditions")
    effect_commands = effects.add_subparsers(dest="effects_command", required=True)
    preview = effect_commands.add_parser("preview", help="apply a condition to an audio file")
    preview.add_argument("input", type=Path)
    preview.add_argument("--condition", required=True)
    preview.add_argument("--seed", type=int, default=0)
    preview.add_argument("--out", type=Path, required=True)
    assets = commands.add_parser("assets", help="install optional data assets")
    asset_commands = assets.add_subparsers(dest="assets_command", required=True)
    noise = asset_commands.add_parser("noise", help="download MUSAN noise recordings")
    noise.add_argument("--out", type=Path, default=Path("data/noise"))
    metrics = commands.add_parser("metrics", help="compute conversation metrics")
    metrics.add_argument("path", type=Path, help="timeline JSON file or timed run directory")
    metrics.add_argument("--group", choices=("all", "timing", "overlap"), default="all")
    transcribe = commands.add_parser("transcribe", help="write a transcript for stored runs")
    transcribe.add_argument("path", type=Path)
    transcribe.add_argument("--track", choices=("model",), default="model")
    transcribe.add_argument("--backend", choices=("whisper", "injected"), default="whisper")
    transcribe.add_argument("--whisper-model", default="small.en")
    timeline = commands.add_parser("timeline", help="build speech timelines for audio runs")
    timeline.add_argument("path", type=Path)
    timeline.add_argument("--plot", type=Path, help="write a PNG for one run")
    interviewer = commands.add_parser("interviewer", help="debug interviewer scenarios")
    interviewer_commands = interviewer.add_subparsers(dest="interviewer_command", required=True)
    simulate = interviewer_commands.add_parser("simulate", help="run a disclosure script")
    simulate.add_argument("scenario", type=Path)
    simulate.add_argument("--script", type=Path, required=True)
    simulate.add_argument("--suite", default="interviewer")
    new_suite = commands.add_parser("new-suite", help="create a suite from the template")
    new_suite.add_argument("name")
    new_suite.add_argument("--out", type=Path, help="destination directory")
    report = commands.add_parser("report", help="summarize stored runs")
    report.add_argument("path", type=Path)
    report.add_argument("--json", action="store_true", help="print machine-readable output")
    replay = commands.add_parser("replay", help="verify stored runs")
    replay.add_argument("path", type=Path)
    replay.add_argument("--verify", action="store_true", help="rescore and replay the run")
    replay.add_argument("--suite", help="suite to use instead of the stored reference")
    replay.add_argument("--noise-dir", type=Path, help="installed noise recordings")
    suites = commands.add_parser("suites", help="list or validate suites")
    suite_commands = suites.add_subparsers(dest="suites_command", required=True)
    suite_commands.add_parser("list", help="list available suites")
    validate = suite_commands.add_parser("validate", help="validate suite scenarios")
    validate.add_argument("suite")
    validate.add_argument("--split", choices=("dev", "test"), default="dev")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.json)
    if args.command == "run":
        return _run(args)
    if args.command == "effects":
        return _effects(args)
    if args.command == "assets":
        return _assets(args)
    if args.command == "metrics":
        return _metrics(args)
    if args.command == "transcribe":
        return _transcribe(args)
    if args.command == "timeline":
        return _timeline(args)
    if args.command == "interviewer":
        return _simulate_script(args)
    if args.command == "new-suite":
        return _new_suite(args)
    if args.command == "report":
        return _report(args)
    if args.command == "replay":
        if not args.verify:
            parser.error("replay currently requires --verify")
        return _replay(args)
    if args.command == "suites":
        return _suites(args)
    parser.print_help()
    return 0
