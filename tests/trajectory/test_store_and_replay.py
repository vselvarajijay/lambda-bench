from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from lambda_duplex import cli
from lambda_duplex.llm import FakeProvider, LLMGateway
from lambda_duplex.suites.loader import load_suite, valid_scenarios
from lambda_duplex.suites.runtime import mock_for, run_scenario
from lambda_duplex.trajectory import store
from lambda_duplex.trajectory.replay import verify_run
from lambda_duplex.trajectory.seeds import random_stream
from lambda_duplex.trajectory.store import RunSpec, TrajectoryError, find_run_dirs, load_run

ROOT = Path(__file__).resolve().parents[2]
LLM_SUITE = ROOT / "tests/fixtures/suites/llm_stub"


def demo_run(out: Path) -> Path:
    arguments = ["run", "--suite", str(ROOT / "suites/demo"), "--model", "mock", "--out", str(out)]
    assert cli.main(arguments) == 0
    return find_run_dirs(out)[0]


def llm_run(out: Path, seed: int = 4) -> Path:
    loaded = load_suite(str(LLM_SUITE))
    scenario = valid_scenarios(loaded)[0]
    adapter = mock_for(loaded.suite, scenario)
    gateway = LLMGateway("record", FakeProvider(lambda request, call: "recorded reply"))
    run = asyncio.run(
        run_scenario(loaded.suite, scenario, adapter, seed=seed, max_ticks=10, llm=gateway)
    )
    spec = RunSpec("llm_stub", "mock", "text", seed, 10, 160, "clean")
    return store.write_run(
        out / "run",
        loaded.suite,
        spec,
        adapter.describe(),
        run.trajectory,
        run.result,
        run.decision,
        gateway.entries,
    )


def verify(path: Path) -> tuple[bool, bool, tuple[str, ...]]:
    report = asyncio.run(verify_run(path, load_suite(str(ROOT / "suites/demo")).suite))
    return report.rescore_ok, report.replay_ok, report.problems


def rewrite_first_event(path: Path, **changes: object) -> None:
    events = path / "events.jsonl"
    lines = events.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0]) | changes
    lines[0] = json.dumps(first)
    events.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_named_streams_are_stable_and_independent() -> None:
    first = random_stream(7, "noise")
    expected = [first.random() for _ in range(3)]
    second = random_stream(7, "noise")
    random_stream(7, "voices").random()

    assert [second.random() for _ in range(3)] == expected
    assert random_stream(8, "noise").random() != expected[0]


def test_run_directory_contains_expected_files(tmp_path: Path) -> None:
    path = demo_run(tmp_path / "runs")

    assert sorted(item.name for item in path.iterdir()) == [
        "events.jsonl",
        "llm_cache.jsonl",
        "manifest.json",
        "result.json",
        "scenario.json",
    ]
    stored = load_run(path)
    assert stored.decision.passed is True
    assert stored.manifest["suite"]["reference"] == "demo"
    assert stored.manifest["model"]["reference"] == "mock"


def test_clean_run_rescoring_and_replay_match(tmp_path: Path) -> None:
    assert verify(demo_run(tmp_path / "runs")) == (True, True, ())


def test_replay_cli_verifies_run_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "runs"
    demo_run(out)
    capsys.readouterr()

    assert cli.main(["replay", str(out), "--verify", "--suite", str(ROOT / "suites/demo")]) == 0
    assert "rescore=ok replay=ok" in capsys.readouterr().out


def test_changed_result_fails_rescoring(tmp_path: Path) -> None:
    path = demo_run(tmp_path / "runs")
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    result["decision"]["passed"] = False
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")

    assert verify(path)[0] is False


def test_changed_event_fails_replay(tmp_path: Path) -> None:
    path = demo_run(tmp_path / "runs")
    rewrite_first_event(path, type="changed")

    rescore_ok, replay_ok, problems = verify(path)

    assert rescore_ok is True
    assert replay_ok is False
    assert "event 0 differs" in problems[0]


def test_changed_scenario_is_rejected(tmp_path: Path) -> None:
    path = demo_run(tmp_path / "runs")
    scenario = json.loads((path / "scenario.json").read_text(encoding="utf-8"))
    scenario["hidden"]["code_word"] = "changed"
    (path / "scenario.json").write_text(json.dumps(scenario), encoding="utf-8")

    with pytest.raises(TrajectoryError, match="hash"):
        load_run(path)


def test_existing_destination_is_not_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "runs"
    path = demo_run(out)
    before = (path / "manifest.json").read_bytes()

    arguments = ["run", "--suite", str(ROOT / "suites/demo"), "--model", "mock", "--out", str(out)]

    assert cli.main(arguments) == 2
    assert "already exists" in capsys.readouterr().err
    assert (path / "manifest.json").read_bytes() == before


def test_failed_write_leaves_no_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(path: Path, rows: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_jsonl", fail)

    with pytest.raises(OSError, match="disk full"):
        llm_run(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_llm_and_seeded_run_replays_without_provider(tmp_path: Path) -> None:
    path = llm_run(tmp_path)

    report = asyncio.run(verify_run(path, load_suite(str(LLM_SUITE)).suite))

    assert report.ok, report.problems
    assert load_run(path).llm_entries[0]["response"]["text"] == "recorded reply"


def test_removed_llm_entry_fails_replay(tmp_path: Path) -> None:
    path = llm_run(tmp_path)
    (path / "llm_cache.jsonl").write_text("", encoding="utf-8")

    report = asyncio.run(verify_run(path, load_suite(str(LLM_SUITE)).suite))

    assert report.rescore_ok is True
    assert report.replay_ok is False
