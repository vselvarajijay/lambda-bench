from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lambda_duplex import cli
from lambda_duplex.batch import (
    TIERS,
    BatchError,
    BatchItem,
    BatchPlan,
    estimate_cost,
    plan_batch,
    run_batch,
    run_mock_item,
)
from lambda_duplex.metrics.report import build_report
from lambda_duplex.suites.base import Scenario
from lambda_duplex.suites.loader import LoadedSuite, load_suite
from lambda_duplex.suites.runtime import ScenarioRun
from lambda_duplex.trajectory.store import find_run_dirs, load_run

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "suites/demo"


def plan_for(tier: str = "dev") -> tuple[LoadedSuite, BatchPlan]:
    loaded = load_suite(str(DEMO))
    return loaded, plan_batch(loaded, str(DEMO), tier, "mock", max_ticks=120)


def snapshot(folder: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in find_run_dirs(folder):
        result[path.name] = {
            "result": (path / "result.json").read_text(encoding="utf-8"),
            "events": (path / "events.jsonl").read_text(encoding="utf-8"),
            "audio": load_run(path).manifest["audio"]["tracks"],
        }
    return result


def test_tier_sizes_match_plan() -> None:
    assert TIERS["full"].scenarios * TIERS["full"].personas * TIERS["full"].trials * 2 == 960


def test_plan_reports_available_matrix_and_warnings() -> None:
    _, plan = plan_for()

    assert plan.requested_runs == 160
    assert len(plan.items) == 4
    assert plan.items[0].id == "code_word_001__neutral__clean__trial0"
    assert any("scenarios" in warning for warning in plan.warnings)
    assert any("personas" in warning for warning in plan.warnings)


def test_interviewer_plan_uses_scenario_budget(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "batch",
            "plan",
            "--suite",
            str(ROOT / "suites/interviewer"),
            "--tier",
            "dev",
            "--model",
            "mock",
            "--cost-per-minute",
            "0.5",
        ]
    )

    plan = json.loads(capsys.readouterr().out)
    assert code == 0
    assert plan["call_minutes"] == 8
    assert plan["estimated_cost"] == pytest.approx(len(plan["items"]) * 4)


def test_unknown_tier_is_rejected() -> None:
    with pytest.raises(BatchError, match="unknown tier"):
        plan_batch(load_suite(str(DEMO)), "demo", "missing", "mock")


def test_batch_run_writes_reportable_results(tmp_path: Path) -> None:
    loaded, plan = plan_for()

    counts = asyncio.run(run_batch(loaded, plan, tmp_path, jobs=2))

    assert counts == {"planned": 4, "skipped": 0, "completed": 4, "invalid": 0}
    report = build_report(load_run(path) for path in find_run_dirs(tmp_path))
    assert {group["condition"] for group in report["groups"]} == {"clean", "realistic"}


def test_interrupted_batch_resumes_to_same_results(tmp_path: Path) -> None:
    loaded, plan = plan_for()
    complete = tmp_path / "complete"
    interrupted = tmp_path / "interrupted"
    asyncio.run(run_batch(loaded, plan, complete))

    class Stop(Exception):
        pass

    def stop(item: BatchItem) -> None:
        if len(find_run_dirs(interrupted)) == 2:
            raise Stop

    with pytest.raises(Stop):
        asyncio.run(run_batch(loaded, plan, interrupted, on_complete=stop))
    assert len(find_run_dirs(interrupted)) == 2

    counts = asyncio.run(run_batch(loaded, plan, interrupted))

    assert counts["skipped"] == 2
    assert counts["completed"] == 2
    assert snapshot(interrupted) == snapshot(complete)


def test_partial_temporary_folder_is_removed(tmp_path: Path) -> None:
    loaded, plan = plan_for("smoke")
    stale = tmp_path / ".code_word_001.tmp"
    stale.mkdir()

    asyncio.run(run_batch(loaded, plan, tmp_path))

    assert not stale.exists()


def test_batch_rejects_changed_plan(tmp_path: Path) -> None:
    loaded, plan = plan_for("smoke")
    asyncio.run(run_batch(loaded, plan, tmp_path))

    with pytest.raises(BatchError, match="different batch plan"):
        asyncio.run(run_batch(loaded, replace(plan, seed=9), tmp_path))


def test_budget_cap_stops_before_running(tmp_path: Path) -> None:
    loaded, plan = plan_for()

    with pytest.raises(BatchError, match="exceeds budget"):
        asyncio.run(run_batch(loaded, plan, tmp_path, cost_per_minute=1, budget=1))

    assert find_run_dirs(tmp_path) == []
    assert estimate_cost(plan, 1) == pytest.approx(32)


def test_invalid_run_is_retried_then_recorded(tmp_path: Path) -> None:
    loaded, plan = plan_for("smoke")
    attempts = 0

    async def flaky(
        suite: LoadedSuite, scenario: Scenario, batch: BatchPlan, item: BatchItem
    ) -> ScenarioRun:
        nonlocal attempts
        attempts += 1
        run = await run_mock_item(suite, scenario, batch, item)
        if attempts == 1:
            outcome = replace(run.outcome, status="counterpart_error")
            return replace(run, outcome=outcome)
        return run

    counts = asyncio.run(run_batch(loaded, plan, tmp_path, retries=1, runner=flaky))

    assert attempts == 2
    assert counts["completed"] == 1


def test_persistent_invalid_run_is_not_stored(tmp_path: Path) -> None:
    loaded, plan = plan_for("smoke")

    async def invalid(
        suite: LoadedSuite, scenario: Scenario, batch: BatchPlan, item: BatchItem
    ) -> ScenarioRun:
        run = await run_mock_item(suite, scenario, batch, item)
        return replace(run, outcome=replace(run.outcome, status="counterpart_error"))

    counts = asyncio.run(run_batch(loaded, plan, tmp_path, retries=0, runner=invalid))

    assert counts["invalid"] == 1
    assert find_run_dirs(tmp_path) == []


def test_cli_batch_run_and_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arguments = ["batch", "run", "--suite", str(DEMO), "--tier", "smoke", "--model", "mock"]

    assert cli.main([*arguments, "--out", str(tmp_path), "--ticks", "120"]) == 0
    assert json.loads(capsys.readouterr().out)["completed"] == 1
    assert cli.main(["replay", str(tmp_path), "--verify", "--suite", str(DEMO)]) == 0
