from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lambda_duplex import cli
from lambda_duplex.metrics.report import build_report
from lambda_duplex.metrics.report_html import format_html
from lambda_duplex.suites.base import RunResult
from lambda_duplex.suites.loader import load_suite
from lambda_duplex.trajectory.store import find_run_dirs, load_run

ROOT = Path(__file__).resolve().parents[2]
DEMO = str(ROOT / "suites/demo")
GOLDEN = ROOT / "tests/golden/report_demo.html"
CHECK = "code_word_disclosed"


class CompositeSuite:
    """Demo suite with a weighted composite."""

    def __init__(
        self,
        weights: dict[str, float],
        value: Callable[[RunResult], float | None] = lambda result: 0.75,
    ) -> None:
        self.manifest = replace(load_suite(DEMO).suite.manifest, composite_weights=weights)
        self._value = value

    def composite(self, result: RunResult) -> float | None:
        return self._value(result)


def make_runs(out: Path, model: str = "mock:good+good+bad", judged: bool = True) -> list[Path]:
    arguments = ["run", "--suite", DEMO, "--model", model, "--k", "3", "--ticks", "12"]
    assert cli.main([*arguments, "--out", str(out)]) == 0
    if judged:
        judge = ["judge", str(out), "--rubric", "question_quality", "--provider", "fake"]
        assert cli.main([*judge, "--suite", DEMO]) == 0
    return find_run_dirs(out)


def report_for(out: Path, suite: Any = None) -> dict[str, Any]:
    lookup = None if suite is None else lambda reference: suite
    return build_report((load_run(path) for path in find_run_dirs(out)), lookup)


def group_for(out: Path, suite: Any = None) -> dict[str, Any]:
    group: dict[str, Any] = report_for(out, suite)["groups"][0]
    return group


def edit_judgment(run: Path, change: Callable[[dict[str, Any]], Any]) -> None:
    path = run / "judges.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    change(row)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_html_report_matches_snapshot(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs")
    suite = CompositeSuite({CHECK: 1.0})

    page = format_html(report_for(tmp_path / "runs", suite)) + "\n"

    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(page, encoding="utf-8")
    assert page == GOLDEN.read_text(encoding="utf-8")


def test_html_report_is_self_contained_and_has_every_section(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs")

    page = format_html(report_for(tmp_path / "runs"))

    for marker in ('data-kind="deterministic"', 'data-kind="classified"', 'class="judges"'):
        assert marker in page
    assert "pass^3" in page and "Valid runs" in page
    assert "non-deterministic and uncalibrated" in page
    assert "<script" not in page and "http://" not in page and "https://" not in page
    assert str(tmp_path) not in page


def test_run_derived_text_is_escaped(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs")
    hostile = "<script>alert(1)</script>\"'"
    report = report_for(tmp_path / "runs")
    group = report["groups"][0]
    group["model"] = hostile
    group["checks"][0]["id"] = hostile
    group["scenarios"] = {hostile: {"trials": 1, "passed": 1}}
    group["judges"]["unreadable_runs"] = [hostile]
    group["judges"]["rubrics"][0]["rubric"]["id"] = hostile
    group["judges"]["rubrics"][0]["items"][0]["id"] = hostile
    group["composite"]["reason"] = hostile

    page = format_html(report)

    assert "<script" not in page
    assert page.count("&lt;script&gt;") >= 7
    assert "&quot;&#x27;" in page


def test_composite_is_shown_beside_every_part(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs", judged=False)

    group = group_for(tmp_path / "runs", CompositeSuite({CHECK: 0.5}))
    page = format_html({"groups": [group]})

    composite = group["composite"]
    assert composite["status"] == "shown"
    assert composite["interval"]["estimate"] == 0.75
    assert [(part["id"], part["weight"]) for part in composite["parts"]] == [(CHECK, 0.5)]
    section = page.split('class="composite"')[1].split("</section>")[0]
    assert "0.750" in section and CHECK in section and "<td>0.5</td>" in section


@pytest.mark.parametrize(
    ("weights", "reason"),
    [
        ({}, "does not declare"),
        ({"hidden_part": 1.0}, "hidden_part"),
        ({CHECK: math.inf}, "finite"),
    ],
)
def test_composite_without_all_parts_is_withheld(
    tmp_path: Path, weights: dict[str, float], reason: str
) -> None:
    make_runs(tmp_path / "runs", judged=False)

    group = group_for(tmp_path / "runs", CompositeSuite(weights, lambda result: 0.123))
    page = format_html({"groups": [group]})

    assert group["composite"]["status"] == "withheld"
    assert reason in group["composite"]["reason"]
    assert group["composite"]["interval"] is None
    assert "0.123" not in page
    assert 'data-status="withheld"' in page


def test_composite_error_and_empty_values_are_withheld(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs", judged=False)

    def broken(result: RunResult) -> float:
        raise KeyError("bug")

    failed = group_for(tmp_path / "runs", CompositeSuite({CHECK: 1.0}, broken))["composite"]
    empty = group_for(tmp_path / "runs", CompositeSuite({CHECK: 1.0}, lambda r: None))["composite"]

    assert failed["status"] == "withheld" and "KeyError" in failed["reason"]
    assert empty["status"] == "withheld"


def test_non_finite_composite_values_count_as_missing(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs", judged=False)
    values = iter([0.5, math.nan, 1.0])

    composite = group_for(
        tmp_path / "runs", CompositeSuite({CHECK: 1.0}, lambda result: next(values))
    )["composite"]

    assert composite["missing"] == 1
    assert composite["interval"]["estimate"] == 0.75


def test_composite_needs_the_matching_suite(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs", judged=False)
    suite = CompositeSuite({CHECK: 1.0})
    suite.manifest = replace(suite.manifest, version="other")

    assert group_for(tmp_path / "runs")["composite"]["status"] == "unavailable"
    assert group_for(tmp_path / "runs", suite)["composite"]["status"] == "unavailable"


def test_suite_without_composite_says_so(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs", judged=False)

    composite = group_for(tmp_path / "runs", load_suite(DEMO).suite)["composite"]

    assert composite["status"] == "none"


def test_judge_scores_are_summarized_and_labeled(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs")

    rubric = group_for(tmp_path / "runs")["judges"]["rubrics"][0]

    assert rubric["deterministic"] is False
    assert rubric["calibration"] == "uncalibrated"
    assert (rubric["scored"], rubric["failed"], rubric["not_judged"]) == (3, 0, 0)
    assert rubric["items"]
    assert all(item["interval"]["clusters"] == 1 for item in rubric["items"])


def test_judge_scores_never_change_pass_results(tmp_path: Path) -> None:
    runs = make_runs(tmp_path / "runs")
    before = group_for(tmp_path / "runs")
    for run in runs:
        (run / "judges.jsonl").unlink()

    after = group_for(tmp_path / "runs")

    for key in ("pass_rate", "pass_k", "scenarios", "checks", "validity"):
        assert before[key] == after[key]


def test_runs_without_judgments_still_report(tmp_path: Path) -> None:
    make_runs(tmp_path / "runs", judged=False)

    group = group_for(tmp_path / "runs")

    assert group["judges"] == {"unreadable_runs": [], "rubrics": []}
    assert "No judgments in these runs" in format_html({"groups": [group]})


def test_partly_judged_runs_are_counted(tmp_path: Path) -> None:
    runs = make_runs(tmp_path / "runs")
    (runs[0] / "judges.jsonl").unlink()

    rubric = group_for(tmp_path / "runs")["judges"]["rubrics"][0]

    assert (rubric["scored"], rubric["not_judged"]) == (2, 1)


@pytest.mark.parametrize("body", ["not json\n", '{"kind": "judge"}\n', '{"deterministic": true}\n'])
def test_malformed_judge_file_is_reported_not_used(tmp_path: Path, body: str) -> None:
    runs = make_runs(tmp_path / "runs")
    (runs[0] / "judges.jsonl").write_text(body, encoding="utf-8")

    judges = group_for(tmp_path / "runs")["judges"]

    assert judges["unreadable_runs"] == [runs[0].name]
    assert judges["rubrics"][0]["scored"] == 2
    assert judges["rubrics"][0]["not_judged"] == 0
    assert str(tmp_path) not in format_html(report_for(tmp_path / "runs"))


@pytest.mark.parametrize("status", ["failed", "unscorable"])
def test_unscored_judgments_add_no_scores(tmp_path: Path, status: str) -> None:
    runs = make_runs(tmp_path / "runs")
    for run in runs:
        edit_judgment(run, lambda row: row.update(status=status))

    rubric = group_for(tmp_path / "runs")["judges"]["rubrics"][0]

    assert rubric["items"] == []
    assert rubric[status] == 3 and rubric["scored"] == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.update(items=[]),
        lambda row: row["items"][0].update(score=99),
        lambda row: row["items"][0].update(score=True),
        lambda row: row["items"][0].update(score=1.5),
        lambda row: row["items"][0].pop("scale"),
        lambda row: row["items"].append("bad"),
    ],
)
def test_malformed_scored_record_is_excluded(
    tmp_path: Path, change: Callable[[dict[str, Any]], Any]
) -> None:
    runs = make_runs(tmp_path / "runs")
    edit_judgment(runs[0], change)

    rubric = group_for(tmp_path / "runs")["judges"]["rubrics"][0]

    assert (rubric["malformed"], rubric["scored"], rubric["not_judged"]) == (1, 2, 0)


def test_not_applicable_scores_are_counted_apart(tmp_path: Path) -> None:
    runs = make_runs(tmp_path / "runs")
    for run in runs:
        edit_judgment(run, lambda row: row["items"][0].update(score=None))
    item_id = json.loads((runs[0] / "judges.jsonl").read_text(encoding="utf-8"))["items"][0]["id"]

    items = group_for(tmp_path / "runs")["judges"]["rubrics"][0]["items"]

    item = next(entry for entry in items if entry["id"] == item_id)
    assert item["interval"] is None and item["not_applicable"] == 3


def test_invalid_runs_are_counted_and_not_judged(tmp_path: Path) -> None:
    runs = make_runs(tmp_path / "runs")
    path = runs[0] / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result.update(status="counterpart_error", ended_by="error")
    result["decision"] = {"passed": False, "reasons": ["run counterpart_error"]}
    path.write_text(json.dumps(result), encoding="utf-8")

    group = group_for(tmp_path / "runs")

    assert group["invalid_runs"] == 1
    assert group["validity"]["statuses"] == {"completed": 2, "counterpart_error": 1}
    assert group["judges"]["rubrics"][0]["scored"] == 2


def test_cli_writes_html_and_keeps_text_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_runs(tmp_path / "runs")
    runs = str(tmp_path / "runs")
    target = tmp_path / "out/report.html"
    capsys.readouterr()

    assert cli.main(["report", runs, "--format", "html"]) == 0
    assert capsys.readouterr().out.startswith("<!doctype html>")
    assert cli.main(["report", runs, "--format", "html", "--out", str(target)]) == 0
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert cli.main(["report", runs, "--format", "json"]) == 0
    assert "groups" in json.loads(capsys.readouterr().out)
    assert cli.main(["report", runs]) == 0
    text = capsys.readouterr().out
    assert "not used for pass" in text and "composite:" in text


def test_cli_report_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_runs(tmp_path / "runs", judged=False)
    runs = str(tmp_path / "runs")
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")

    assert cli.main(["report", runs, "--suite", "missing-suite"]) == 2
    assert cli.main(["report", runs, "--out", str(blocker / "report.html")]) == 2
    with pytest.raises(SystemExit):
        cli.main(["report", runs, "--json", "--format", "html"])
    assert cli.main(["report", runs, "--suite", DEMO]) == 0
