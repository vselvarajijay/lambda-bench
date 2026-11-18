from __future__ import annotations

import json
from pathlib import Path

import pytest

from lambda_duplex import cli
from lambda_duplex.metrics.reliability import clustered_bootstrap, pass_hat_k
from lambda_duplex.metrics.report import build_report
from lambda_duplex.trajectory.store import find_run_dirs, load_run

ROOT = Path(__file__).resolve().parents[2]
DEMO = str(ROOT / "suites/demo")


@pytest.mark.parametrize(
    ("trials", "successes", "k", "expected"),
    [
        (4, 4, 4, 1.0),
        (4, 3, 1, 0.75),
        (4, 3, 2, 0.5),
        (4, 3, 3, 0.25),
        (4, 3, 4, 0.0),
        (4, 0, 1, 0.0),
        (5, 3, 2, 0.3),
    ],
)
def test_pass_hat_k_matches_hand_values(
    trials: int, successes: int, k: int, expected: float
) -> None:
    assert pass_hat_k(trials, successes, k) == pytest.approx(expected)


@pytest.mark.parametrize(("trials", "successes", "k"), [(3, 1, 4), (3, 4, 1), (3, 1, 0)])
def test_pass_hat_k_rejects_invalid_input(trials: int, successes: int, k: int) -> None:
    with pytest.raises(ValueError):
        pass_hat_k(trials, successes, k)


def test_bootstrap_is_repeatable_and_resamples_clusters() -> None:
    clusters = {"a": [0.0, 0.0, 0.0], "b": [1.0]}

    first = clustered_bootstrap(clusters, seed=3)

    assert first == clustered_bootstrap(clusters, seed=3)
    assert first.estimate == pytest.approx(0.25)
    assert first.low == 0.0
    assert first.high == 1.0
    assert first.clusters == 2


def test_bootstrap_single_value_has_no_width() -> None:
    interval = clustered_bootstrap({"a": [0.5]})

    assert (interval.low, interval.estimate, interval.high) == (0.5, 0.5, 0.5)


def test_bootstrap_rejects_empty_clusters() -> None:
    with pytest.raises(ValueError):
        clustered_bootstrap({"a": []})


def run_demo(out: Path, model: str, k: int = 4) -> None:
    arguments = ["run", "--suite", DEMO, "--model", model, "--k", str(k), "--ticks", "12"]
    assert cli.main([*arguments, "--out", str(out)]) == 0


def test_cli_reports_three_of_four(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "runs"
    run_demo(out, "mock:good+good+good+bad")
    capsys.readouterr()

    assert cli.main(["report", str(out)]) == 0
    text = capsys.readouterr().out
    assert "pass^3: 0.250" in text
    assert "pass^4: 0.000" in text
    assert "code_word_001: 3/4 passed" in text

    assert cli.main(["report", str(out), "--json"]) == 0
    group = json.loads(capsys.readouterr().out)["groups"][0]
    assert group["pass_k"]["4"]["estimate"] == 0.0
    assert group["pass_rate"]["estimate"] == 0.75
    assert group["checks"][0]["kind"] == "deterministic"


def test_trials_use_distinct_seeds_and_all_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "runs"
    run_demo(out, "mock", 3)

    seeds = {load_run(path).manifest["run"]["seed"] for path in find_run_dirs(out)}

    assert len(seeds) == 3
    assert cli.main(["replay", str(out), "--verify", "--suite", DEMO]) == 0


def test_counterpart_failures_are_invalid_not_model_failures(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    run_demo(out, "mock", 2)
    path = find_run_dirs(out)[0]
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    result["status"] = "counterpart_error"
    result["ended_by"] = "error"
    result["decision"] = {"passed": False, "reasons": ["run counterpart_error"]}
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")

    group = build_report(load_run(item) for item in find_run_dirs(out))["groups"][0]

    assert group["invalid_runs"] == 1
    assert group["valid_runs"] == 1
    assert group["pass_rate"]["estimate"] == 1.0
    assert list(group["pass_k"]) == ["1"]


def test_report_rejects_missing_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report", str(tmp_path)]) == 2
    assert "no runs found" in capsys.readouterr().err


def test_invalid_k_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--suite", DEMO, "--model", "mock", "--k", "0"]) == 2
    assert "--k" in capsys.readouterr().err
