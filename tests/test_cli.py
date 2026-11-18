from __future__ import annotations

import json
import subprocess
import sys

import pytest

from lambda_duplex import __version__, cli


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"lambda-duplex {__version__}"


def test_doctor_json_reports_required_and_optional_checks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["doctor", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in report["checks"]}
    assert report["ok"] is True
    assert checks["python"]["required"] is True
    assert checks["silero_vad"]["required"] is False


def test_doctor_fails_when_required_check_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failed = cli.Check("python", ok=False, required=True, detail="too old")
    monkeypatch.setattr(cli, "collect_checks", lambda: [failed])

    assert cli.main(["doctor"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_missing_optional_dependency_does_not_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = cli.Check("openai", ok=False, required=False, detail="not installed")
    monkeypatch.setattr(cli, "collect_checks", lambda: [missing])

    assert cli.main(["doctor"]) == 0


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "lambda_duplex", "--version"],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == f"lambda-duplex {__version__}"


def test_inline_run_is_repeatable(capsys: pytest.CaptureFixture[str]) -> None:
    arguments = ["run", "--suite", "_inline_echo", "--model", "mock", "--mode", "text"]

    assert cli.main([*arguments, "--ticks", "50"]) == 0
    first = capsys.readouterr().out
    assert cli.main([*arguments, "--ticks", "50"]) == 0
    second = capsys.readouterr().out

    assert first == second
    assert "status=completed ticks=50" in first
    assert 'model.spoke {"text": "echo: line 4"}' in first


def test_run_rejects_unavailable_combination(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--suite", "missing", "--model", "mock"]) == 2
    assert "unknown suite" in capsys.readouterr().err


def test_demo_suite_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["suites", "list"]) == 0
    assert "demo" in capsys.readouterr().out.split()
    assert cli.main(["suites", "validate", "demo"]) == 0
    assert "validated 1 scenarios, 0 issues" in capsys.readouterr().out
    assert cli.main(["run", "--suite", "demo", "--model", "mock", "--mode", "text"]) == 0
    assert "passed=true" in capsys.readouterr().out
    assert cli.main(["run", "--suite", "demo", "--model", "mock:bad", "--mode", "text"]) == 0
    assert "passed=false" in capsys.readouterr().out
