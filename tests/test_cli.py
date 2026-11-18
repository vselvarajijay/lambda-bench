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
