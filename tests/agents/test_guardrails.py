from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / ".claude/hooks/guard.py"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def guard(
    profile: str,
    tool: str,
    env: dict[str, str] | None = None,
    project: Path = ROOT,
    **tool_input: str,
) -> int:
    result = subprocess.run(
        [sys.executable, str(GUARD), "--profile", profile],
        input=json.dumps({"tool_name": tool, "tool_input": tool_input}),
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(project), **(env or {})},
    )
    return result.returncode


@pytest.mark.parametrize(
    "command",
    [
        "git status --short",
        "cd /tmp && git status --short --branch",
        "git -C . log --oneline -1",
        "grep -rn 'a|b' src | head -n 3",
        "find src -name '*.py' 2>/dev/null",
        "cat PLAN.md 2>&1 | wc -l",
    ],
)
def test_read_only_bash_is_allowed(command: str) -> None:
    assert guard("read-only", "Bash", command=command) == 0


@pytest.mark.parametrize(
    "command",
    [
        "touch x",
        "echo hi > x",
        "echo hi >> x",
        "rm -rf src",
        "sed -i s/a/b/ PLAN.md",
        "find . -delete",
        "find . -exec rm {} ;",
        "git commit -m x",
        "git -c alias.x=!touch x",
        "git -C . push",
        "git grep -Otouch x",
        "git push",
        "gh pr merge 1",
        "cat $(echo x)",
        "cat `echo x`",
        "python3 -c 'print(1)'",
        "ls & rm x",
        "sort -o x PLAN.md",
    ],
)
def test_mutating_bash_is_blocked(command: str) -> None:
    assert guard("read-only", "Bash", command=command) == 2


@pytest.mark.parametrize(
    ("profile", "path", "code"),
    [
        ("read-only", "README.md", 2),
        ("reviewer", "suites/x/dev/a.yaml", 2),
        ("architect", "PLAN.md", 0),
        ("architect", "docs/design/a.md", 0),
        ("architect", "src/lambda_duplex/cli.py", 2),
        ("metrics-verifier", "tests/golden/a.json", 0),
        ("metrics-verifier", "src/lambda_duplex/metrics/x.py", 2),
        ("metrics-verifier", "tests/../src/x.py", 2),
        ("suite-author", "suites/interviewer/dev/a.yaml", 0),
        ("suite-author", "suites/interviewer/policy.py", 2),
        ("docs-writer", "docs/metrics.md", 0),
        ("docs-writer", "src/x.py", 2),
        ("builder", "src/x.py", 0),
        ("architect", "/tmp/outside.md", 2),
    ],
)
def test_write_scopes(profile: str, path: str, code: int) -> None:
    assert guard(profile, "Write", file_path=path, content="x") == code


@pytest.mark.parametrize("profile", ["read-only", "builder", "suite-author", "sim-auditor"])
def test_heldout_is_blocked(profile: str) -> None:
    assert guard(profile, "Read", file_path="suites/interviewer/test/a.yaml") == 2
    assert guard(profile, "Bash", command="cat suites/interviewer/test/a.yaml") == 2
    assert guard(profile, "Grep", pattern="x", path="suites/interviewer/test") == 2


def test_heldout_exceptions() -> None:
    path = "suites/interviewer/test/a.yaml"

    assert guard("reviewer", "Read", file_path=path) == 0
    assert guard("reviewer", "Write", file_path=path, content="x") == 2
    assert guard("builder", "Read", {"LAMBDA_DUPLEX_HELDOUT_ACCESS": "1"}, file_path=path) == 0


def test_private_paths_are_blocked(tmp_path: Path) -> None:
    (tmp_path / ".claude").mkdir()
    private = tmp_path.parent / "private-training"
    (tmp_path / ".claude/private-paths.txt").write_text(f"# note\n{private}\n", encoding="utf-8")

    assert guard("builder", "Read", project=tmp_path, file_path=str(private / "a")) == 2
    assert guard("builder", "Bash", project=tmp_path, command=f"ls {private}") == 2
    assert guard("builder", "Bash", project=tmp_path, command="ls") == 0


@pytest.mark.parametrize(
    ("profile", "command", "code"),
    [
        ("metrics-verifier", "uv run pytest tests/metrics", 0),
        ("metrics-verifier", "uv run python x.py", 2),
        ("sim-auditor", "uv run lambda-duplex report runs/a", 0),
        ("sim-auditor", "uv run lambda-duplex run --suite demo --model mock --out runs/a", 0),
        ("sim-auditor", "uv run lambda-duplex run --suite demo --model mock --out src/a", 2),
        ("sim-auditor", "uv run lambda-duplex new-suite x", 2),
    ],
)
def test_profile_commands(profile: str, command: str, code: int) -> None:
    assert guard(profile, "Bash", command=command) == code


def test_invalid_hook_input_is_blocked() -> None:
    result = subprocess.run(
        [sys.executable, str(GUARD), "--profile", "builder"],
        input="not json",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2


def test_agent_definitions_are_valid() -> None:
    assert load_script("check_agents").check_agents() == []


def test_agent_checker_finds_missing_agents(tmp_path: Path) -> None:
    assert "architect: missing definition" in load_script("check_agents").check_agents(tmp_path)


def test_review_required_for_score_affecting_files() -> None:
    check = load_script("check_review")
    metric = ["src/lambda_duplex/metrics/report.py"]
    body = "Independent review: reviewer found no issues"

    assert check.review_problems(["README.md"], [], "") == []
    assert check.review_problems(["suites/demo/dev/a.yaml"], [], "") == []
    assert len(check.review_problems(metric, [], "")) == 2
    assert len(check.review_problems(metric, [check.LABEL], "")) == 1
    assert check.review_problems(metric, [check.LABEL], body) == []
    assert check.review_problems(["suites/interviewer/dev/a.yaml"], [check.LABEL], body) == []
