#!/usr/bin/env python3
"""Validate project subagent definitions."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "adapter-engineer": "builder",
    "architect": "architect",
    "audio-engineer": "builder",
    "core-engineer": "builder",
    "docs-writer": "docs-writer",
    "license-auditor": "read-only",
    "metrics-verifier": "metrics-verifier",
    "reuse-scout": "read-only",
    "scenario-reviewer": "reviewer",
    "sim-auditor": "sim-auditor",
    "suite-author": "suite-author",
}
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
NO_WRITE_TOOLS = {"license-auditor", "reuse-scout", "scenario-reviewer", "sim-auditor"}


def parse_agent(path: Path) -> tuple[dict[str, Any], str]:
    match = re.match(r"---\n(.*?)\n---\n(.*)", path.read_text(encoding="utf-8"), re.S)
    if match is None:
        raise ValueError("missing frontmatter")
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a mapping")
    return data, match.group(2)


def check_agent(name: str, profile: str, agents_dir: Path) -> list[str]:
    path = agents_dir / f"{name}.md"
    if not path.is_file():
        return [f"{name}: missing definition"]
    try:
        data, body = parse_agent(path)
    except (ValueError, yaml.YAMLError) as error:
        return [f"{name}: {error}"]
    problems = []
    if data.get("name") != name:
        problems.append(f"{name}: name does not match file")
    for field in ("description", "tools", "model"):
        if not data.get(field):
            problems.append(f"{name}: missing {field}")
    if "Guardrails:" not in body:
        problems.append(f"{name}: missing guardrails")
    tools = {tool.strip() for tool in str(data.get("tools", "")).split(",")}
    if name in NO_WRITE_TOOLS and tools & WRITE_TOOLS:
        problems.append(f"{name}: read-only agent has write tools")
    try:
        command = data["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    except (KeyError, IndexError, TypeError):
        return [*problems, f"{name}: missing guard hook"]
    if not str(command).endswith(f'guard.py" --profile {profile}'):
        problems.append(f"{name}: expected guard profile {profile}")
    return problems


def check_agents(agents_dir: Path = ROOT / ".claude" / "agents") -> list[str]:
    return [
        problem
        for name, profile in sorted(EXPECTED.items())
        for problem in check_agent(name, profile, agents_dir)
    ]


def main() -> int:
    problems = check_agents()
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"validated {len(EXPECTED)} agents")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
