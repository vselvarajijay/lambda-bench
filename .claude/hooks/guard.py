#!/usr/bin/env python3
"""PreToolUse guard for project subagents. Exit code 2 blocks the tool call."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2])).resolve()
HELDOUT = re.compile(r"(^|[^A-Za-z0-9_.-])(\./)?suites/[^/\s'\"]+/test(/|\b)")
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
PATH_KEYS = ("file_path", "notebook_path", "path")
READ_COMMANDS = {
    "basename",
    "cat",
    "cd",
    "cut",
    "diff",
    "dirname",
    "echo",
    "file",
    "find",
    "gh",
    "git",
    "grep",
    "head",
    "jq",
    "ls",
    "pwd",
    "realpath",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "tr",
    "tree",
    "uniq",
    "wc",
}
GIT_READ = {"blame", "diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}
GH_READ = {("pr", "checks"), ("pr", "diff"), ("pr", "list"), ("pr", "view"), ("repo", "view")}
FIND_WRITES = {"-delete", "-exec", "-execdir", "-fls", "-fprint", "-fprint0", "-fprintf", "-ok"}
OPERATORS = {";", "&&", "||", "|", ";;"}
AUDIT_COMMANDS = {"audit", "replay", "report", "run", "suites", "timeline"}


@dataclass(frozen=True)
class Profile:
    writes: tuple[str, ...] = ()
    commands: tuple[tuple[str, ...], ...] = ()
    free_bash: bool = False
    free_writes: bool = False
    heldout: bool = False
    audit_cli: bool = False


PROFILES = {
    "read-only": Profile(),
    "reviewer": Profile(heldout=True),
    "architect": Profile(writes=("PLAN.md", "docs/design/*")),
    "suite-author": Profile(writes=("suites/*/dev/*", "suites/*/DATASHEET.md")),
    "metrics-verifier": Profile(
        writes=("tests/*",),
        commands=(("uv", "run", "pytest"), ("uv", "run", "ruff"), ("uv", "run", "mypy")),
    ),
    "sim-auditor": Profile(audit_cli=True),
    "docs-writer": Profile(
        writes=(
            "README.md",
            "CONTRIBUTING.md",
            "THIRD_PARTY_NOTICES",
            "docs/*",
            "suites/*/DATASHEET.md",
        )
    ),
    "builder": Profile(free_bash=True, free_writes=True),
}


class Blocked(Exception):
    pass


def project_path(value: str) -> str | None:
    path = Path(value).expanduser()
    path = path if path.is_absolute() else ROOT / path
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def private_paths() -> list[str]:
    source = ROOT / ".claude" / "private-paths.txt"
    if not source.is_file():
        return []
    entries: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            entries.extend({entry, str(Path(entry).expanduser())})
    return entries


def check_restricted_text(text: str, profile: Profile) -> None:
    allowed = profile.heldout or os.environ.get("LAMBDA_DUPLEX_HELDOUT_ACCESS") == "1"
    if not allowed and HELDOUT.search(text):
        raise Blocked("held-out test scenarios are not available to this agent")
    if any(entry in text for entry in private_paths()):
        raise Blocked("this path is private to the maintainer and must not be used")


def split_commands(command: str) -> list[list[str]]:
    if "$(" in command or "`" in command:
        raise Blocked("command substitution is not allowed")
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError as error:
        raise Blocked(f"cannot parse command: {error}") from error
    commands: list[list[str]] = [[]]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token in OPERATORS:
            commands.append([])
        elif token == ">&" and following.isdigit():
            if commands[-1][-1:] == ["2"]:
                commands[-1].pop()
            index += 1
        elif token == ">" and following == "/dev/null":
            if commands[-1][-1:] == ["2"]:
                commands[-1].pop()
            index += 1
        elif set(token) <= set("<>&()|;"):
            if token != "<":
                raise Blocked(f"shell operator {token!r} is not allowed")
        else:
            commands[-1].append(token)
        index += 1
    return [item for item in commands if item]


def read_command_allowed(words: list[str]) -> bool:
    name = words[0]
    if name not in READ_COMMANDS:
        return False
    if name == "find":
        return not any(word in FIND_WRITES for word in words)
    if name == "sed":
        return not any(word.startswith("-i") or word.startswith("--in-place") for word in words)
    if name == "sort":
        return not any(word.startswith("-o") or word.startswith("--output") for word in words)
    if name == "git":
        arguments = words[1:]
        if arguments[:1] == ["-C"]:
            arguments = arguments[2:]
        if not arguments or arguments[0] not in GIT_READ:
            return False
        blocked = ("--output", "--ext-diff", "--open-files-in-pager", "-O")
        return not any(word.startswith(blocked) for word in arguments)
    if name == "gh":
        return tuple(words[1:3]) in GH_READ
    return True


def audit_command_allowed(words: list[str]) -> bool:
    if words[:3] != ["uv", "run", "lambda-duplex"] or len(words) < 4:
        return False
    if words[3] not in AUDIT_COMMANDS:
        return False
    for index, word in enumerate(words):
        value = None
        if word == "--out" and index + 1 < len(words):
            value = words[index + 1]
        elif word.startswith("--out="):
            value = word.split("=", 1)[1]
        if value is not None:
            target = project_path(value)
            if target is None or not target.startswith(("runs/", "reports/")):
                return False
    return True


def check_bash(command: str, profile: Profile) -> None:
    check_restricted_text(command, profile)
    if profile.free_bash:
        return
    for words in split_commands(command):
        allowed = read_command_allowed(words)
        allowed = allowed or any(tuple(words[: len(p)]) == p for p in profile.commands)
        allowed = allowed or (profile.audit_cli and audit_command_allowed(words))
        if not allowed:
            raise Blocked(f"command is not allowed for this agent: {shlex.join(words)}")


def check_tool(tool: str, tool_input: dict[str, object], profile: Profile) -> None:
    if tool == "Bash":
        check_bash(str(tool_input.get("command", "")), profile)
        return
    values = [str(tool_input[key]) for key in (*PATH_KEYS, "pattern", "glob") if key in tool_input]
    for value in values:
        check_restricted_text(value, profile)
    paths = [str(tool_input[key]) for key in PATH_KEYS if key in tool_input]
    for value in paths:
        target = project_path(value)
        if target is not None:
            check_restricted_text(target, profile)
    if tool not in WRITE_TOOLS or profile.free_writes:
        return
    for value in paths:
        target = project_path(value)
        if target is None or not any(fnmatch.fnmatch(target, item) for item in profile.writes):
            raise Blocked(f"this agent may not write {value}")
    if not paths:
        raise Blocked("write target is missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        check_tool(
            str(payload["tool_name"]), dict(payload.get("tool_input") or {}), PROFILES[args.profile]
        )
    except Blocked as error:
        print(f"Blocked by project guard: {error}", file=sys.stderr)
        return 2
    except (KeyError, TypeError, ValueError) as error:
        print(f"Blocked by project guard: invalid hook input: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
