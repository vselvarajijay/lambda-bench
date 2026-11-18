#!/usr/bin/env python3
"""Require recorded independent review for score-affecting pull requests."""

from __future__ import annotations

import fnmatch
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

LABEL = "independent-review-passed"
PROTECTED = (
    "src/lambda_duplex/metrics/*",
    "src/lambda_duplex/counterpart/*",
    "src/lambda_duplex/judges/*",
    "src/lambda_duplex/adapters/base.py",
    "src/lambda_duplex/suites/base.py",
    "src/lambda_duplex/trajectory/store.py",
    "suites/*/policy.py",
    "suites/*/checks.py",
    "suites/*/schema.json",
    "suites/*/dev/*",
    "suites/*/test/*",
    "suites/*/rubrics/*",
)
EXEMPT = ("suites/_template/*", "suites/demo/*")


def protected_files(files: Iterable[str]) -> list[str]:
    return [
        path
        for path in files
        if any(fnmatch.fnmatch(path, pattern) for pattern in PROTECTED)
        and not any(fnmatch.fnmatch(path, pattern) for pattern in EXEMPT)
    ]


def review_problems(files: Iterable[str], labels: Iterable[str], body: str) -> list[str]:
    changed = protected_files(files)
    if not changed:
        return []
    problems = []
    if LABEL not in set(labels):
        problems.append(f"missing label {LABEL}")
    if not re.search(r"^Independent review:\s*\S+", body or "", re.M):
        problems.append("missing 'Independent review:' line in the PR description")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_review.py <gh-pr-view.json>", file=sys.stderr)
        return 2
    pull_request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    files = [item["path"] for item in pull_request["files"]]
    labels = [item["name"] for item in pull_request["labels"]]
    problems = review_problems(files, labels, pull_request.get("body") or "")
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"review requirement satisfied for {len(protected_files(files))} protected files")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
