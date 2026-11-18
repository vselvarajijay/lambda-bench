"""Summaries built from stored runs."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from lambda_duplex.judges.runner import JudgeError, load_judgments
from lambda_duplex.metrics.reliability import clustered_bootstrap, pass_hat_k
from lambda_duplex.suites.base import Suite
from lambda_duplex.trajectory.store import StoredRun

INVALID_STATUSES = frozenset({"counterpart_error"})
CHECK_KINDS = ("deterministic", "classified", "judge")
SuiteLookup = Callable[[str], Suite | None]


def _interval(clusters: dict[str, list[float]]) -> dict[str, Any]:
    return asdict(clustered_bootstrap(clusters))


def _check_summaries(runs: list[StoredRun]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    missing: dict[tuple[str, str], int] = defaultdict(int)
    for run in runs:
        for check in run.result.checks:
            key = (check.id, check.kind)
            if check.value is None:
                missing[key] += 1
                grouped[key]
            else:
                grouped[key][run.scenario.id].append(float(check.value))
    summaries = []
    for (check_id, kind), clusters in sorted(grouped.items()):
        summaries.append(
            {
                "id": check_id,
                "kind": kind,
                "interval": _interval(dict(clusters)) if clusters else None,
                "missing": missing[(check_id, kind)],
            }
        )
    return summaries


def _validity(runs: Sequence[StoredRun], valid: Sequence[StoredRun]) -> dict[str, Any]:
    return {
        "statuses": dict(sorted(Counter(run.result.status for run in runs).items())),
        "ended_by": dict(sorted(Counter(run.result.ended_by for run in runs).items())),
        "checks_not_computed": sum(
            check.value is None for run in valid for check in run.result.checks
        ),
    }


def _composite(status: str, reason: str, weights: Mapping[str, float]) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "interval": None,
        "missing": 0,
        "weights": dict(sorted(weights.items())),
        "parts": [],
    }


def _composite_parts(
    weights: Mapping[str, float], checks: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    parts: list[dict[str, Any]] = []
    absent: list[str] = []
    for check_id, weight in sorted(weights.items()):
        matches = [check for check in checks if check["id"] == check_id]
        if not any(check["interval"] is not None for check in matches):
            absent.append(check_id)
        parts.extend(
            {
                "id": check_id,
                "weight": weight,
                "kind": check["kind"],
                "interval": check["interval"],
                "missing": check["missing"],
            }
            for check in matches
        )
    return parts, absent


def _composite_values(
    suite: Suite, runs: Sequence[StoredRun]
) -> tuple[dict[str, list[float]], int]:
    clusters: dict[str, list[float]] = defaultdict(list)
    missing = 0
    for run in runs:
        value = suite.composite(run.result)
        if value is None or not math.isfinite(value):
            missing += 1
        else:
            clusters[run.scenario.id].append(float(value))
    return dict(clusters), missing


def _composite_report(
    runs: Sequence[StoredRun],
    valid: Sequence[StoredRun],
    checks: Sequence[Mapping[str, Any]],
    suites: SuiteLookup | None,
) -> dict[str, Any]:
    """Report a composite only when every one of its parts can sit beside it."""
    stored = {
        (run.manifest["suite"]["reference"], run.manifest["suite"]["version"]) for run in runs
    }
    if len(stored) != 1:
        return _composite("unavailable", "runs come from different suite versions", {})
    reference, version = next(iter(stored))
    suite = None if suites is None else suites(reference)
    if suite is None:
        return _composite("unavailable", "the suite could not be loaded", {})
    if suite.manifest.version != version:
        return _composite("unavailable", "the loaded suite version differs from the runs", {})
    weights = dict(suite.manifest.composite_weights)
    try:
        clusters, missing = _composite_values(suite, valid)
    except Exception as error:
        reason = f"the suite composite failed: {type(error).__name__}"
        return _composite("withheld", reason, weights)
    if not weights and not clusters:
        return _composite("none", "the suite defines no composite", {})
    if not weights:
        return _composite("withheld", "the suite does not declare the composite's parts", {})
    if not all(math.isfinite(weight) for weight in weights.values()):
        return _composite("withheld", "a composite weight is not finite", weights)
    parts, absent = _composite_parts(weights, checks)
    if absent:
        report = _composite("withheld", f"parts have no values: {', '.join(absent)}", weights)
        report["parts"] = parts
        return report
    if not clusters:
        report = _composite("withheld", "the suite returned no composite values", weights)
        report["parts"] = parts
        return report
    report = _composite("shown", "", weights)
    report.update(interval=_interval(clusters), missing=missing, parts=parts)
    return report


def _scored_items(row: Mapping[str, Any]) -> list[tuple[str, int, int, int | None]] | None:
    """Return checked item scores, or None if a scored record is malformed."""
    items = row["items"]
    if not items:
        return None
    parsed = []
    try:
        for item in items:
            low, high = item["scale"]["min"], item["scale"]["max"]
            score = item["score"]
            numbers = [low, high] if score is None else [low, high, score]
            if not isinstance(item["id"], str) or any(type(n) is not int for n in numbers):
                return None
            if low >= high or (score is not None and not low <= score <= high):
                return None
            parsed.append((item["id"], low, high, score))
    except (KeyError, TypeError):
        return None
    return parsed


def _judge_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, int]:
    rubric, judge = row["rubric"], row["judge"]
    return (
        rubric["id"],
        rubric["version"],
        rubric["sha256"],
        judge["provider"],
        judge["model"],
        row["blinding"]["seed"],
    )


def _judge_summary(
    key: tuple[str, str, str, str, str, int],
    rows: Sequence[tuple[StoredRun, Mapping[str, Any]]],
    considered: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    scores: dict[tuple[str, int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    not_applicable: Counter[tuple[str, int, int]] = Counter()
    seen: set[str] = set()
    for run, row in rows:
        items = _scored_items(row) if row["status"] == "scored" else []
        if str(run.path) in seen or items is None:
            counts["malformed"] += 1
            continue
        seen.add(str(run.path))
        counts[row["status"]] += 1
        for item_id, low, high, score in items:
            item = (item_id, low, high)
            scores[item]  # An item scored only as not applicable is still listed.
            if score is None:
                not_applicable[item] += 1
            else:
                scores[item][run.scenario.id].append(float(score))
    rubric_id, version, sha256, provider, model, seed = key
    return {
        "rubric": {"id": rubric_id, "version": version, "sha256": sha256},
        "judge": {"provider": provider, "model": model},
        "seed": seed,
        "deterministic": False,
        "calibration": "uncalibrated",
        "scored": counts["scored"],
        "failed": counts["failed"],
        "unscorable": counts["unscorable"],
        "malformed": counts["malformed"],
        "not_judged": max(0, considered - len(seen) - counts["malformed"]),
        "items": [
            {
                "id": item_id,
                "scale": {"min": low, "max": high},
                "interval": _interval(dict(clusters)) if clusters else None,
                "not_applicable": not_applicable[(item_id, low, high)],
            }
            for (item_id, low, high), clusters in sorted(scores.items())
        ],
    }


def _judge_report(valid: Sequence[StoredRun]) -> dict[str, Any]:
    """Summarize stored judgments. They never change a pass decision."""
    grouped: dict[tuple[str, str, str, str, str, int], list[Any]] = defaultdict(list)
    unreadable = []
    for run in valid:
        try:
            rows = load_judgments(run.path)
        except JudgeError:
            unreadable.append(run.path.name)
            continue
        for row in rows:
            grouped[_judge_key(row)].append((run, row))
    considered = len(valid) - len(unreadable)
    return {
        "unreadable_runs": sorted(unreadable),
        "rubrics": [_judge_summary(key, grouped[key], considered) for key in sorted(grouped)],
    }


def _group_report(
    key: tuple[str, str, str], runs: list[StoredRun], suites: SuiteLookup | None
) -> dict[str, Any]:
    suite, model, condition = key
    valid = [run for run in runs if run.result.status not in INVALID_STATUSES]
    cells: dict[str, list[bool]] = defaultdict(list)
    for run in valid:
        cells[run.scenario.id].append(run.decision.passed)
    checks = _check_summaries(valid)
    report: dict[str, Any] = {
        "suite": suite,
        "model": model,
        "condition": condition,
        "runs": len(runs),
        "valid_runs": len(valid),
        "invalid_runs": len(runs) - len(valid),
        "scenarios": {
            scenario: {"trials": len(results), "passed": sum(results)}
            for scenario, results in sorted(cells.items())
        },
        "pass_rate": None,
        "pass_k": {},
        "checks": checks,
        "validity": _validity(runs, valid),
        "composite": _composite_report(runs, valid, checks, suites),
        "judges": _judge_report(valid),
    }
    if not cells:
        return report
    report["pass_rate"] = _interval(
        {name: [float(value) for value in results] for name, results in cells.items()}
    )
    for k in range(1, min(len(results) for results in cells.values()) + 1):
        by_cell = {
            name: [pass_hat_k(len(results), sum(results), k)] for name, results in cells.items()
        }
        report["pass_k"][str(k)] = _interval(by_cell)
    return report


def build_report(runs: Iterable[StoredRun], suites: SuiteLookup | None = None) -> dict[str, Any]:
    """Group stored runs by suite, model, and condition.

    ``suites`` finds a suite by its stored reference. Without it no composite is computed.
    """
    groups: dict[tuple[str, str, str], list[StoredRun]] = defaultdict(list)
    for run in runs:
        manifest = run.manifest
        key = (
            manifest["suite"]["name"],
            manifest["model"]["reference"],
            manifest["run"]["condition"],
        )
        groups[key].append(run)
    return {"groups": [_group_report(key, groups[key], suites) for key in sorted(groups)]}


def format_interval(interval: dict[str, Any] | None) -> str:
    if interval is None:
        return "n/a"
    if interval["clusters"] < 2:
        return f"{interval['estimate']:.3f} (1 scenario, no interval)"
    return (
        f"{interval['estimate']:.3f} "
        f"[{interval['low']:.3f}, {interval['high']:.3f}] "
        f"({interval['clusters']} scenarios)"
    )


def _format_judges(judges: Mapping[str, Any]) -> list[str]:
    lines = []
    if judges["rubrics"] or judges["unreadable_runs"]:
        lines.append("judge scores (non-deterministic, uncalibrated, not used for pass):")
    for rubric in judges["rubrics"]:
        lines.append(
            f"  {rubric['rubric']['id']}@{rubric['rubric']['version']} "
            f"by {rubric['judge']['provider']}:{rubric['judge']['model']} "
            f"seed {rubric['seed']}: {rubric['scored']} scored, {rubric['failed']} failed, "
            f"{rubric['unscorable']} unscorable, {rubric['malformed']} malformed, "
            f"{rubric['not_judged']} not judged"
        )
        for item in rubric["items"]:
            scale = f"{item['scale']['min']}-{item['scale']['max']}"
            lines.append(f"    {item['id']} ({scale}): {format_interval(item['interval'])}")
    if judges["unreadable_runs"]:
        lines.append(f"  unreadable judge files: {len(judges['unreadable_runs'])} runs")
    return lines


def format_report(report: dict[str, Any]) -> str:
    """Render a report for the terminal."""
    lines: list[str] = []
    for group in report["groups"]:
        lines.append(
            f"suite={group['suite']} model={group['model']} condition={group['condition']}"
        )
        lines.append(
            f"runs: {group['runs']} total, {group['valid_runs']} valid, "
            f"{group['invalid_runs']} invalid"
        )
        lines.append(f"pass rate: {format_interval(group['pass_rate'])}")
        for k, interval in group["pass_k"].items():
            lines.append(f"pass^{k}: {format_interval(interval)}")
        lines.append("checks:")
        for check in group["checks"]:
            suffix = f", {check['missing']} not computed" if check["missing"] else ""
            lines.append(
                f"  {check['id']} ({check['kind']}): {format_interval(check['interval'])}{suffix}"
            )
        validity = group["validity"]
        statuses = ", ".join(f"{name}={count}" for name, count in validity["statuses"].items())
        lines.append(f"run statuses: {statuses}")
        composite = group["composite"]
        if composite["status"] == "shown":
            lines.append(f"composite: {format_interval(composite['interval'])}")
            for part in composite["parts"]:
                lines.append(
                    f"  part {part['id']} ({part['kind']}, weight {part['weight']:g}): "
                    f"{format_interval(part['interval'])}"
                )
        else:
            lines.append(f"composite: {composite['status']} ({composite['reason']})")
        lines.extend(_format_judges(group["judges"]))
        lines.append("scenarios:")
        for name, scenario in group["scenarios"].items():
            lines.append(f"  {name}: {scenario['passed']}/{scenario['trials']} passed")
        lines.append("")
    return "\n".join(lines).rstrip()
