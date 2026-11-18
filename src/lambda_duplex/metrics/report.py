"""Summaries built from stored runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from lambda_duplex.metrics.reliability import clustered_bootstrap, pass_hat_k
from lambda_duplex.trajectory.store import StoredRun

INVALID_STATUSES = frozenset({"counterpart_error"})


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


def _group_report(key: tuple[str, str, str], runs: list[StoredRun]) -> dict[str, Any]:
    suite, model, condition = key
    valid = [run for run in runs if run.result.status not in INVALID_STATUSES]
    cells: dict[str, list[bool]] = defaultdict(list)
    for run in valid:
        cells[run.scenario.id].append(run.decision.passed)
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
        "checks": _check_summaries(valid),
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


def build_report(runs: Iterable[StoredRun]) -> dict[str, Any]:
    """Group stored runs by suite, model, and condition."""
    groups: dict[tuple[str, str, str], list[StoredRun]] = defaultdict(list)
    for run in runs:
        manifest = run.manifest
        key = (
            manifest["suite"]["name"],
            manifest["model"]["reference"],
            manifest["run"]["condition"],
        )
        groups[key].append(run)
    return {"groups": [_group_report(key, groups[key]) for key in sorted(groups)]}


def _format_interval(interval: dict[str, Any] | None) -> str:
    if interval is None:
        return "n/a"
    if interval["clusters"] < 2:
        return f"{interval['estimate']:.3f} (1 scenario, no interval)"
    return (
        f"{interval['estimate']:.3f} "
        f"[{interval['low']:.3f}, {interval['high']:.3f}] "
        f"({interval['clusters']} scenarios)"
    )


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
        lines.append(f"pass rate: {_format_interval(group['pass_rate'])}")
        for k, interval in group["pass_k"].items():
            lines.append(f"pass^{k}: {_format_interval(interval)}")
        lines.append("checks:")
        for check in group["checks"]:
            suffix = f", {check['missing']} not computed" if check["missing"] else ""
            lines.append(
                f"  {check['id']} ({check['kind']}): {_format_interval(check['interval'])}{suffix}"
            )
        lines.append("scenarios:")
        for name, scenario in group["scenarios"].items():
            lines.append(f"  {name}: {scenario['passed']}/{scenario['trials']} passed")
        lines.append("")
    return "\n".join(lines).rstrip()
