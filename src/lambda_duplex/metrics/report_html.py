"""Render a suite report as one self-contained HTML page."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from html import escape
from typing import Any

from lambda_duplex.metrics.report import CHECK_KINDS, format_interval

STYLE = """
:root { color-scheme: light dark; --bg: #fff; --fg: #1c2024; --muted: #5c6670;
  --line: #d8dde2; --panel: #f5f7f8; --warn: #fff4d6; --warn-fg: #5f4500; }
@media (prefers-color-scheme: dark) { :root { --bg: #14171a; --fg: #e8ebee;
  --muted: #a2acb6; --line: #333a41; --panel: #1d2125; --warn: #3a2f12; --warn-fg: #f4dc9c; } }
body { margin: 0 auto; max-width: 960px; padding: 24px 16px; background: var(--bg);
  color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }
h1 { font-size: 26px; } h2 { margin-top: 40px; font-size: 20px; overflow-wrap: anywhere; }
h3 { margin-top: 24px; font-size: 16px; }
.table { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: left; }
th { color: var(--muted); font-weight: 600; }
.note { padding: 8px 12px; background: var(--panel); }
.warning { padding: 8px 12px; background: var(--warn); color: var(--warn-fg); }
.empty { color: var(--muted); }
"""
POLICY = "default-src 'none'; style-src 'unsafe-inline'"
JUDGE_NOTE = (
    "Judge scores are non-deterministic and uncalibrated. They are never used in pass decisions."
)
SECTION_TITLES = {
    "deterministic": "Deterministic checks",
    "classified": "Classified checks",
    "judge": "Judge checks stored in results (non-deterministic)",
}


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    head = "".join(f'<th scope="col">{_e(name)}</th>' for name in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_e(cell)}</td>" for cell in row) + "</tr>\n" for row in rows
    )
    return (
        f'<div class="table"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>\n{body}</tbody></table></div>"
    )


def _counts(values: Mapping[str, int]) -> str:
    return ", ".join(f"{name}: {count}" for name, count in values.items()) or "none"


def _validity(group: Mapping[str, Any]) -> str:
    validity = group["validity"]
    rows = [
        ("Runs", group["runs"]),
        ("Valid runs", group["valid_runs"]),
        ("Invalid runs", group["invalid_runs"]),
        ("Run status", _counts(validity["statuses"])),
        ("Ended by", _counts(validity["ended_by"])),
        ("Checks not computed", validity["checks_not_computed"]),
    ]
    return "<h3>Validity</h3>" + _table(("Measure", "Count"), rows)


def _reliability(group: Mapping[str, Any]) -> str:
    rows = [("Pass rate", format_interval(group["pass_rate"]))]
    rows.extend((f"pass^{k}", format_interval(value)) for k, value in group["pass_k"].items())
    scenarios = [
        (name, cell["trials"], cell["passed"]) for name, cell in group["scenarios"].items()
    ]
    return (
        "<h3>Reliability</h3>"
        + _table(("Measure", "Estimate [95% interval]"), rows)
        + "<h3>Scenarios</h3>"
        + _table(("Scenario", "Trials", "Passed"), scenarios)
    )


def _check_section(title: str, kind: str, checks: Sequence[Mapping[str, Any]]) -> str:
    chosen = [check for check in checks if check["kind"] == kind]
    heading = f'<section data-kind="{_e(kind)}"><h3>{_e(title)}</h3>'
    if not chosen:
        return f'{heading}<p class="empty">None in these runs.</p></section>'
    rows = [(check["id"], format_interval(check["interval"]), check["missing"]) for check in chosen]
    table = _table(("Check", "Estimate [95% interval]", "Not computed"), rows)
    return f"{heading}{table}</section>"


def _checks(checks: Sequence[Mapping[str, Any]]) -> str:
    known = [_check_section(SECTION_TITLES[kind], kind, checks) for kind in CHECK_KINDS]
    other = sorted({check["kind"] for check in checks} - set(CHECK_KINDS))
    return "".join(known + [_check_section(f"Other: {kind}", kind, checks) for kind in other])


def _composite(composite: Mapping[str, Any]) -> str:
    status = composite["status"]
    opening = f'<section class="composite" data-status="{_e(status)}"><h3>Composite</h3>'
    if status == "shown":
        summary = (
            "<p>Weighted sum of parts: "
            f"<strong>{_e(format_interval(composite['interval']))}</strong>. "
            f"Not computed for {_e(composite['missing'])} runs.</p>"
        )
    else:
        summary = f'<p class="empty">No composite shown: {_e(composite["reason"])}.</p>'
    parts = ""
    if composite["parts"]:
        rows = [
            (
                part["id"],
                part["kind"],
                f"{part['weight']:g}",
                format_interval(part["interval"]),
                part["missing"],
            )
            for part in composite["parts"]
        ]
        headers = ("Part", "Kind", "Weight", "Estimate [95% interval]", "Not computed")
        parts = _table(headers, rows)
    return f"{opening}{summary}{parts}</section>"


def _rubric(rubric: Mapping[str, Any]) -> str:
    identity = rubric["rubric"]
    title = f"{identity['id']} version {identity['version']} ({identity['sha256'][:12]})"
    judge = f"{rubric['judge']['provider']}:{rubric['judge']['model']}"
    counts = (
        ("Scored", rubric["scored"]),
        ("Failed", rubric["failed"]),
        ("Unscorable", rubric["unscorable"]),
        ("Malformed", rubric["malformed"]),
        ("Stale (needs re-judging)", rubric["stale"]),
        ("Not judged", rubric["not_judged"]),
    )
    items = [
        (
            item["id"],
            f"{item['scale']['min']} to {item['scale']['max']}",
            format_interval(item["interval"]),
            item["not_applicable"],
        )
        for item in rubric["items"]
    ]
    table = '<p class="empty">No scores.</p>'
    if items:
        table = _table(("Item", "Scale", "Mean [95% interval]", "Not applicable"), items)
    return (
        f"<h4>{_e(title)}</h4><p>Judge {_e(judge)}, seed {_e(rubric['seed'])}, "
        f"{_e(rubric['calibration'])}, non-deterministic.</p>"
        + _table(("Judgments", "Runs"), counts)
        + table
    )


def _judges(judges: Mapping[str, Any]) -> str:
    parts = [
        '<section class="judges"><h3>Judge scores</h3>',
        f'<p class="warning">{_e(JUDGE_NOTE)}</p>',
    ]
    if judges["unreadable_runs"]:
        names = ", ".join(judges["unreadable_runs"])
        parts.append(f'<p class="warning">Unreadable judge files: {_e(names)}.</p>')
    if not judges["rubrics"]:
        parts.append('<p class="empty">No judgments in these runs.</p>')
    parts.extend(_rubric(rubric) for rubric in judges["rubrics"])
    parts.append("</section>")
    return "".join(parts)


def _group(group: Mapping[str, Any]) -> str:
    title = f"{group['suite']} / {group['model']} / {group['condition']}"
    return (
        f'<section class="group"><h2>{_e(title)}</h2>'
        + _validity(group)
        + _reliability(group)
        + _checks(group["checks"])
        + _composite(group["composite"])
        + _judges(group["judges"])
        + "</section>\n"
    )


def format_html(report: Mapping[str, Any]) -> str:
    """Return a complete page with no scripts or external resources."""
    groups = "".join(_group(group) for group in report["groups"])
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{_e(POLICY)}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Suite report</title><style>{STYLE}</style></head>\n<body>"
        '<h1>Suite report</h1><p class="note">Pass rates come only from stored '
        "pass decisions. Intervals resample whole scenarios.</p>\n"
        f"{groups}</body></html>"
    )
