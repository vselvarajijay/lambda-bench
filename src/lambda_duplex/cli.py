"""Command-line interface for lambda-duplex."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from lambda_duplex import __version__

MINIMUM_PYTHON = (3, 12)
OPTIONAL_MODULES = {
    "silero_vad": "speech activity timelines",
    "faster_whisper": "model audio transcription",
    "pipecat": "cascaded baseline adapter",
    "openai": "OpenAI Realtime adapter",
    "google.genai": "Gemini Live adapter",
    "websockets": "Grok Voice adapter",
}


@dataclass(frozen=True)
class Check:
    """One environment check reported by ``doctor``."""

    name: str
    ok: bool
    required: bool
    detail: str


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def collect_checks() -> list[Check]:
    """Return required and optional environment checks."""
    python_ok = sys.version_info[:2] >= MINIMUM_PYTHON
    checks = [
        Check(
            name="python",
            ok=python_ok,
            required=True,
            detail=f"{platform.python_version()} (requires 3.12 or newer)",
        )
    ]
    for module, purpose in OPTIONAL_MODULES.items():
        available = _module_available(module)
        state = "installed" if available else "not installed"
        checks.append(Check(module, available, False, f"{state}; used for {purpose}"))
    return checks


def _doctor(as_json: bool) -> int:
    checks = collect_checks()
    healthy = all(check.ok for check in checks if check.required)
    if as_json:
        print(json.dumps({"ok": healthy, "checks": [asdict(check) for check in checks]}, indent=2))
    else:
        for check in checks:
            status = "ok" if check.ok else ("FAIL" if check.required else "missing")
            print(f"{status:7} {check.name}: {check.detail}")
    return 0 if healthy else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="lambda-duplex", description=__doc__)
    parser.add_argument("--version", action="version", version=f"lambda-duplex {__version__}")
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="check the local environment")
    doctor.add_argument("--json", action="store_true", help="print machine-readable output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.json)
    parser.print_help()
    return 0
