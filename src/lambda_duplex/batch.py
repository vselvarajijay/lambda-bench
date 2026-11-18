"""Plan and run repeatable benchmark batches."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lambda_duplex.audio.pcm import INTERNAL_SAMPLE_RATE
from lambda_duplex.suites.base import Scenario
from lambda_duplex.suites.loader import LoadedSuite, valid_scenarios
from lambda_duplex.suites.runtime import ScenarioRun, mock_for, run_scenario
from lambda_duplex.trajectory.seeds import trial_seed
from lambda_duplex.trajectory.store import RunSpec, TrajectoryError, load_run, write_run

PLAN_FILE = "batch-plan.json"
DEFAULT_CALL_MINUTES = 8.0
DEFAULT_MAX_TICKS = 3000


class BatchError(RuntimeError):
    """A batch cannot be planned or run safely."""


@dataclass(frozen=True)
class Tier:
    scenarios: int
    personas: int
    trials: int
    conditions: tuple[str, ...]


TIERS = {
    "smoke": Tier(1, 1, 1, ("clean",)),
    "pilot": Tier(5, 1, 1, ("clean",)),
    "dev": Tier(10, 4, 2, ("clean", "realistic")),
    "full": Tier(30, 4, 4, ("clean", "realistic")),
    "ablations": Tier(
        10,
        2,
        2,
        ("ablate_noise", "ablate_channel", "ablate_accent", "ablate_turns"),
    ),
}


@dataclass(frozen=True)
class BatchItem:
    id: str
    scenario_id: str
    persona: str
    condition: str
    trial: int
    seed: int


@dataclass(frozen=True)
class BatchPlan:
    suite: str
    suite_reference: str
    model: str
    tier: str
    split: str
    seed: int
    max_ticks: int
    call_minutes: float
    requested_runs: int
    items: tuple[BatchItem, ...]
    warnings: tuple[str, ...]

    @property
    def audio_minutes(self) -> float:
        return len(self.items) * self.call_minutes

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["audio_minutes"] = self.audio_minutes
        data["fingerprint"] = plan_fingerprint(self)
        return data


def plan_fingerprint(plan: BatchPlan) -> str:
    data = asdict(plan)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _call_minutes(scenarios: list[Scenario]) -> float:
    budgets = []
    for scenario in scenarios:
        brief = scenario.data.get("brief")
        if isinstance(brief, dict) and isinstance(brief.get("time_budget_s"), int | float):
            budgets.append(float(brief["time_budget_s"]) / 60)
    return max(budgets, default=DEFAULT_CALL_MINUTES)


def plan_batch(
    loaded: LoadedSuite,
    suite_reference: str,
    tier_name: str,
    model: str,
    *,
    split: str = "dev",
    seed: int = 0,
    max_ticks: int = DEFAULT_MAX_TICKS,
) -> BatchPlan:
    """Build the run matrix available for a suite and tier."""
    try:
        tier = TIERS[tier_name]
    except KeyError as error:
        raise BatchError(f"unknown tier {tier_name}") from error
    if split not in {"dev", "test"}:
        raise BatchError("split must be dev or test")
    scenarios = valid_scenarios(loaded, split)  # type: ignore[arg-type]
    selected = scenarios[: tier.scenarios]
    personas = list(loaded.suite.manifest.default_personas[: tier.personas]) or ["default"]
    warnings: list[str] = []
    if len(selected) < tier.scenarios:
        warnings.append(f"only {len(selected)} of {tier.scenarios} scenarios are available")
    if len(personas) < tier.personas:
        warnings.append(f"only {len(personas)} of {tier.personas} personas are available")
    items = [
        BatchItem(
            f"{scenario.id}__{persona}__{condition}__trial{trial}",
            scenario.id,
            persona,
            condition,
            trial,
            trial_seed(seed, scenario.id, trial),
        )
        for scenario in selected
        for persona in personas
        for condition in tier.conditions
        for trial in range(tier.trials)
    ]
    requested = tier.scenarios * tier.personas * tier.trials * len(tier.conditions)
    return BatchPlan(
        loaded.suite.manifest.name,
        suite_reference,
        model,
        tier_name,
        split,
        seed,
        max_ticks,
        _call_minutes(selected),
        requested,
        tuple(items),
        tuple(warnings),
    )


def estimate_cost(plan: BatchPlan, cost_per_minute: float, runs: int | None = None) -> float:
    if cost_per_minute < 0:
        raise BatchError("cost per minute must not be negative")
    count = len(plan.items) if runs is None else runs
    return count * plan.call_minutes * cost_per_minute


def run_is_complete(path: Path) -> bool:
    if not (path / "manifest.json").is_file():
        return False
    try:
        return load_run(path).result.status != "counterpart_error"
    except TrajectoryError:
        return False


Runner = Callable[[LoadedSuite, Scenario, BatchPlan, BatchItem], Awaitable[ScenarioRun]]
Completion = Callable[[BatchItem], None]


async def run_mock_item(
    loaded: LoadedSuite, scenario: Scenario, plan: BatchPlan, item: BatchItem
) -> ScenarioRun:
    """Run one item with the mock model and synthetic speech."""
    from lambda_duplex.audio.synthetic import SyntheticSpeech

    model, _, behavior = plan.model.partition(":")
    if model != "mock":
        raise BatchError("only mock models can run in batches so far")
    speech = SyntheticSpeech(item.seed, INTERNAL_SAMPLE_RATE)
    adapter = mock_for(loaded.suite, scenario, behavior or "good", speech)
    return await run_scenario(
        loaded.suite,
        scenario,
        adapter,
        seed=item.seed,
        max_ticks=plan.max_ticks,
        condition=item.condition,
        speech=speech,
    )


def _prepare_output(plan: BatchPlan, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / PLAN_FILE
    current = plan.to_dict()
    if plan_path.is_file():
        stored = json.loads(plan_path.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != current["fingerprint"]:
            raise BatchError("output folder belongs to a different batch plan")
    else:
        plan_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for child in output.iterdir():
        if child.is_dir() and child.name.startswith("."):
            shutil.rmtree(child)


async def run_batch(
    loaded: LoadedSuite,
    plan: BatchPlan,
    output: Path,
    *,
    jobs: int = 1,
    retries: int = 1,
    cost_per_minute: float = 0.0,
    budget: float | None = None,
    runner: Runner = run_mock_item,
    on_complete: Completion | None = None,
) -> dict[str, int]:
    """Run missing items, retrying simulator failures and resuming safely."""
    if jobs < 1 or retries < 0:
        raise BatchError("jobs must be positive and retries must not be negative")
    _prepare_output(plan, output)
    pending = [item for item in plan.items if not run_is_complete(output / item.id)]
    projected = estimate_cost(plan, cost_per_minute, len(pending))
    if budget is not None and projected > budget:
        raise BatchError(f"estimated cost {projected:.2f} exceeds budget {budget:.2f}")
    scenarios = {scenario.id: scenario for scenario in valid_scenarios(loaded, plan.split)}  # type: ignore[arg-type]
    limit = asyncio.Semaphore(jobs)
    counts = {
        "planned": len(plan.items),
        "skipped": len(plan.items) - len(pending),
        "completed": 0,
        "invalid": 0,
    }

    stopping = asyncio.Event()

    async def run_item(item: BatchItem) -> None:
        async with limit:
            if stopping.is_set():
                return
            try:
                await execute_item(item)
            except BaseException:
                stopping.set()
                raise

    async def execute_item(item: BatchItem) -> None:
        run: ScenarioRun | None = None
        for _ in range(retries + 1):
            run = await runner(loaded, scenarios[item.scenario_id], plan, item)
            if run.outcome.status != "counterpart_error":
                break
        assert run is not None
        if run.outcome.status == "counterpart_error":
            counts["invalid"] += 1
            return
        spec = RunSpec(
            plan.suite_reference,
            plan.model,
            "audio",
            item.seed,
            plan.max_ticks,
            run.trajectory.tick_ms,
            item.condition,
            audio_backend="synthetic",
        )
        write_run(
            output / item.id,
            loaded.suite,
            spec,
            _model_identity(plan),
            run.trajectory,
            run.result,
            run.decision,
            tracks=run.tracks,
            effects=run.effects,
            condition=run.condition_details,
            transcript={"mode": "injected"},
        )
        counts["completed"] += 1
        if on_complete is not None:
            on_complete(item)

    tasks = [asyncio.create_task(run_item(item)) for item in pending]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return counts


def _model_identity(plan: BatchPlan) -> Any:
    from lambda_duplex.adapters.base import ModelIdentity

    return ModelIdentity("mock", "1", {"reference": plan.model})
