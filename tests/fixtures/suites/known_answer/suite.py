"""Suite whose counterpart speaks fixed turns on fixed ticks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lambda_duplex.adapters.base import Capabilities, ModelContext
from lambda_duplex.suites.base import (
    Check,
    CheckKind,
    CheckResult,
    ConversationView,
    CounterpartPolicy,
    DeliveredSpan,
    Issue,
    ModelUtterance,
    PassDecision,
    PolicyOutput,
    Rubric,
    RunContext,
    RunResult,
    Scenario,
    SpeechSegment,
    SuiteEvent,
    SuiteManifest,
    Trajectory,
)


@dataclass
class ScriptPolicy:
    turns: dict[int, str]
    end_tick: int

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        return PolicyOutput()

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        text = self.turns.get(view.tick)
        return None if text is None else PolicyOutput((SpeechSegment(text),))

    def on_delivered(self, span: DeliveredSpan) -> Sequence[SuiteEvent]:
        return ()

    def is_finished(self, view: ConversationView) -> bool:
        return view.tick >= self.end_tick


class FinishedCheck:
    id = "finished"
    kind: CheckKind = "deterministic"

    def compute(self, trajectory: Trajectory) -> CheckResult:
        return CheckResult(self.id, self.kind, True, True)


class KnownAnswerSuite:
    manifest = SuiteManifest("known_answer", "1", "clean", ("script",), (0.2, 0.8))

    def scenario_schema(self) -> Mapping[str, Any]:
        return {"type": "object", "required": ["turns", "model", "end_tick"]}

    def load_scenario(self, raw: Mapping[str, Any]) -> Scenario:
        return Scenario.from_raw(raw, ())

    def validate_scenario(self, scenario: Scenario) -> Sequence[Issue]:
        return ()

    def model_context(self, scenario: Scenario, capabilities: Capabilities) -> ModelContext:
        return ModelContext("Talk.", "Talk.")

    def create_policy(self, scenario: Scenario, context: RunContext) -> CounterpartPolicy:
        turns = {int(tick): text for tick, text in scenario.data["turns"].items()}
        return ScriptPolicy(turns, int(scenario.data["end_tick"]))

    def checks(self) -> Sequence[Check]:
        return (FinishedCheck(),)

    def rubrics(self) -> Sequence[Rubric]:
        return ()

    def is_pass(self, result: RunResult) -> PassDecision:
        return PassDecision(True)

    def composite(self, result: RunResult) -> float | None:
        return None

    def mock_scripts(self, scenario: Scenario) -> Mapping[str, Mapping[int, str]]:
        return {"known_behavior": {int(k): v for k, v in scenario.data["model"].items()}}


def create_suite() -> KnownAnswerSuite:
    return KnownAnswerSuite()
