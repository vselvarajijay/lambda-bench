from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from lambda_duplex.adapters.base import Capabilities, MediaFrame, ModelContext, StepResult
from lambda_duplex.adapters.mock import MockAdapter
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
from lambda_duplex.suites.loader import (
    LoadedSuite,
    SuiteError,
    load_scenario_files,
    load_suite,
    validate_scenario_data,
)
from lambda_duplex.suites.runtime import ScenarioRun, decide_pass, mock_for, run_scenario

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = Scenario("s", 1, "dev", {"brief": "Talk."}, {"secret": "lantern", "items": ["f1"]})


@dataclass
class Policy:
    output_on_tick: dict[int, PolicyOutput] = field(default_factory=dict)
    reply: PolicyOutput | None = None
    delivered_events: tuple[SuiteEvent, ...] = ()
    finish_at: int | None = None
    finished_views: list[int] = field(default_factory=list)
    replies: int = 0

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput:
        self.replies += 1
        return self.reply or PolicyOutput()

    def on_tick(self, view: ConversationView) -> PolicyOutput | None:
        return self.output_on_tick.get(view.tick)

    def on_delivered(self, span: DeliveredSpan) -> Sequence[SuiteEvent]:
        return self.delivered_events

    def is_finished(self, view: ConversationView) -> bool:
        self.finished_views.append(view.tick)
        return self.finish_at is not None and view.tick >= self.finish_at


class AlwaysPassesCheck:
    id = "no_violation"
    kind: CheckKind = "deterministic"

    def compute(self, trajectory: Trajectory) -> CheckResult:
        return CheckResult(self.id, self.kind, True, True)


class CrashingCheck:
    id = "crash"
    kind: CheckKind = "deterministic"

    def compute(self, trajectory: Trajectory) -> CheckResult:
        raise IndexError("missing event")


@dataclass
class FakeSuite:
    policy: CounterpartPolicy
    suite_checks: tuple[Check, ...] = (AlwaysPassesCheck(),)
    context_text: str = "Talk."
    seen_scenarios: list[Scenario] = field(default_factory=list)
    manifest: SuiteManifest = field(
        default_factory=lambda: SuiteManifest("fake", "1", "clean", (), (0.0, 1.0))
    )

    def scenario_schema(self) -> Mapping[str, Any]:
        return {"type": "object"}

    def load_scenario(self, raw: Mapping[str, Any]) -> Scenario:
        return Scenario.from_raw(raw, ("secret",))

    def validate_scenario(self, scenario: Scenario) -> Sequence[Issue]:
        return ()

    def model_context(self, scenario: Scenario, capabilities: Capabilities) -> ModelContext:
        self.seen_scenarios.append(scenario)
        return ModelContext(self.context_text, self.context_text)

    def create_policy(self, scenario: Scenario, context: RunContext) -> CounterpartPolicy:
        scenario.hidden["items"].append("changed")
        return self.policy

    def checks(self) -> Sequence[Check]:
        return self.suite_checks

    def rubrics(self) -> Sequence[Rubric]:
        return ()

    def is_pass(self, result: RunResult) -> PassDecision:
        return PassDecision(True)

    def composite(self, result: RunResult) -> float | None:
        return None


class CrashingAdapter(MockAdapter):
    async def step(self, frame: MediaFrame, tick: int) -> StepResult:
        raise RuntimeError("model crashed")


def execute(suite: FakeSuite, adapter: MockAdapter | None = None, ticks: int = 10) -> ScenarioRun:
    scenario = replace(SCENARIO, hidden={"secret": "lantern", "items": ["f1"]})
    return asyncio.run(run_scenario(suite, scenario, adapter or MockAdapter(), max_ticks=ticks))


def spoken_ticks(run: ScenarioRun) -> list[int]:
    return [
        event.tick
        for event in run.outcome.events
        if event.source == "counterpart" and event.type == "spoke"
    ]


def test_model_context_never_receives_hidden_state() -> None:
    suite = FakeSuite(Policy(finish_at=0))

    execute(suite)

    assert suite.seen_scenarios[0].hidden == {}


def test_validation_rejects_context_leak() -> None:
    suite = FakeSuite(Policy(), context_text="The word is LANTERN.")
    raw = {"id": "s", "version": 1, "split": "dev", "secret": "lantern"}

    result = validate_scenario_data(LoadedSuite(suite, None), raw, "dev")

    assert {issue.path for issue in result.issues} == {
        "model_context.compact",
        "model_context.full",
    }


def test_identifier_fields_are_not_secrets() -> None:
    scenario = replace(SCENARIO, hidden={"facts": [{"id": "fact1", "content": "Hidden detail"}]})

    assert scenario.secrets() == ("Hidden detail",)


@pytest.mark.parametrize(
    ("status", "ended_by", "reason"),
    [("adapter_error", "error", "run adapter_error"), ("completed", "tick_limit", "tick_limit")],
)
def test_stored_failed_runs_cannot_pass(status: Any, ended_by: Any, reason: str) -> None:
    decision = decide_pass(FakeSuite(Policy()), RunResult("s", status, ended_by, ()))

    assert decision.passed is False
    assert reason in decision.reasons[0]


def test_tick_limit_is_not_a_pass_even_without_violations() -> None:
    run = execute(FakeSuite(Policy()), ticks=3)

    assert run.outcome.status == "completed"
    assert run.outcome.ended_by == "tick_limit"
    assert run.decision.passed is False


def test_zero_tick_run_cannot_pass() -> None:
    assert execute(FakeSuite(Policy()), ticks=0).decision.passed is False


def test_reserved_looking_event_keys_are_allowed() -> None:
    event = SuiteEvent("tagged", {"source": "whisper", "event_type": "x"})
    policy = Policy(
        {0: PolicyOutput((SpeechSegment("Hi"),))}, delivered_events=(event,), finish_at=1
    )

    run = execute(FakeSuite(policy))

    recorded = run.trajectory.suite_events("tagged")[0]
    assert run.outcome.status == "completed"
    assert recorded.data == {"source": "whisper", "event_type": "x"}
    assert recorded.origin == "delivery"


def test_policy_cannot_change_the_scored_scenario() -> None:
    assert execute(FakeSuite(Policy(finish_at=0))).trajectory.scenario.hidden["items"] == ["f1"]


def test_event_data_is_copied_when_recorded() -> None:
    values = [0]
    policy = Policy({0: PolicyOutput(events=(SuiteEvent("e", {"values": values}),))}, finish_at=2)
    original_on_tick = policy.on_tick

    def mutate(view: ConversationView) -> PolicyOutput | None:
        output = original_on_tick(view)
        if view.tick > 0:
            values.append(view.tick)
        return output

    policy.on_tick = mutate  # type: ignore[method-assign]
    run = execute(FakeSuite(policy))

    assert run.trajectory.suite_events("e")[0].data == {"values": [0]}


def test_unstorable_event_data_is_a_counterpart_error() -> None:
    event = SuiteEvent("bad", {"value": float("nan")})
    run = execute(FakeSuite(Policy({0: PolicyOutput(events=(event,))})))

    assert run.outcome.status == "counterpart_error"


def test_check_crash_on_failed_run_keeps_failure_status() -> None:
    suite = FakeSuite(Policy(), suite_checks=(CrashingCheck(),))

    run = execute(suite, CrashingAdapter())

    assert run.outcome.status == "adapter_error"
    assert run.result.check("crash").value is None
    assert "missing event" in run.result.check("crash").detail


def test_check_crash_on_clean_run_is_a_suite_bug() -> None:
    with pytest.raises(IndexError):
        execute(FakeSuite(Policy(finish_at=0), suite_checks=(CrashingCheck(),)))


def test_end_conversation_stops_policy_replies() -> None:
    ending = PolicyOutput((SpeechSegment("Bye"),), end_conversation=True)
    policy = Policy(reply=ending)
    adapter = MockAdapter({tick: "still talking" for tick in range(10)}, echo=False)

    run = execute(FakeSuite(policy), adapter)

    assert run.outcome.ended_by == "counterpart"
    assert policy.replies == 1
    assert len(spoken_ticks(run)) == 1


def test_finish_check_sees_current_tick_without_model_speech() -> None:
    policy = Policy(finish_at=3)

    run = execute(FakeSuite(policy))

    assert policy.finished_views == [0, 1, 2, 3]
    assert run.outcome.ticks == 3


def test_pause_after_segment_holds_following_speech() -> None:
    segments = (SpeechSegment("Well", pause_after_ms=320), SpeechSegment("yes"))
    policy = Policy({0: PolicyOutput(segments)}, finish_at=1)

    assert spoken_ticks(execute(FakeSuite(policy))) == [0, 3]


def test_preempt_replaces_queued_speech() -> None:
    first = PolicyOutput((SpeechSegment("one"), SpeechSegment("two")))
    second = PolicyOutput((SpeechSegment("now"),), preempt=True)
    policy = Policy({0: first, 1: second}, finish_at=2)

    run = execute(FakeSuite(policy))

    spoken = [
        event.data["text"]
        for event in run.outcome.events
        if event.source == "counterpart" and event.type == "spoke"
    ]
    assert spoken == ["one", "now"]


def test_multi_file_suite_can_use_relative_imports(tmp_path: Path) -> None:
    source = (ROOT / "suites/demo/suite.py").read_text(encoding="utf-8")
    (tmp_path / "helper.py").write_text("NAME = 'multi'\n", encoding="utf-8")
    (tmp_path / "suite.py").write_text(
        source.replace('name="demo"', "name=NAME").replace(
            "from dataclasses", "from .helper import NAME\nfrom dataclasses", 1
        ),
        encoding="utf-8",
    )

    assert load_suite(str(tmp_path)).suite.manifest.name == "multi"


def test_empty_split_is_an_error() -> None:
    with pytest.raises(SuiteError, match="no scenarios"):
        load_scenario_files(load_suite("demo", ROOT), "test")


def test_suite_without_mock_hook_still_loads_but_cannot_build_mock() -> None:
    with pytest.raises(ValueError, match="mock behaviors"):
        mock_for(FakeSuite(Policy()), SCENARIO)


def test_invalid_suite_schema_is_reported() -> None:
    class BadSchema(FakeSuite):
        def scenario_schema(self) -> Mapping[str, Any]:
            return {"type": "not-a-type"}

    raw = {"id": "s", "version": 1, "split": "dev"}
    result = validate_scenario_data(LoadedSuite(BadSchema(Policy()), None), raw, "dev")

    assert result.issues[0].path == "$schema"
