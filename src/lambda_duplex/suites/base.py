"""Public contract between the core framework and benchmark suites."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from lambda_duplex.adapters.base import Capabilities, ModelContext
from lambda_duplex.llm import LLMGateway
from lambda_duplex.orchestrator.engine import EndedBy, Event, RunStatus
from lambda_duplex.trajectory.seeds import random_stream

CheckKind = Literal["deterministic", "classified", "judge"]
Split = Literal["dev", "test"]
NON_SECRET_KEYS = frozenset(
    {
        "id",
        "topic",
        "tier",
        "type",
        "unlock",
        "trigger",
        "severity",
        "memory_distance",
        "success",
        "traits",
        "voice_id",
        "turn_decider_model",
    }
)


@dataclass(frozen=True)
class Issue:
    """A scenario validation problem."""

    path: str
    message: str


@dataclass(frozen=True)
class Scenario:
    """Validated scenario with model-safe ``data`` and simulator-only ``hidden`` state."""

    id: str
    version: int
    split: Split
    data: Mapping[str, Any]
    hidden: Mapping[str, Any]

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any], hidden_keys: Sequence[str]) -> Scenario:
        """Split raw scenario data into public and hidden fields."""
        core = {"id", "version", "split"}
        hidden_names = set(hidden_keys)
        return cls(
            id=raw["id"],
            version=raw["version"],
            split=raw["split"],
            data={k: v for k, v in raw.items() if k not in core | hidden_names},
            hidden={k: v for k, v in raw.items() if k in hidden_names},
        )

    def secrets(self) -> tuple[str, ...]:
        """Return hidden text that must never appear in model-visible context."""
        found: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, str):
                if len(value.strip()) >= 4:
                    found.append(value.strip())
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    if key not in NON_SECRET_KEYS:
                        walk(item)
            elif isinstance(value, list | tuple):
                for item in value:
                    walk(item)

        walk(self.hidden)
        return tuple(found)


@dataclass(frozen=True)
class SuiteManifest:
    """Stable suite metadata and defaults."""

    name: str
    version: str
    default_condition: str
    default_personas: tuple[str, ...]
    talk_ratio_target: tuple[float, float]
    goal_event: str | None = None
    composite_weights: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RunContext:
    """Run settings available to a counterpart policy."""

    seed: int
    condition: str
    persona: str | None = None
    llm: LLMGateway = field(default_factory=LLMGateway)

    def rng(self, name: str) -> random.Random:
        """Return the named random stream for this run."""
        return random_stream(self.seed, name)


@dataclass(frozen=True)
class SuiteEvent:
    """Structured suite fact recorded in the trajectory."""

    type: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeechSegment:
    """Text the counterpart plans to say, with optional credit tags."""

    text: str
    tags: tuple[str, ...] = ()
    pause_after_ms: int = 0


@dataclass(frozen=True)
class PolicyOutput:
    """Counterpart speech and events produced by a policy."""

    segments: tuple[SpeechSegment, ...] = ()
    events: tuple[SuiteEvent, ...] = ()
    end_conversation: bool = False
    preempt: bool = False


@dataclass(frozen=True)
class ModelUtterance:
    """A model utterance after transcription or text delivery."""

    text: str
    tick: int


@dataclass(frozen=True)
class DeliveredSpan:
    """Counterpart speech delivered to the model.

    ``tags`` are credit tags. They are empty unless the tagged segment played in full.
    """

    segment_id: int
    text: str
    tags: tuple[str, ...]
    start_tick: int
    end_tick: int
    interrupted: bool = False


@dataclass(frozen=True)
class ConversationView:
    """Read-only conversation state available to a policy."""

    tick: int
    time_ms: int
    model_utterances: tuple[ModelUtterance, ...]
    delivered: tuple[DeliveredSpan, ...]


class CounterpartPolicy(Protocol):
    """Suite logic that decides what the simulated speaker does."""

    async def on_model_utterance(
        self, utterance: ModelUtterance, view: ConversationView
    ) -> PolicyOutput: ...

    def on_tick(self, view: ConversationView) -> PolicyOutput | None: ...

    def on_delivered(self, span: DeliveredSpan) -> Sequence[SuiteEvent]: ...

    def is_finished(self, view: ConversationView) -> bool: ...


@dataclass(frozen=True)
class Trajectory:
    """Stored evidence that checks may read."""

    scenario: Scenario
    status: RunStatus
    ended_by: EndedBy
    ticks: int
    tick_ms: int
    events: tuple[Event, ...]

    def suite_events(self, event_type: str | None = None) -> tuple[Event, ...]:
        return tuple(
            event
            for event in self.events
            if event.source == "suite" and (event_type is None or event.type == event_type)
        )


@dataclass(frozen=True)
class CheckResult:
    """Result from one suite check."""

    id: str
    kind: CheckKind
    value: float | bool | None
    passed: bool | None = None
    detail: str = ""


@dataclass(frozen=True)
class RunResult:
    """Scored checks for one run."""

    scenario_id: str
    status: RunStatus
    ended_by: EndedBy
    checks: tuple[CheckResult, ...]

    def check(self, check_id: str) -> CheckResult:
        for result in self.checks:
            if result.id == check_id:
                return result
        raise KeyError(check_id)


@dataclass(frozen=True)
class PassDecision:
    """Whether a run passed, with reasons when it did not."""

    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rubric:
    """Judge rubric supplied by a suite."""

    id: str
    version: str
    prompt: str


class Check(Protocol):
    """Metric or rule computed from a stored trajectory."""

    id: str
    kind: CheckKind

    def compute(self, trajectory: Trajectory) -> CheckResult: ...


@runtime_checkable
class Suite(Protocol):
    """Plugin that defines one conversational use case."""

    manifest: SuiteManifest

    def scenario_schema(self) -> Mapping[str, Any]: ...

    def load_scenario(self, raw: Mapping[str, Any]) -> Scenario: ...

    def validate_scenario(self, scenario: Scenario) -> Sequence[Issue]: ...

    def model_context(self, scenario: Scenario, capabilities: Capabilities) -> ModelContext: ...

    def create_policy(self, scenario: Scenario, context: RunContext) -> CounterpartPolicy: ...

    def checks(self) -> Sequence[Check]: ...

    def rubrics(self) -> Sequence[Rubric]: ...

    def is_pass(self, result: RunResult) -> PassDecision: ...

    def composite(self, result: RunResult) -> float | None: ...


@runtime_checkable
class MockScripted(Protocol):
    """Optional suite hook that supplies scripted mock behaviors."""

    def mock_scripts(self, scenario: Scenario) -> Mapping[str, Mapping[int, str]]: ...
