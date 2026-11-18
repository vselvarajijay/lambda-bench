"""Model adapter contract shared by local, hosted, and mock models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ClockMode = Literal["lockstep", "realtime"]


@dataclass(frozen=True)
class MediaFrame:
    """Media exchanged during one tick."""

    tick: int
    pcm: bytes = b""
    sample_rate: int | None = None
    text: str | None = None
    clean_pcm: bytes | None = None


@dataclass(frozen=True)
class AudioRates:
    """Sample rates accepted and produced by a model."""

    input_hz: int
    output_hz: int


@dataclass(frozen=True)
class Capabilities:
    """Declared model features that affect how results are interpreted."""

    clock_modes: frozenset[ClockMode]
    sample_rates: AudioRates | None = None
    native_frame_ms: int | None = None
    exposes_text_stream: bool = False
    system_prompt_max_tokens: int | None = None
    context_window_s: float | None = None
    context_injection: bool = False
    tools: bool = False
    server_side_vad: bool = False
    deterministic_seed: bool = False


@dataclass(frozen=True)
class ModelIdentity:
    """Exact identity of the system under test."""

    name: str
    version: str
    settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """Tool a suite offers to models that declare tool support."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelContext:
    """Information the model may see. It must never contain hidden suite state."""

    full: str = ""
    compact: str = ""
    tools: tuple[ToolSpec, ...] = ()


@dataclass(frozen=True)
class SessionConfig:
    """Settings for one model session."""

    seed: int
    tick_ms: int
    clock_mode: ClockMode


@dataclass(frozen=True)
class SessionInfo:
    """Details returned after a model session starts."""

    session_id: str


@dataclass(frozen=True)
class StepResult:
    """Exactly one tick of model output."""

    output: MediaFrame
    truncated_ms: int = 0
    compute_ms: float | None = None
    provider_events: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class SessionSummary:
    """Details returned after a model session stops."""

    steps: int


class ModelAdapter(Protocol):
    """Interface implemented by every model under test."""

    capabilities: Capabilities

    def describe(self) -> ModelIdentity: ...

    async def start(self, context: ModelContext, config: SessionConfig) -> SessionInfo: ...

    async def step(self, frame: MediaFrame, tick: int) -> StepResult: ...

    async def stop(self) -> SessionSummary: ...
