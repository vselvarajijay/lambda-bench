"""Named audio conditions used by benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from lambda_duplex.counterpart.effects import (
    ChannelConfig,
    EffectConfig,
    MuffleConfig,
    NoiseConfig,
)

VoiceMode = Literal["single", "varied"]
TurnTaking = Literal["strict", "free"]


@dataclass(frozen=True)
class Condition:
    """Audio and turn-taking settings for a run."""

    name: str
    voices: VoiceMode
    effects: EffectConfig
    turn_taking: TurnTaking


_NOISE = NoiseConfig()
_MUFFLE = MuffleConfig()
_CHANNEL = ChannelConfig()
CONDITIONS = {
    "clean": Condition("clean", "single", EffectConfig(), "strict"),
    "realistic": Condition("realistic", "varied", EffectConfig(_NOISE, _MUFFLE, _CHANNEL), "free"),
    "ablate_noise": Condition("ablate_noise", "single", EffectConfig(noise=_NOISE), "strict"),
    "ablate_channel": Condition(
        "ablate_channel",
        "single",
        EffectConfig(channel=ChannelConfig(drop_rate=0.0)),
        "strict",
    ),
    "ablate_accent": Condition("ablate_accent", "varied", EffectConfig(), "strict"),
    "ablate_turns": Condition("ablate_turns", "single", EffectConfig(), "free"),
}


def get_condition(name: str) -> Condition:
    """Return a named condition."""
    try:
        return CONDITIONS[name]
    except KeyError as error:
        known = ", ".join(sorted(CONDITIONS))
        raise ValueError(f"unknown condition {name!r}; available: {known}") from error


def condition_record(condition: Condition) -> dict[str, Any]:
    """Describe requested settings and what the current framework applied."""
    missing = []
    if condition.voices == "varied":
        missing.append("varied_voices")
    if condition.turn_taking == "free":
        missing.append("free_turn_taking")
    if condition.name == "realistic":
        missing.extend(("vocal_tics", "non_directed_speech"))
    return {
        "name": condition.name,
        "requested": {"voices": condition.voices, "turn_taking": condition.turn_taking},
        "applied": {"voices": "single", "turn_taking": "scripted"},
        "unimplemented": missing,
    }
