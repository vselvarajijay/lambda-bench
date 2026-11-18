"""Reliability statistics for repeated conversation trials."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """Point estimate with a percentile bootstrap interval."""

    estimate: float
    low: float
    high: float
    clusters: int


def pass_hat_k(trials: int, successes: int, k: int) -> float:
    """Return the chance that k trials drawn without replacement all passed."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if trials < k:
        raise ValueError(f"{trials} trials is fewer than k={k}")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    return math.comb(successes, k) / math.comb(trials, k)


def quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def clustered_bootstrap(
    clusters: Mapping[str, Sequence[float]],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Estimate a mean, resampling whole clusters such as scenarios."""
    if not clusters or any(not values for values in clusters.values()):
        raise ValueError("every cluster must contain at least one value")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    groups = [list(values) for _, values in sorted(clusters.items())]

    def mean(selection: Sequence[Sequence[float]]) -> float:
        values = [value for group in selection for value in group]
        return sum(values) / len(values)

    generator = random.Random(seed)
    samples = sorted(mean([generator.choice(groups) for _ in groups]) for _ in range(iterations))
    tail = (1 - confidence) / 2
    return Interval(mean(groups), quantile(samples, tail), quantile(samples, 1 - tail), len(groups))
