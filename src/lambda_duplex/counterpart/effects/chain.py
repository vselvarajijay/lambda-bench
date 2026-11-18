"""Streaming effect chain applied one tick at a time."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from lambda_duplex.trajectory.seeds import stream_seed

SAMPLE_RATE = 24_000
TELEPHONY_DECIMATION = 3
PACKET_MS = 20
NOMINAL_SPEECH_RMS = 10 ** (-26 / 20)
ACTIVE_RMS_FLOOR = 10 ** (-50 / 20)
ACTIVE_RELATIVE_DB = -30.0
NOISE_FRAME_MS = 100
MIN_ACTIVE_NOISE_SAMPLES = SAMPLE_RATE
MAX_NOISE_LEVEL_SPREAD_DB = 1.0
MUFFLE_FILTER_DELAY = 31
TELEPHONY_FILTER_DELAY = 63
MULAW_BIAS = 132
MULAW_CLIP = 32_635


@dataclass(frozen=True)
class NoiseConfig:
    snr_db: float = 15.0
    drift_db: float = 3.0
    bursts_per_minute: float = 1.0
    burst_snr_db: tuple[float, float] = (5.0, 15.0)


@dataclass(frozen=True)
class MuffleConfig:
    events_per_minute: float = 2.0
    duration_ms: int = 1500
    cutoff_hz: float = 1200.0
    transition_ms: int = 20


@dataclass(frozen=True)
class ChannelConfig:
    telephony: bool = True
    drop_rate: float = 0.02
    drop_burst_ms: float = 100.0


@dataclass(frozen=True)
class EffectConfig:
    noise: NoiseConfig | None = None
    muffle: MuffleConfig | None = None
    channel: ChannelConfig | None = None


def mulaw_encode(samples: np.ndarray) -> np.ndarray:
    """Encode signed 16-bit samples with ITU-T G.711 mu-law."""
    values = np.asarray(samples, dtype=np.int32)
    sign = np.where(values < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(values), MULAW_CLIP) + MULAW_BIAS
    exponent = np.floor(np.log2(magnitude)).astype(np.int32) - 7
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return np.asarray(~(sign | (exponent << 4) | mantissa) & 0xFF, dtype=np.uint8)


def mulaw_decode(encoded: np.ndarray) -> np.ndarray:
    """Decode ITU-T G.711 mu-law bytes to signed 16-bit samples."""
    values = ~np.asarray(encoded, dtype=np.int32) & 0xFF
    exponent = (values >> 4) & 0x07
    mantissa = values & 0x0F
    magnitude = (((mantissa << 3) + MULAW_BIAS) << exponent) - MULAW_BIAS
    return np.asarray(np.where(values & 0x80, -magnitude, magnitude), dtype=np.int16)


def lowpass_taps(cutoff_hz: float, count: int) -> np.ndarray:
    """Design a unity-gain Hamming-window FIR low-pass filter."""
    center = (count - 1) / 2
    positions = np.arange(count) - center
    taps = np.sinc(2 * cutoff_hz / SAMPLE_RATE * positions) * np.hamming(count)
    result: np.ndarray = taps / np.sum(taps)
    return result


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


def _frame_levels(samples: np.ndarray, frame: int) -> np.ndarray:
    count = samples.size // frame
    if count == 0:
        return np.zeros(0)
    frames = samples[: count * frame].reshape(count, frame)
    return np.asarray(np.sqrt(np.mean(frames**2, axis=1)))


def active_rms(pcm: bytes) -> float:
    """Measure active speech level from 20 ms frames, as a fraction of full scale."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
    levels = _frame_levels(samples, SAMPLE_RATE * PACKET_MS // 1000)
    if levels.size == 0 or float(np.max(levels)) < ACTIVE_RMS_FLOOR:
        return NOMINAL_SPEECH_RMS
    gate = max(ACTIVE_RMS_FLOOR, float(np.max(levels)) * 10 ** (ACTIVE_RELATIVE_DB / 20))
    active = levels[levels >= gate]
    return float(np.sqrt(np.mean(active**2)))


def active_noise_bed(samples: np.ndarray) -> np.ndarray:
    """Keep active noise regions and normalize them to unit RMS."""
    frame = SAMPLE_RATE * NOISE_FRAME_MS // 1000
    levels = _frame_levels(samples, frame)
    if levels.size == 0 or float(np.max(levels)) <= 0:
        raise ValueError("noise recording is empty or silent")
    gate = float(np.max(levels)) * 10 ** (ACTIVE_RELATIVE_DB / 20)
    active = levels >= gate
    parts: list[np.ndarray] = []
    fade = SAMPLE_RATE // 200
    start: int | None = None
    for index, value in enumerate([*active.tolist(), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            part = samples[start * frame : index * frame].copy()
            ramp = np.linspace(0, 1, min(fade, part.size // 2))
            if ramp.size:
                part[: ramp.size] *= ramp
                part[-ramp.size :] *= ramp[::-1]
            parts.append(part)
            start = None
    bed = np.concatenate(parts)
    if bed.size < MIN_ACTIVE_NOISE_SAMPLES:
        raise ValueError("noise recording has too little active audio")
    median_level = float(np.median(levels[active]))
    spread_db = abs(20 * np.log10(_rms(bed) / max(median_level, 1e-12)))
    if spread_db > MAX_NOISE_LEVEL_SPREAD_DB:
        raise ValueError("noise recording level is too uneven")
    return bed / max(_rms(bed), 1e-12)


class _Fir:
    def __init__(self, taps: np.ndarray) -> None:
        self.taps = taps
        self.history = np.zeros(taps.size - 1)

    def process(self, samples: np.ndarray) -> np.ndarray:
        joined = np.concatenate((self.history, samples))
        self.history = joined[-self.history.size :]
        return np.convolve(joined, self.taps, mode="valid")


@dataclass
class _Burst:
    samples: np.ndarray
    position: int = 0


@dataclass
class _Metadata:
    bursts: list[dict[str, float]] = field(default_factory=list)
    muffles: list[dict[str, float]] = field(default_factory=list)
    dropped_packets: list[int] = field(default_factory=list)


class EffectChain:
    """Apply repeatable effects to 16-bit mono PCM ticks."""

    def __init__(
        self,
        config: EffectConfig,
        seed: int,
        samples_per_tick: int = 3840,
        noise_bed: np.ndarray | None = None,
        noise_source: dict[str, Any] | None = None,
        speech_rms: float = NOMINAL_SPEECH_RMS,
    ) -> None:
        if samples_per_tick % TELEPHONY_DECIMATION:
            raise ValueError("tick must contain a whole number of telephony samples")
        self.config = config
        self.seed = seed
        self.samples_per_tick = samples_per_tick
        if speech_rms <= 0:
            raise ValueError("speech_rms must be positive")
        self.speech_rms = speech_rms
        self.noise_source = None
        if config.noise:
            self.noise_source = noise_source or {"type": "generated"}
        self.tick = 0
        self._tick_s = samples_per_tick / SAMPLE_RATE
        self._metadata = _Metadata()
        self._noise_offset: int | None = None
        self._bursts: list[_Burst] = []
        self._muffle_end = -1
        self._muffle_level = 0.0
        self._rngs = {
            name: np.random.default_rng(stream_seed(seed, f"effects:{name}"))
            for name in ("noise", "burst", "muffle", "drops")
        }
        self._noise = self._prepare_noise(noise_bed)
        self._drift_phase = float(self._rngs["noise"].uniform(0, 2 * np.pi))
        self._muffle_filter = None
        self._dry_delay = None
        if config.muffle:
            self._muffle_filter = _Fir(lowpass_taps(config.muffle.cutoff_hz, 63))
            delay = np.zeros(63)
            delay[MUFFLE_FILTER_DELAY] = 1.0
            self._dry_delay = _Fir(delay)
        telephone = lowpass_taps(3650.0, 127)
        self._down = _Fir(telephone)
        self._up = _Fir(telephone * TELEPHONY_DECIMATION)
        self._bad_channel = False
        self._packet = 0

    def _prepare_noise(self, bed: np.ndarray | None) -> np.ndarray | None:
        if self.config.noise is None:
            return None
        rng = self._rngs["noise"]
        if bed is None:
            white = rng.normal(0.0, 1.0, SAMPLE_RATE * 20)
            spectrum = np.fft.rfft(white)
            frequency = np.maximum(np.fft.rfftfreq(white.size, 1 / SAMPLE_RATE), 40.0)
            bed = np.fft.irfft(spectrum / np.sqrt(frequency), white.size)
        self._noise_offset = int(rng.integers(0, bed.size))
        rolled = np.roll(np.asarray(bed, dtype=np.float64), -self._noise_offset)
        return rolled / max(_rms(rolled), 1e-12)

    def _new_burst(self, config: NoiseConfig, reference: float) -> None:
        rng = self._rngs["burst"]
        duration_ms = float(rng.uniform(300, 800))
        count = round(SAMPLE_RATE * duration_ms / 1000)
        snr = float(rng.uniform(*config.burst_snr_db))
        noise = np.convolve(rng.normal(0, 1, count), np.ones(6) / 6, mode="same")
        noise *= np.hanning(count)
        noise *= reference * 10 ** (-snr / 20) / max(_rms(noise), 1e-12)
        self._bursts.append(_Burst(noise))
        self._metadata.bursts.append(
            {"tick": self.tick, "duration_ms": round(duration_ms, 3), "snr_db": round(snr, 3)}
        )

    def _add_noise(self, output: np.ndarray) -> np.ndarray:
        config = self.config.noise
        if config is None or self._noise is None:
            return output
        reference = self.speech_rms
        time_s = self.tick * self._tick_s
        drift = config.drift_db * np.sin(2 * np.pi * time_s / 30 + self._drift_phase)
        gain = reference * 10 ** (-(config.snr_db + drift) / 20)
        start = self.tick * output.size % self._noise.size
        indexes = (start + np.arange(output.size)) % self._noise.size
        mixed = output + self._noise[indexes] * gain
        rate = config.bursts_per_minute * self._tick_s / 60
        if self._rngs["burst"].random() < rate:
            self._new_burst(config, reference)
        for burst in self._bursts:
            part = burst.samples[burst.position : burst.position + output.size]
            mixed[: part.size] += part
            burst.position += part.size
        self._bursts = [item for item in self._bursts if item.position < item.samples.size]
        return np.asarray(mixed)

    def _muffle(self, samples: np.ndarray) -> np.ndarray:
        config = self.config.muffle
        if config is None or self._muffle_filter is None or self._dry_delay is None:
            return samples
        filtered = self._muffle_filter.process(samples)
        dry = self._dry_delay.process(samples)
        rate = config.events_per_minute * self._tick_s / 60
        if self.tick > self._muffle_end and self._rngs["muffle"].random() < rate:
            ticks = max(1, round(config.duration_ms / (self._tick_s * 1000)))
            self._muffle_end = self.tick + ticks - 1
            self._metadata.muffles.append(
                {
                    "tick": self.tick,
                    "ticks": ticks,
                    "duration_ms": ticks * self._tick_s * 1000,
                }
            )
        target = 1.0 if self.tick <= self._muffle_end else 0.0
        ramp = max(1, round(SAMPLE_RATE * config.transition_ms / 1000))
        step = np.sign(target - self._muffle_level) / ramp
        levels = self._muffle_level + step * np.arange(1, samples.size + 1)
        levels = np.minimum(levels, target) if step > 0 else np.maximum(levels, target)
        self._muffle_level = float(levels[-1])
        return np.asarray(dry * (1 - levels) + filtered * levels)

    def _telephony(self, samples: np.ndarray) -> np.ndarray:
        limited = self._down.process(samples)
        narrow = np.rint(limited[::TELEPHONY_DECIMATION]).clip(-32768, 32767).astype(np.int16)
        decoded = mulaw_decode(mulaw_encode(narrow)).astype(np.float64)
        stuffed = np.zeros(samples.size)
        stuffed[::TELEPHONY_DECIMATION] = decoded
        return np.asarray(self._up.process(stuffed))

    def _drop_packets(self, samples: np.ndarray, config: ChannelConfig) -> np.ndarray:
        packet_samples = SAMPLE_RATE * PACKET_MS // 1000
        recover = min(1.0, PACKET_MS / config.drop_burst_ms)
        enter = 0.0
        if config.drop_rate > 0:
            enter = min(1.0, config.drop_rate * recover / (1 - config.drop_rate))
        rng = self._rngs["drops"]
        for start in range(0, samples.size, packet_samples):
            chance = recover if self._bad_channel else enter
            if rng.random() < chance:
                self._bad_channel = not self._bad_channel
            if self._bad_channel:
                samples[start : start + packet_samples] = 0
                self._metadata.dropped_packets.append(self._packet)
            self._packet += 1
        return samples

    def process(self, pcm: bytes) -> bytes:
        """Return one degraded tick."""
        clean = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        if clean.size != self.samples_per_tick:
            raise ValueError("effect input must contain exactly one tick")
        output = self._muffle(clean)
        scaled = self._add_noise(output / 32768) * 32768
        channel = self.config.channel
        if channel is not None:
            if channel.telephony:
                scaled = self._telephony(scaled)
            scaled = self._drop_packets(scaled, channel)
        self.tick += 1
        return np.rint(scaled).clip(-32768, 32767).astype("<i2").tobytes()

    def _latency_samples(self) -> int:
        latency = MUFFLE_FILTER_DELAY if self.config.muffle else 0
        channel = self.config.channel
        if channel is not None and channel.telephony:
            latency += 2 * TELEPHONY_FILTER_DELAY
        return latency

    def metadata(self) -> dict[str, Any]:
        """Describe the configuration and effects applied so far."""
        details = {
            "seed": self.seed,
            "config": asdict(self.config),
            "noise_source": self.noise_source,
            "noise_offset": self._noise_offset,
            "speech_rms": self.speech_rms,
            "latency_samples": self._latency_samples(),
            "bursts": list(self._metadata.bursts),
            "muffles": list(self._metadata.muffles),
            "dropped_packets": list(self._metadata.dropped_packets),
            "packet_ms": PACKET_MS,
        }
        stored: dict[str, Any] = json.loads(json.dumps(details, allow_nan=False))
        return stored
