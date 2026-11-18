"""Build judge transcripts that hide which model produced a run."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from lambda_duplex.llm import LLMRequest
from lambda_duplex.trajectory.store import StoredRun, TrajectoryError

EVALUATED = "A"
OTHER = "B"
REDACTED = "[REDACTED]"
MAX_SETTING_TERM = 64
PAYLOAD_KEYS = {"items", "transcript"}
TURN_KEYS = {"turn", "speaker", "text", "cut_off"}


class BlindingError(RuntimeError):
    """A judge request could reveal the identity of the evaluated model."""


@dataclass(frozen=True)
class Turn:
    """One blinded turn of the conversation."""

    turn: int
    speaker: str
    text: str
    cut_off: bool = False

    def public(self) -> dict[str, Any]:
        """Return the turn as shown to the judge."""
        row = asdict(self)
        if not self.cut_off:
            del row["cut_off"]
        return row


@dataclass(frozen=True)
class BlindedTranscript:
    """Turns ready for a judge, with how they were obtained."""

    turns: tuple[Turn, ...]
    source: str
    redactions: int
    terms: tuple[str, ...]
    delivery_known: bool = True
    input_sha256: str = ""

    @property
    def sha256(self) -> str:
        rows = [turn.public() for turn in self.turns]
        body = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()


def _usable(term: str) -> bool:
    if not any(character.isalpha() for character in term):
        return False
    return len(term) >= 3 or (len(term) == 2 and any(c.isdigit() for c in term))


def _looks_like_identifier(value: str) -> bool:
    """Separate values like ``tuned-v3`` from ordinary words like ``main`` or ``text``."""
    return len(value) <= MAX_SETTING_TERM and re.search(r"[\d_./:-]", value) is not None


def _reference_terms(reference: str) -> set[str]:
    """Return the reference, its model name, and any identifier-like parts."""
    model_name = re.split(r"[:@]", reference.rstrip("/").rsplit("/", 1)[-1])[0]
    parts = {*re.split(r"[:/@]", reference), *re.split(r"[:@]", reference)}
    return {reference, model_name, *filter(_looks_like_identifier, parts)}


def identity_terms(run: StoredRun, extra: Iterable[str] = ()) -> tuple[str, ...]:
    """Return strings that would identify the evaluated model."""
    model: Mapping[str, Any] = run.manifest["model"]
    reference = str(model["reference"])
    terms = {str(model["name"]), *_reference_terms(reference)}
    candidates = [model["version"], *dict(model["settings"]).values()]
    terms.update(
        value for value in candidates if isinstance(value, str) and _looks_like_identifier(value)
    )
    for event in run.trajectory.events:
        if event.source == "model" and event.type == "started":
            terms.add(str(event.data.get("session_id", "")))
    usable = {term.strip() for term in terms if _usable(term.strip())}
    usable.update(term.strip() for term in extra if term.strip())
    return tuple(sorted(usable, key=lambda term: (-len(term), term)))


def _term_pattern(terms: Sequence[str]) -> re.Pattern[str] | None:
    parts = []
    for term in terms:
        pieces = [re.escape(piece) for piece in re.split(r"[-_\s]+", term) if piece]
        if pieces:
            parts.append(r"[-_\s]?".join(pieces))
    if not parts:
        return None
    return re.compile(rf"(?<!\w)(?:{'|'.join(parts)})(?!\w)", re.IGNORECASE)


def redact(text: str, terms: Sequence[str]) -> tuple[str, int]:
    """Replace identity terms, returning the text and replacement count."""
    pattern = _term_pattern(terms)
    return (text, 0) if pattern is None else pattern.subn(REDACTED, text)


def _transcript_rows(run: StoredRun) -> tuple[list[tuple[float, str]], str] | None:
    path = run.path / "transcript.jsonl"
    if not path.is_file():
        return None
    rows: list[tuple[float, str]] = []
    sources = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row["text"], str):
                raise TypeError("text must be a string")
            rows.append((float(row.get("start_ms", row.get("time_ms", 0))), row["text"]))
            sources.add(str(row.get("source", "unknown")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise TrajectoryError(f"invalid transcript in {path}: {error}") from error
    return rows, "transcript:" + "+".join(sorted(sources))


def _model_rows(run: StoredRun) -> tuple[list[tuple[float, str]], str, bool]:
    """Return model speech, preferring text derived from the delivered audio.

    Adapter text can run ahead of playback, so in an audio run it does not show what
    was delivered. The final value says whether delivery is known.
    """
    stored = _transcript_rows(run)
    if stored is not None and "injected" not in stored[1]:
        return stored[0], stored[1], True
    events = run.trajectory.events
    tick_ms = run.trajectory.tick_ms
    heard = [e for e in events if e.source == "counterpart" and e.type == "model_transcript"]
    if heard:
        rows = [(e.data["start_tick"] * tick_ms, e.data["text"]) for e in heard]
        return rows, "asr_events", True
    truncated = any(e.source == "model" and e.type == "truncated" for e in events)
    known = run.manifest.get("audio") is None and not truncated
    if stored is not None:
        return stored[0], stored[1], known
    spoke = [e for e in events if e.source == "model" and e.type == "spoke"]
    return [(e.time_ms, e.data["text"]) for e in spoke], "adapter_text", known


def _conversation_rows(run: StoredRun) -> tuple[list[tuple[float, int, str, str, bool]], str, bool]:
    """Return unredacted turns in order, the model text source and delivery status."""
    events = run.trajectory.events
    finished_starts = {
        event.data["start_tick"]
        for event in events
        if event.source == "counterpart"
        and event.type == "segment_delivered"
        and not event.data.get("interrupted")
    }
    rows: list[tuple[float, int, str, str, bool]] = [
        (
            event.time_ms,
            0,
            OTHER,
            event.data["text"] if event.tick in finished_starts else "",
            event.tick not in finished_starts,
        )
        for event in events
        if event.source == "counterpart" and event.type == "spoke"
    ]
    model_rows, source, delivery_known = _model_rows(run)
    rows.extend((time_ms, 1, EVALUATED, text, False) for time_ms, text in model_rows)
    rows.sort(key=lambda row: row[:2])
    return rows, source, delivery_known


def _fingerprint(rows: Sequence[tuple[float, int, str, str, bool]]) -> str:
    body = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def transcript_fingerprint(run: StoredRun) -> str:
    """Identify the conversation a judge would read, before any redaction."""
    return _fingerprint(_conversation_rows(run)[0])


def blinded_transcript(run: StoredRun, extra_terms: Iterable[str] = ()) -> BlindedTranscript:
    """Return the conversation with neutral speakers and identity terms removed.

    Counterpart words are shown only for turns that finished playing. An interrupted
    turn, or one still playing when the run ended, may hold words nobody heard, and the
    stored events do not say which words were played.
    """
    rows, source, delivery_known = _conversation_rows(run)
    terms = identity_terms(run, extra_terms)
    turns: list[Turn] = []
    total = 0
    for _, _, speaker, raw, cut_off in rows:
        text, count = redact(raw.strip(), terms)
        total += count
        if text or cut_off:
            turns.append(Turn(len(turns), speaker, text, cut_off))
    fingerprint = _fingerprint(rows)
    return BlindedTranscript(tuple(turns), source, total, terms, delivery_known, fingerprint)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def assert_blinded(
    request: LLMRequest, terms: Sequence[str], run: StoredRun, fixed: Sequence[str] = ()
) -> None:
    """Refuse a request whose run-specific or rubric text could identify the model.

    ``fixed`` lists prompt text that is the same for every run. It and the JSON field
    names cannot reveal the model, so a model called ``json`` is not a leak.
    """
    try:
        payload = json.loads(request.messages[1]["content"])
        keys = set(payload)
        turn_keys = {key for row in payload["transcript"] for key in row}
        spoken = [str(row.get("text", "")) for row in payload["transcript"]]
        system = request.messages[0]["content"]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise BlindingError(f"judge input has an unexpected shape: {error}") from error
    if keys != PAYLOAD_KEYS or turn_keys - TURN_KEYS:
        raise BlindingError("judge input carries fields outside the blinded format")
    for text in fixed:
        system = system.replace(text, "")
    params = list(_strings(dict(request.params)))
    content = "\n".join([system, *_strings(payload["items"]), *spoken, *params])
    pattern = _term_pattern(terms)
    found = None if pattern is None else pattern.search(content)
    if found is not None:
        raise BlindingError(f"judge input contains the identity term {found.group()!r}")
    if str(run.path.resolve()) in content:
        raise BlindingError("judge input contains the run path")
