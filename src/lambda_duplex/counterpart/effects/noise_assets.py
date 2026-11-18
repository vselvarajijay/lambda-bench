"""Download and load noise recordings with attribution metadata."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import soxr  # type: ignore[import-untyped]

from lambda_duplex.counterpart.effects.chain import SAMPLE_RATE, active_noise_bed
from lambda_duplex.counterpart.errors import NoiseAssetError

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MANIFEST_NAME = "manifest.json"
ATTRIBUTION_PREFIXES = ("LICENSE", "ANNOTATION", "README")
CORPUS_FOLDER = "_corpus"


@dataclass(frozen=True)
class NoiseBed:
    """Prepared recording and the attribution record stored with a run."""

    samples: np.ndarray
    record: dict[str, Any]


def _is_attribution(name: str) -> bool:
    return name.upper().startswith(ATTRIBUTION_PREFIXES)


def _target(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "musan":
        return None
    if len(path.parts) == 2 and _is_attribution(path.name):
        return PurePosixPath(CORPUS_FOLDER, path.name)
    if len(path.parts) < 4 or path.parts[1] != "noise":
        return None
    if path.suffix.lower() == ".wav" or _is_attribution(path.name):
        return PurePosixPath(*path.parts[2:])
    return None


def _extract(destination: Path, url: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    files: list[dict[str, Any]] = []
    attribution: dict[str, list[str]] = {}
    seen: set[PurePosixPath] = set()
    with (
        urllib.request.urlopen(url) as response,
        tarfile.open(fileobj=response, mode="r|gz") as tar,
    ):
        for member in tar:
            relative = _target(member.name)
            if relative is None or not member.isfile():
                continue
            if relative in seen:
                raise NoiseAssetError(f"archive repeats {relative}")
            seen.add(relative)
            source = tar.extractfile(member)
            if source is None:
                raise NoiseAssetError(f"cannot read {member.name}")
            output_path = destination / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with output_path.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if relative.suffix.lower() == ".wav":
                files.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": digest.hexdigest(),
                        "collection": relative.parts[0],
                    }
                )
            else:
                attribution.setdefault(relative.parts[0], []).append(relative.as_posix())
    return files, attribution


def download_musan_noise(destination: Path, url: str = MUSAN_URL) -> dict[str, Any]:
    """Install MUSAN noise recordings and attribution files atomically."""
    if destination.exists() and any(destination.iterdir()):
        raise NoiseAssetError(f"{destination} is not empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        files, attribution = _extract(temporary, url)
        if not files:
            raise NoiseAssetError("archive contained no MUSAN noise recordings")
        for item in files:
            sources = sorted(attribution.get(item["collection"], ()))
            if not sources:
                raise NoiseAssetError(f"no attribution file for {item['path']}")
            item["attribution_files"] = sources
        manifest = {
            "source": url,
            "corpus": "MUSAN noise",
            "corpus_files": sorted(attribution.get(CORPUS_FOLDER, ())),
            "files": sorted(files, key=lambda item: item["path"]),
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists():
            destination.rmdir()
        os.chmod(temporary, 0o755)
        os.replace(temporary, destination)
        return manifest
    except (tarfile.TarError, EOFError, zlib.error, OSError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise NoiseAssetError(f"cannot install noise assets: {error}") from error
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _prepare_entry(folder: Path, entry: dict[str, Any]) -> np.ndarray | None:
    path = folder / entry["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise NoiseAssetError(f"{entry['path']} does not match its recorded hash")
    audio, rate = sf.read(path, dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    resampled = np.asarray(soxr.resample(audio, rate, SAMPLE_RATE), dtype=np.float64)
    try:
        return active_noise_bed(resampled)
    except ValueError:
        return None


def load_noise_bed(folder: Path, seed: int) -> NoiseBed:
    """Verify recordings and return the first usable one starting from the seed."""
    try:
        manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
        files = manifest["files"]
        count = len(files)
        start = seed % count
        for step in range(count):
            entry = files[(start + step) % count]
            bed = _prepare_entry(folder, entry)
            if bed is None:
                continue
            record = {
                "type": "musan",
                "path": entry["path"],
                "sha256": entry["sha256"],
                "collection": entry["collection"],
                "attribution_files": entry["attribution_files"],
                "corpus_files": manifest.get("corpus_files", []),
                "skipped_recordings": step,
            }
            return NoiseBed(bed, record)
    except NoiseAssetError:
        raise
    except (OSError, KeyError, IndexError, ZeroDivisionError, ValueError, RuntimeError) as error:
        raise NoiseAssetError(f"cannot load noise from {folder}: {error}") from error
    raise NoiseAssetError(f"no usable noise recordings in {folder}")
