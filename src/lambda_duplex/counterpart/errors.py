"""Errors that do not require optional audio packages."""

from __future__ import annotations


class NoiseAssetError(RuntimeError):
    """Noise assets are invalid or could not be installed."""
