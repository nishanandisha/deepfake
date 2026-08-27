"""One-time acoustic preprocessing cache.

Without this, every epoch re-runs feature extraction on every audio clip.
With `pitch_tracker="pyin"` that costs ~1-3s per clip, so a single epoch
over a few thousand clips takes hours and a real training run becomes
impossible -- the cost is paid once per epoch when it should be paid once,
ever.

Cached artefact per clip:
  <cache_dir>/acoustic/<hash>.npz   named feature matrix, float32 [S, D]

The cache key includes every parameter that changes the output (sample
rate, framing, n_mfcc, pitch tracker), so changing a config value produces
a different key rather than silently serving stale tensors.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.models.acoustic.features import extract_acoustic_features
from src.preprocessing.audio import load_audio


def _cache_key(source_path: str, params: dict) -> str:
    payload = json.dumps({"source": str(source_path), **params}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class PreprocessingCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            (self.cache_dir / "acoustic").mkdir(parents=True, exist_ok=True)

    def acoustic_path(self, audio_path: str, params: dict) -> Path:
        return self.cache_dir / "acoustic" / f"{_cache_key(audio_path, params)}.npz"

    def get_acoustic(
        self,
        audio_path: str,
        sample_rate: int,
        frame_ms: float,
        hop_ms: float,
        n_mfcc: int,
        pitch_tracker: str = "yin",
    ) -> Tuple[np.ndarray, list]:
        """Named acoustic features [S, D] float32 plus feature_names."""
        params = {
            "kind": "acoustic",
            "sample_rate": sample_rate,
            "frame_ms": frame_ms,
            "hop_ms": hop_ms,
            "n_mfcc": n_mfcc,
            "pitch_tracker": pitch_tracker,
        }

        if self.enabled:
            path = self.acoustic_path(audio_path, params)
            if path.exists():
                try:
                    with np.load(path, allow_pickle=False) as data:
                        return data["features"], [str(n) for n in data["feature_names"]]
                except (OSError, ValueError, KeyError):
                    # A truncated file from an interrupted run should be
                    # regenerated, not crash the whole training job.
                    path.unlink(missing_ok=True)

        signal = load_audio(audio_path, sample_rate=sample_rate)
        features, feature_names = extract_acoustic_features(
            signal, sample_rate=sample_rate, frame_ms=frame_ms, hop_ms=hop_ms, n_mfcc=n_mfcc,
            pitch_tracker=pitch_tracker,
        )

        if self.enabled:
            self._atomic_savez(
                self.acoustic_path(audio_path, params),
                features=features,
                feature_names=np.array(feature_names, dtype=np.str_),
            )
        return features, feature_names

    # -- internals --------------------------------------------------------

    @staticmethod
    def _atomic_savez(path: Path, **arrays) -> None:
        """Write to a temp file then rename, so an interrupted run can't
        leave a half-written .npz that later loads as corrupt."""
        temp_path = path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(temp_path, **arrays)
            temp_path.replace(path)
        except OSError:
            temp_path.unlink(missing_ok=True)
            raise

    def stats(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        acoustic_files = list((self.cache_dir / "acoustic").glob("*.npz"))
        total_bytes = sum(f.stat().st_size for f in acoustic_files)
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "acoustic_entries": len(acoustic_files),
            "total_mb": round(total_bytes / (1024 * 1024), 1),
        }


def warm_cache(
    manifest,
    cache: PreprocessingCache,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
    n_mfcc: int,
    pitch_tracker: str = "yin",
    logger=None,
    log_every: int = 25,
    fail_fast_after: int = 20,
) -> dict:
    """Pre-populates the cache for every row in a manifest. Run this once
    (see scripts/warm_cache.py) before training so epochs are I/O-bound
    instead of CPU-bound.

    Individual failures are tolerated -- one unreadable clip shouldn't
    abandon hours of work -- but a *systematic* failure must not be
    swallowed. Tolerating every error previously hid a missing ffmpeg
    backend that failed 100% of audio decodes, leaving an empty cache that
    would only have surfaced as a broken training run. So if the first
    `fail_fast_after` clips all fail, stop and raise.
    """
    failures = []
    for i, row in enumerate(manifest.itertuples(index=False)):
        try:
            cache.get_acoustic(
                row.audio_path, sample_rate, frame_ms, hop_ms, n_mfcc,
                pitch_tracker=pitch_tracker,
            )
        except Exception as error:  # noqa: BLE001 - report and continue
            failures.append({"sample_id": getattr(row, "sample_id", ""), "error": str(error)})

        if len(failures) >= fail_fast_after and len(failures) == i + 1:
            raise RuntimeError(
                f"Every one of the first {len(failures)} clips failed to preprocess -- "
                f"this is an environment problem, not bad data. First error: "
                f"{failures[0]['error']}"
            )

        if logger and (i + 1) % log_every == 0:
            logger.info(f"Warmed {i + 1}/{len(manifest)} clips ({len(failures)} failures)")

    if failures and logger:
        rate = len(failures) / max(len(manifest), 1)
        logger.info(f"WARNING: {len(failures)}/{len(manifest)} clips failed ({rate:.1%})")

    return {"num_processed": len(manifest), "failures": failures, **cache.stats()}


def get_shared_cache(
    cache_dir: Optional[str], enabled: bool = True
) -> Optional[PreprocessingCache]:
    if not cache_dir or not enabled:
        return None
    return PreprocessingCache(cache_dir, enabled=True)
