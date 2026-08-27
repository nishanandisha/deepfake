"""Index the raw audio tree into a labelled manifest.

The dataset ships as two flat directories, `real/` and `fake/`, with the
provenance encoded in the filename prefix rather than in any metadata file:

    fake/el_0001_c_part_002.flac   -> ElevenLabs, source clip el_0001
    real/yt_0000_part_167.flac     -> YouTube,    source clip yt_0000

Both halves of that prefix matter and for different reasons:

* `source` (the platform) is the axis we hold out to measure generalisation
  to a TTS system the model has never heard.
* `group` (the originating clip) is the axis we must not leak across splits.
  Every file is a 2.5-13s chunk of a longer recording, so two chunks of
  `yt_0000` share a speaker, a microphone and a room. Splitting them randomly
  would let the model recognise the recording rather than the synthesis.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Filename prefix -> human-readable generator. Kept explicit rather than
# inferred so an unrecognised prefix fails loudly instead of silently
# becoming its own pseudo-platform.
SOURCE_NAMES = {
    "el": "elevenlabs",
    "hg": "kokoro",
    "hu": "hume",
    "lv": "luvvoice",
    "po": "polly",
    "sp": "speechify",
    "yt": "youtube",
}

LABELS = {"real": 0, "fake": 1}  # 0 = human, 1 = AI-generated

AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}


def _parse_stem(stem: str) -> tuple[str, str]:
    """`el_0001_c_part_002` -> ("elevenlabs", "el_0001")."""
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot parse source/group from filename stem: {stem!r}")

    prefix = parts[0]
    if prefix not in SOURCE_NAMES:
        raise ValueError(
            f"unknown source prefix {prefix!r} in {stem!r}. "
            f"Add it to SOURCE_NAMES if this is a new generator."
        )
    return SOURCE_NAMES[prefix], f"{parts[0]}_{parts[1]}"


def build_manifest(raw_dir: str, with_duration: bool = False) -> pd.DataFrame:
    """Scan `raw_dir/{real,fake}` into a DataFrame.

    Columns: path, label (0 human / 1 AI), label_name, source, group.
    `with_duration` opens every file with soundfile, which is slow enough
    (~1866 files) to be worth making optional.
    """
    root = Path(raw_dir)
    rows = []

    for label_name, label in LABELS.items():
        directory = root / label_name
        if not directory.is_dir():
            raise FileNotFoundError(
                f"expected {directory} to exist -- run scripts/download_data.py first"
            )

        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            source, group = _parse_stem(path.stem)
            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "label_name": label_name,
                    "source": source,
                    "group": group,
                }
            )

    if not rows:
        raise RuntimeError(f"no audio files found under {root}")

    manifest = pd.DataFrame(rows)

    if with_duration:
        import soundfile as sf

        def _seconds(path: str) -> Optional[float]:
            try:
                info = sf.info(path)
                return info.frames / float(info.samplerate)
            except Exception:  # noqa: BLE001 - a bad file should not kill indexing
                return None

        manifest["duration"] = manifest["path"].map(_seconds)

    return manifest


def manifest_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-source clip and group counts -- the table worth eyeballing before
    trusting any split."""
    summary = (
        manifest.groupby(["label_name", "source"])
        .agg(clips=("path", "size"), groups=("group", "nunique"))
        .reset_index()
        .sort_values(["label_name", "clips"], ascending=[True, False])
    )
    return summary
