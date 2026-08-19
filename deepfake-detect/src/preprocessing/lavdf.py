"""LAV-DF (Localized Audio Visual DeepFake) indexer.

Replaces FakeAVCeleb as the primary dataset (see README "Data"): LAV-DF is
reachable via a HuggingFace click-through licence instead of a manual
approval queue, and -- critically -- it labels `modify_video` and
`modify_audio` separately, which is exactly the per-modality ground truth
the Stage 7 attribution-agreement metric scores against.

The upstream metadata schema is not publicly documented, so this module
introspects the real file rather than assuming field names: FIELD_ALIASES
lists the candidate spellings for each column we need, `inspect_metadata`
reports what was actually found, and indexing fails loudly with the
observed keys if a required field is missing. That is deliberate -- the
FakeAVCeleb indexer's guessed directory layout was a latent bug for
several stages precisely because it failed silently (returning 0 rows).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Candidate field names per logical column, tried in order. Extend these
# rather than rewriting the indexer if the release uses other spellings.
FIELD_ALIASES: Dict[str, List[str]] = {
    "file": ["file", "filename", "path", "video_path"],
    "original": ["original", "source", "src", "original_file"],
    "modify_video": ["modify_video", "video_modified", "fake_video", "visual_modified"],
    "modify_audio": ["modify_audio", "audio_modified", "fake_audio"],
    "split": ["split", "subset", "partition"],
    "n_fakes": ["n_fakes", "num_fakes"],
    "fake_periods": ["fake_periods", "fake_segments", "timestamps"],
}

MANIFEST_COLUMNS = [
    "sample_id",
    "video_path",
    "audio_path",
    "identity_id",
    "label",
    "manipulated_modality",
    "source_generator",
]


def _first_present(entry: dict, logical_name: str) -> Optional[str]:
    """Returns the actual key present in `entry` for a logical column."""
    for candidate in FIELD_ALIASES[logical_name]:
        if candidate in entry:
            return candidate
    return None


def load_metadata(metadata_path: str) -> List[dict]:
    """Reads metadata.json, tolerating either a top-level list or a dict
    keyed by filename."""
    raw = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        entries = []
        for key, value in raw.items():
            if isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("file", key)
                entries.append(entry)
        return entries
    raise ValueError(f"Unrecognised metadata.json top-level type: {type(raw).__name__}")


def inspect_metadata(metadata_path: str) -> Dict[str, object]:
    """Diagnostic: reports how many entries were found, the observed keys,
    and which logical columns this indexer managed to resolve. Run this
    first against a fresh download to confirm the schema before indexing."""
    entries = load_metadata(metadata_path)
    if not entries:
        return {"num_entries": 0, "observed_keys": [], "resolved": {}, "unresolved": []}

    sample = entries[0]
    resolved = {name: _first_present(sample, name) for name in FIELD_ALIASES}
    return {
        "num_entries": len(entries),
        "observed_keys": sorted(sample.keys()),
        "resolved": {k: v for k, v in resolved.items() if v is not None},
        "unresolved": [k for k, v in resolved.items() if v is None],
        "example_entry": sample,
    }


def _classify_modality(modify_video: bool, modify_audio: bool) -> tuple:
    """(label, manipulated_modality) from the two per-modality flags."""
    if modify_video and modify_audio:
        return "fake", "both"
    if modify_video:
        return "fake", "video"
    if modify_audio:
        return "fake", "audio"
    return "real", "none"


def _identity_from_entry(entry: dict, file_key: str, original_key: Optional[str]) -> str:
    """Groups each fake with the real clip it was derived from.

    In LAV-DF, a fake's `original` names its source clip, while a real clip
    has `original: null` because it *is* the source. So a real clip's
    identity is its own stem, and fakes inherit their source's stem --
    which puts a real clip and every fake derived from it in the same
    split, exactly what identity-disjoint splitting requires.

    Do NOT fall back to the parent directory here: LAV-DF stores every clip
    of a split in one flat folder (`test/`, `train/`, ...), so that would
    collapse thousands of distinct speakers into a single identity and
    silently defeat the whole leakage guard.
    """
    if original_key and entry.get(original_key):
        return Path(str(entry[original_key])).stem
    return Path(str(entry[file_key])).stem


def index_lavdf(root_dir: str, metadata_filename: str = "metadata.json") -> pd.DataFrame:
    """Walks a local LAV-DF root into the shared manifest schema.

    Audio is embedded in the mp4 (as with DFDC), so audio_path == video_path
    and src/preprocessing/audio.py decodes it directly.
    """
    root = Path(root_dir)
    metadata_path = root / metadata_filename
    if not metadata_path.exists():
        matches = list(root.rglob(metadata_filename))
        if not matches:
            raise FileNotFoundError(
                f"No {metadata_filename} under {root}. Point data.root_dir at the "
                "extracted LAV-DF directory (the one containing metadata.json)."
            )
        metadata_path = matches[0]

    entries = load_metadata(str(metadata_path))
    if not entries:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    sample = entries[0]
    file_key = _first_present(sample, "file")
    video_key = _first_present(sample, "modify_video")
    audio_key = _first_present(sample, "modify_audio")
    original_key = _first_present(sample, "original")

    missing = [
        name
        for name, key in [("file", file_key), ("modify_video", video_key),
                          ("modify_audio", audio_key)]
        if key is None
    ]
    if missing:
        raise KeyError(
            f"LAV-DF metadata at {metadata_path} is missing required field(s) {missing}. "
            f"Observed keys: {sorted(sample.keys())}. Add the real spelling to "
            "FIELD_ALIASES in src/preprocessing/lavdf.py."
        )

    dataset_root = metadata_path.parent
    rows = []
    for entry in entries:
        relative_path = str(entry[file_key])
        video_path = dataset_root / relative_path
        label, manipulated_modality = _classify_modality(
            bool(entry.get(video_key, False)), bool(entry.get(audio_key, False))
        )

        rows.append(
            {
                "sample_id": relative_path,
                "video_path": str(video_path),
                "audio_path": str(video_path),  # audio is muxed into the mp4
                "identity_id": _identity_from_entry(entry, file_key, original_key),
                "label": label,
                "manipulated_modality": manipulated_modality,
                "source_generator": entry.get(original_key) if original_key else None,
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def filter_to_existing_files(manifest: pd.DataFrame) -> pd.DataFrame:
    """Drops rows whose video file isn't on disk -- useful when only a
    partial download was fetched."""
    exists = manifest["video_path"].map(lambda p: Path(p).exists())
    return manifest[exists].reset_index(drop=True)
