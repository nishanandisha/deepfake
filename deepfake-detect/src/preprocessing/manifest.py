"""Dataset indexers: walk raw FakeAVCeleb / DFDC directories into manifest
dataframes with columns: sample_id, video_path, audio_path, identity_id,
label, manipulated_modality, source_generator.

FakeAVCeleb ships as four top-level category folders, each real/fake
combination of (video, audio), with identity subfolders beneath a
demographic (ethnicity/gender) folder, and audio embedded in the mp4 for
most samples:

    <root>/RealVideo-RealAudio/<ethnicity>/<gender>/<identity_id>/*.mp4
    <root>/FakeVideo-FakeAudio/<ethnicity>/<gender>/<identity_id>/<method>/*.mp4
    <root>/FakeVideo-RealAudio/...
    <root>/RealVideo-FakeAudio/...

`identity_dir_depth` controls how many directory levels up from the media
file the identity folder sits -- verify this against the actual downloaded
dataset (this indexer has not been run against real FakeAVCeleb data yet)
and adjust if the release layout differs.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
AUDIO_EXTENSIONS = {".wav", ".flac"}

# Maps FakeAVCeleb's four category folder names to (label, manipulated_modality).
_CATEGORY_MAP = {
    "RealVideo-RealAudio": ("real", "none"),
    "FakeVideo-FakeAudio": ("fake", "both"),
    "FakeVideo-RealAudio": ("fake", "video"),
    "RealVideo-FakeAudio": ("fake", "audio"),
}


def _find_sibling_audio(video_path: Path) -> Optional[Path]:
    for ext in AUDIO_EXTENSIONS:
        candidate = video_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _identity_from_path(video_path: Path, category_dir: Path, identity_dir_depth: int) -> str:
    """Identity id = the directory `identity_dir_depth` levels above the file,
    relative to the category directory. Falls back to the immediate parent
    directory name if the path is shallower than expected."""
    rel_parts = video_path.relative_to(category_dir).parts
    idx = len(rel_parts) - 1 - identity_dir_depth
    if 0 <= idx < len(rel_parts) - 1:
        return rel_parts[idx]
    return video_path.parent.name


def index_fakeavceleb(root_dir: str, identity_dir_depth: int = 1) -> pd.DataFrame:
    """Walk a local FakeAVCeleb root directory into a manifest dataframe.

    identity_dir_depth: how many directory levels above the media file the
    identity folder lives (1 = immediate parent, matching
    RealVideo-RealAudio/<eth>/<gender>/<identity>/*.mp4; increase to 2 for
    layouts with an extra <method> folder between identity and file, as in
    the Fake*/Fake* categories above).
    """
    root = Path(root_dir)
    rows = []

    for category_name, (label, manipulated_modality) in _CATEGORY_MAP.items():
        category_dir = root / category_name
        if not category_dir.is_dir():
            continue

        depth = identity_dir_depth if manipulated_modality == "none" else identity_dir_depth
        for video_path in sorted(category_dir.rglob("*")):
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            audio_path = _find_sibling_audio(video_path)
            identity_id = _identity_from_path(video_path, category_dir, depth)
            source_generator = video_path.parent.name if manipulated_modality != "none" else None

            rows.append(
                {
                    "sample_id": f"{category_name}/{video_path.relative_to(category_dir)}",
                    "video_path": str(video_path),
                    "audio_path": str(audio_path) if audio_path else str(video_path),
                    "identity_id": identity_id,
                    "label": label,
                    "manipulated_modality": manipulated_modality,
                    "source_generator": source_generator,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "video_path",
            "audio_path",
            "identity_id",
            "label",
            "manipulated_modality",
            "source_generator",
        ],
    )


def index_dfdc(root_dir: str) -> pd.DataFrame:
    """Walk a local DFDC root (one or more dfdc_train_part_N folders, each
    with a metadata.json) into the same manifest schema. DFDC has no
    identity labels, so identity_id is set to the sample_id itself --
    this dataset is never split or trained on (see Stage 1 of the build
    plan), only used for held-out cross-dataset evaluation, so identity
    grouping is irrelevant here.
    """
    import json

    root = Path(root_dir)
    rows = []

    for metadata_path in sorted(root.rglob("metadata.json")):
        part_dir = metadata_path.parent
        with open(metadata_path) as f:
            metadata = json.load(f)

        for filename, info in metadata.items():
            video_path = part_dir / filename
            if not video_path.exists():
                continue

            label = "fake" if info.get("label", "").upper() == "FAKE" else "real"
            sample_id = f"{part_dir.name}/{filename}"

            rows.append(
                {
                    "sample_id": sample_id,
                    "video_path": str(video_path),
                    "audio_path": str(video_path),  # DFDC audio is embedded in the mp4
                    "identity_id": sample_id,
                    "label": label,
                    "manipulated_modality": "both" if label == "fake" else "none",
                    "source_generator": info.get("original"),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "video_path",
            "audio_path",
            "identity_id",
            "label",
            "manipulated_modality",
            "source_generator",
        ],
    )
