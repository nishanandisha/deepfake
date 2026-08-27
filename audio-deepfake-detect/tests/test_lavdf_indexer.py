import json
from pathlib import Path

import pytest

from src.preprocessing.lavdf import index_lavdf, inspect_metadata


def _write_metadata(tmp_path: Path, entries, filename="metadata.json") -> Path:
    (tmp_path / filename).write_text(json.dumps(entries), encoding="utf-8")
    for entry in entries if isinstance(entries, list) else entries.values():
        file_field = entry.get("file") or entry.get("filename")
        if file_field:
            video_path = tmp_path / file_field
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.touch()
    return tmp_path / filename


def test_index_maps_modality_flags_to_labels(tmp_path: Path):
    entries = [
        {"file": "id0/real.mp4", "modify_video": False, "modify_audio": False,
         "original": "id0/src.mp4"},
        {"file": "id0/fv.mp4", "modify_video": True, "modify_audio": False,
         "original": "id0/src.mp4"},
        {"file": "id1/fa.mp4", "modify_video": False, "modify_audio": True,
         "original": "id1/src.mp4"},
        {"file": "id1/both.mp4", "modify_video": True, "modify_audio": True,
         "original": "id1/src.mp4"},
    ]
    _write_metadata(tmp_path, entries)

    manifest = index_lavdf(str(tmp_path))

    by_id = manifest.set_index("sample_id")
    assert by_id.loc["id0/real.mp4", "label"] == "real"
    assert by_id.loc["id0/real.mp4", "manipulated_modality"] == "none"
    assert by_id.loc["id0/fv.mp4", "manipulated_modality"] == "video"
    assert by_id.loc["id1/fa.mp4", "manipulated_modality"] == "audio"
    assert by_id.loc["id1/both.mp4", "manipulated_modality"] == "both"
    assert (by_id.loc[["id0/fv.mp4", "id1/fa.mp4", "id1/both.mp4"], "label"] == "fake").all()


def test_identity_grouped_by_source_video(tmp_path: Path):
    """Clips derived from the same source share a speaker; grouping them
    keeps identity-disjoint splitting meaningful."""
    entries = [
        {"file": "a/1.mp4", "modify_video": True, "modify_audio": False, "original": "spk7.mp4"},
        {"file": "b/2.mp4", "modify_video": False, "modify_audio": True, "original": "spk7.mp4"},
        {"file": "c/3.mp4", "modify_video": True, "modify_audio": True, "original": "spk9.mp4"},
    ]
    _write_metadata(tmp_path, entries)

    manifest = index_lavdf(str(tmp_path))

    identities = manifest.set_index("sample_id")["identity_id"]
    assert identities["a/1.mp4"] == identities["b/2.mp4"] == "spk7"
    assert identities["c/3.mp4"] == "spk9"


def test_real_clips_get_distinct_identities(tmp_path: Path):
    """Regression: real clips carry `original: null`, and LAV-DF stores a
    whole split in one flat folder. Falling back to the parent directory
    would give every real clip the identity "test", collapsing thousands of
    speakers into one and silently defeating identity-disjoint splitting.
    """
    entries = [
        {"file": "test/000001.mp4", "modify_video": False, "modify_audio": False,
         "original": None},
        {"file": "test/000002.mp4", "modify_video": False, "modify_audio": False,
         "original": None},
        {"file": "test/000003.mp4", "modify_video": False, "modify_audio": False,
         "original": None},
    ]
    _write_metadata(tmp_path, entries)

    manifest = index_lavdf(str(tmp_path))

    assert manifest["identity_id"].nunique() == 3
    assert set(manifest["identity_id"]) == {"000001", "000002", "000003"}


def test_fake_groups_with_the_real_clip_it_came_from(tmp_path: Path):
    """A real clip and every fake derived from it must land in the same
    split, or the model could train on a face it is later tested on."""
    entries = [
        {"file": "test/000001.mp4", "modify_video": False, "modify_audio": False,
         "original": None},
        {"file": "test/090000.mp4", "modify_video": True, "modify_audio": False,
         "original": "test/000001.mp4"},
    ]
    _write_metadata(tmp_path, entries)

    manifest = index_lavdf(str(tmp_path))
    assert manifest["identity_id"].nunique() == 1
    assert set(manifest["identity_id"]) == {"000001"}


def test_audio_path_points_at_muxed_video(tmp_path: Path):
    entries = [{"file": "x/1.mp4", "modify_video": True, "modify_audio": False}]
    _write_metadata(tmp_path, entries)

    manifest = index_lavdf(str(tmp_path))
    assert manifest.iloc[0]["audio_path"] == manifest.iloc[0]["video_path"]


def test_dict_keyed_metadata_is_supported(tmp_path: Path):
    entries = {
        "id0/a.mp4": {"modify_video": True, "modify_audio": False},
        "id0/b.mp4": {"modify_video": False, "modify_audio": False},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(entries), encoding="utf-8")
    for name in entries:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    manifest = index_lavdf(str(tmp_path))
    assert len(manifest) == 2


def test_alternate_field_spellings_resolve(tmp_path: Path):
    entries = [{"filename": "y/1.mp4", "video_modified": True, "audio_modified": False}]
    _write_metadata(tmp_path, entries)

    manifest = index_lavdf(str(tmp_path))
    assert manifest.iloc[0]["manipulated_modality"] == "video"


def test_missing_required_field_fails_loudly(tmp_path: Path):
    """A silent failure here would produce an empty manifest that only
    surfaces hours later in training -- so it must raise with diagnostics."""
    entries = [{"file": "z/1.mp4", "some_unexpected_field": True}]
    (tmp_path / "metadata.json").write_text(json.dumps(entries), encoding="utf-8")

    with pytest.raises(KeyError, match="modify_video"):
        index_lavdf(str(tmp_path))


def test_missing_metadata_file_raises_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        index_lavdf(str(tmp_path))


def test_inspect_metadata_reports_schema(tmp_path: Path):
    entries = [{"file": "a.mp4", "modify_video": True, "modify_audio": False, "extra": 1}]
    metadata_path = _write_metadata(tmp_path, entries)

    report = inspect_metadata(str(metadata_path))

    assert report["num_entries"] == 1
    assert "extra" in report["observed_keys"]
    assert report["resolved"]["modify_video"] == "modify_video"
    assert "split" in report["unresolved"]  # not present in this fixture
