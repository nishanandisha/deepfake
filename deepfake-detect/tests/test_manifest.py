from pathlib import Path

from src.preprocessing.manifest import index_dfdc, index_fakeavceleb


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_index_fakeavceleb_builds_expected_manifest(tmp_path: Path):
    root = tmp_path / "FakeAVCeleb_v1.2"

    _touch(root / "RealVideo-RealAudio" / "asian" / "men" / "id00001" / "clip1.mp4")
    _touch(root / "RealVideo-RealAudio" / "asian" / "men" / "id00001" / "clip2.mp4")
    _touch(root / "FakeVideo-FakeAudio" / "asian" / "men" / "id00001" / "fsgan" / "fake1.mp4")
    _touch(root / "FakeVideo-RealAudio" / "african" / "women" / "id00002" / "wav2lip" / "fake2.mp4")
    _touch(root / "RealVideo-FakeAudio" / "african" / "women" / "id00002" / "rtvc" / "fake3.mp4")

    manifest = index_fakeavceleb(str(root))

    assert len(manifest) == 5
    assert set(manifest.columns) == {
        "sample_id",
        "video_path",
        "audio_path",
        "identity_id",
        "label",
        "manipulated_modality",
        "source_generator",
    }
    assert (manifest["label"] == "real").sum() == 2
    assert (manifest["label"] == "fake").sum() == 3
    assert set(manifest["manipulated_modality"]) == {"none", "both", "video", "audio"}


def test_index_fakeavceleb_empty_dir_returns_empty_manifest(tmp_path: Path):
    manifest = index_fakeavceleb(str(tmp_path / "does_not_exist"))
    assert manifest.empty


def test_index_dfdc_reads_metadata(tmp_path: Path):
    import json

    part_dir = tmp_path / "dfdc_train_part_0"
    part_dir.mkdir(parents=True)
    (part_dir / "real1.mp4").touch()
    (part_dir / "fake1.mp4").touch()

    metadata = {
        "real1.mp4": {"label": "REAL", "split": "train", "original": None},
        "fake1.mp4": {"label": "FAKE", "split": "train", "original": "real1.mp4"},
    }
    (part_dir / "metadata.json").write_text(json.dumps(metadata))

    manifest = index_dfdc(str(tmp_path))

    assert len(manifest) == 2
    assert set(manifest["label"]) == {"real", "fake"}
    fake_row = manifest[manifest["label"] == "fake"].iloc[0]
    assert fake_row["source_generator"] == "real1.mp4"
