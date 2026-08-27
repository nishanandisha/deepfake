"""Pulls a handful of real/fake sample clips out of LAV-DF into samples/.

LAV-DF ships as one ~25GB `LAV-DF.tar` with no per-file access on the Hub, so
downloading 12 clips the obvious way means downloading all of it. tar is a
sequential format, though, so this streams the archive and stops as soon as the
quota is filled -- typically a few hundred MB rather than 25GB.

Labels come from the archive's own metadata.json (`modify_video`/`modify_audio`;
real means both are false), matching src/preprocessing/lavdf.py. If metadata.json
turns up *after* some clips in the stream, those clips' byte offsets are recorded
and re-fetched with an HTTP range request rather than restarting the download.

Requires accepting the CC BY-NC 4.0 licence at
https://huggingface.co/datasets/ControlNet/LAV-DF and a read token
(--token, or HF_TOKEN).

Usage:
  python scripts/fetch_samples.py --per-class 6
"""

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import requests

REPO_ID = "ControlNet/LAV-DF"
TAR_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/LAV-DF.tar"
REPO = Path(__file__).resolve().parents[1]


def _labels_from_metadata(raw: bytes) -> dict:
    """{basename: 'real'|'fake'} from the archive's metadata.json."""
    data = json.loads(raw.decode("utf-8"))
    entries = data if isinstance(data, list) else [
        {**v, "file": v.get("file", k)} for k, v in data.items() if isinstance(v, dict)
    ]

    labels = {}
    for entry in entries:
        name = entry.get("file") or entry.get("filename") or entry.get("path")
        if not name:
            continue
        video = bool(entry.get("modify_video", entry.get("video_modified", False)))
        audio = bool(entry.get("modify_audio", entry.get("audio_modified", False)))
        labels[Path(name).name] = "fake" if (video or audio) else "real"
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch sample clips from LAV-DF.")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--dest", default=str(REPO / "samples"))
    parser.add_argument(
        "--max-bytes", type=int, default=8 * 1024**3,
        help="give up after streaming this much, so a bad layout can't pull 25GB",
    )
    args = parser.parse_args()

    if not args.token:
        sys.exit(
            "No HuggingFace token.\n"
            f"  1. Accept the licence at https://huggingface.co/datasets/{REPO_ID}\n"
            "  2. Create a read token at https://huggingface.co/settings/tokens\n"
            "  3. Re-run with --token hf_... (or set HF_TOKEN)"
        )

    dest = Path(args.dest)
    (dest / "real").mkdir(parents=True, exist_ok=True)
    (dest / "fake").mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {args.token}"

    response = session.get(TAR_URL, stream=True, timeout=60)
    if response.status_code == 401:
        sys.exit("401: token rejected, or the licence has not been accepted yet.")
    if response.status_code == 403:
        sys.exit(f"403: accept the licence at https://huggingface.co/datasets/{REPO_ID}")
    response.raise_for_status()

    labels: dict = {}
    counts = {"real": 0, "fake": 0}
    deferred: list = []          # clips seen before metadata.json arrived
    quota = args.per_class

    print(f"Streaming {TAR_URL} (stops early once {quota} real + {quota} fake are found)...")

    # stream mode: sequential reads only, no seeking back into the archive
    with tarfile.open(fileobj=response.raw, mode="r|") as tar:
        for member in tar:
            if response.raw.tell() > args.max_bytes:
                print("Hit --max-bytes before filling the quota; stopping.")
                break

            name = Path(member.name).name

            if name == "metadata.json" and member.isfile():
                labels = _labels_from_metadata(tar.extractfile(member).read())
                print(f"  metadata.json: {len(labels)} entries")
                # resolve anything encountered before the labels were known
                for pending_name, offset, size in deferred:
                    label = labels.get(pending_name)
                    if label and counts[label] < quota:
                        if _fetch_range(session, offset, size, dest / label / pending_name):
                            counts[label] += 1
                            print(f"  [{label}] {pending_name} (range-fetched)")
                deferred.clear()
                continue

            if not member.isfile() or not name.lower().endswith(".mp4"):
                continue

            if not labels:
                # offset_data is where this member's bytes start in the archive
                deferred.append((name, member.offset_data, member.size))
                if len(deferred) > 4000:
                    deferred.pop(0)
                continue

            label = labels.get(name)
            if not label or counts[label] >= quota:
                continue

            (dest / label / name).write_bytes(tar.extractfile(member).read())
            counts[label] += 1
            print(f"  [{label}] {name}")

            if counts["real"] >= quota and counts["fake"] >= quota:
                break

    response.close()
    print(f"\nDone: {counts['real']} real, {counts['fake']} fake -> {dest}")
    if counts["real"] < quota or counts["fake"] < quota:
        print("Quota not filled. Re-run with a larger --max-bytes.")


def _fetch_range(session, offset: int, size: int, out_path: Path) -> bool:
    """Re-fetches one archive member by byte range (used for clips streamed
    past before metadata.json revealed their label)."""
    headers = {"Range": f"bytes={offset}-{offset + size - 1}"}
    r = session.get(TAR_URL, headers=headers, stream=True, timeout=60)
    if r.status_code not in (200, 206):
        return False
    out_path.write_bytes(r.content)
    return True


if __name__ == "__main__":
    main()
