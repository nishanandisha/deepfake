"""Downloads LAV-DF from HuggingFace.

Requires a HuggingFace token with read access, because the dataset sits
behind a click-through licence (CC BY-NC 4.0). Accepting that licence is a
one-time manual step -- it cannot be automated, by design:

  1. Sign in at https://huggingface.co and open
     https://huggingface.co/datasets/ControlNet/LAV-DF
  2. Click "Agree and access repository"
  3. Create a read token at https://huggingface.co/settings/tokens
  4. Either `huggingface-cli login`, or pass --token / set HF_TOKEN

Usage:
  python scripts/download_data.py --output-dir data/raw/lavdf
  python scripts/download_data.py --output-dir data/raw/lavdf --metadata-only
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ID = "ControlNet/LAV-DF"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the LAV-DF dataset.")
    parser.add_argument("--output-dir", default="data/raw/lavdf")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Fetch only metadata.json -- fast way to verify access and inspect the schema "
             "before committing to the full ~25GB download.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed. Run: python -m pip install huggingface_hub"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.token:
        print(
            "No token supplied. If the download fails with 401/403, accept the licence at\n"
            f"  https://huggingface.co/datasets/{REPO_ID}\n"
            "then pass --token hf_... (or set HF_TOKEN).",
            file=sys.stderr,
        )

    if args.metadata_only:
        path = hf_hub_download(
            repo_id=REPO_ID, filename="metadata.json", repo_type="dataset",
            token=args.token, local_dir=str(output_dir),
        )
        print(f"Downloaded metadata to {path}")
        print("Next: python scripts/inspect_dataset.py --metadata", path)
        return

    print(f"Downloading {REPO_ID} to {output_dir} (~25GB; resumes if interrupted)...")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", token=args.token,
        local_dir=str(output_dir), max_workers=args.max_workers,
    )
    print(f"Done. Next: python scripts/inspect_dataset.py --root {output_dir}")


if __name__ == "__main__":
    main()
