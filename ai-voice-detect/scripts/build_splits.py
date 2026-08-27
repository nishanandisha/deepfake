"""Index the corpus and split it without leaking a recording or a generator.

Writes data/splits/manifest.csv. Prints the tables worth checking before any
training happens: per-source counts, and the split table with its class
balance. Both assertions run every time -- a silent leak here would make
every number downstream meaningless.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.manifest import build_manifest, manifest_summary
from src.preprocessing.splits import (
    DEFAULT_HELD_OUT_SOURCES,
    assert_no_group_leakage,
    assert_sources_held_out,
    make_splits,
    split_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/splits/manifest.csv")
    parser.add_argument("--held-out", nargs="*", default=list(DEFAULT_HELD_OUT_SOURCES))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_manifest(args.raw, with_duration=True)
    print(f"indexed {len(manifest)} clips\n")
    print(manifest_summary(manifest).to_string(index=False))

    total_hours = manifest["duration"].sum() / 3600
    print(f"\ntotal audio: {total_hours:.2f} h  "
          f"(mean clip {manifest['duration'].mean():.2f}s)")

    splits = make_splits(manifest, held_out_sources=args.held_out, seed=args.seed)
    assert_no_group_leakage(splits)
    assert_sources_held_out(splits, args.held_out)

    print(f"\nheld-out generators (test only): {args.held_out}\n")
    print(split_stats(splits).to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
