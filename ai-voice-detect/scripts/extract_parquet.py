"""Materialise the corpus from the Hub's parquet shards into real/ and fake/.

Pulling 1,866 files individually gets rate-limited hard (the Hub degraded to
~1 file/sec and silently skipped an entire directory while still exiting 0).
The same data ships as two parquet shards, so it is two requests instead of
1,867. Each row carries the original filename in `audio.path`, which is what
carries the generator and source-clip identity that splits depend on.
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

LABEL_DIRS = {0: "real", 1: "fake"}


def extract(parquet_dir: str, out_dir: str) -> None:
    shards = sorted(Path(parquet_dir).rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {parquet_dir}")

    out = Path(out_dir)
    for name in LABEL_DIRS.values():
        (out / name).mkdir(parents=True, exist_ok=True)

    written = {name: 0 for name in LABEL_DIRS.values()}
    for shard in shards:
        table = pq.read_table(shard)
        audio = table.column("audio").to_pylist()
        labels = table.column("label").to_pylist()

        for entry, label in zip(audio, labels):
            directory = LABEL_DIRS[int(label)]
            path = out / directory / Path(entry["path"]).name
            if not path.exists():
                path.write_bytes(entry["bytes"])
            written[directory] += 1
        print(f"{shard.name}: {len(labels)} rows", flush=True)

    print("written:", written)
    for name in LABEL_DIRS.values():
        print(f"  {name}: {len(list((out / name).glob('*.flac')))} files on disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="data/parquet")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    extract(args.parquet, args.out)
