"""Run the frozen frontend once per clip and cache the result.

This is the expensive step and the only one that touches WavLM. Everything
downstream reads .npy files, which is why training runs take minutes and can
be repeated freely.

Training clips are encoded three times -- clean plus two augmented variants --
because a frozen frontend cannot be augmented on the fly: by training time
the waveform is gone. Val/calibration/test are encoded clean only, so the
evaluation measures the model rather than the augmentation.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.augment import augment
from src.preprocessing.embeddings import EmbeddingCache, WavLMFrontend, load_audio

AUG_VARIANTS = ["aug1", "aug2"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--augment-splits", nargs="*", default=["train"])
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    frontend = WavLMFrontend(layer=args.layer)
    cache = EmbeddingCache(args.cache_dir, args.layer)
    print(f"device={frontend.device}  layer={args.layer}  clips={len(manifest)}", flush=True)

    started = time.time()
    done = skipped = 0
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        variants = ["clean"]
        if row.split in args.augment_splits:
            variants += AUG_VARIANTS

        signal = None
        for variant in variants:
            if cache.get(row.path, variant) is not None:
                skipped += 1
                continue
            if signal is None:
                signal = load_audio(row.path)
            wave = signal if variant == "clean" else augment(signal, seed=abs(hash((row.path, variant))) % (2**31))
            cache.put(row.path, frontend.embed(wave), variant)
            done += 1

        if i % 200 == 0:
            rate = (time.time() - started) / max(done, 1)
            print(f"  {i}/{len(manifest)} clips  {done} encoded  "
                  f"{rate*1000:.0f} ms/encode", flush=True)

    elapsed = time.time() - started
    size_mb = sum(p.stat().st_size for p in cache.cache_dir.glob("*.npy")) / 1e6
    print(f"\nencoded {done}, reused {skipped}, in {elapsed:.0f}s")
    print(f"cache: {cache.cache_dir}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
