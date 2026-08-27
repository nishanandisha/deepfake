"""Measure which WavLM layer carries the synthesis evidence.

Anti-spoofing cues are widely reported to peak in the middle of a
self-supervised stack: early layers are still close to the waveform, and the
final layers have been trained toward phonetic/semantic content that is by
design invariant to *how* the audio was produced. Which middle layer wins is
checkpoint- and task-specific, so measure it instead of assuming.

Cheap by construction: one forward pass returns every layer, each layer is
reduced to a mean+std vector, and the probe is logistic regression. Fitting
the real head per layer would cost 12x for a decision this makes just as
well. Selection is on val; test is never touched here.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.embeddings import NUM_LAYERS, WavLMFrontend, load_audio
from src.training.metrics import compute_eer


def pooled_stats(states) -> np.ndarray:
    """[num_layers, 2*768] -- mean and std over time for every layer."""
    out = []
    for state in states:
        array = state.numpy()
        out.append(np.concatenate([array.mean(axis=0), array.std(axis=0)]))
    return np.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--per-split", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/layer_sweep.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    rng = np.random.default_rng(args.seed)

    subsets = {}
    for split in ("train", "val"):
        frame = manifest[manifest["split"] == split]
        if len(frame) > args.per_split:
            frame = frame.iloc[rng.choice(len(frame), args.per_split, replace=False)]
        subsets[split] = frame
        print(f"{split}: {len(frame)} clips", flush=True)

    frontend = WavLMFrontend()
    print(f"device={frontend.device}", flush=True)

    features, labels = {}, {}
    for split, frame in subsets.items():
        stack = []
        for i, path in enumerate(frame["path"]):
            states = frontend.hidden_states(load_audio(path), num_layers=NUM_LAYERS)
            stack.append(pooled_stats(states))
            if (i + 1) % 100 == 0:
                print(f"  {split} {i+1}/{len(frame)}", flush=True)
        features[split] = np.stack(stack)  # [N, layers, 1536]
        labels[split] = frame["label"].to_numpy()

    rows = []
    for layer in range(NUM_LAYERS):
        scaler = StandardScaler().fit(features["train"][:, layer])
        probe = LogisticRegression(max_iter=2000, C=1.0)
        probe.fit(scaler.transform(features["train"][:, layer]), labels["train"])
        scores = probe.predict_proba(scaler.transform(features["val"][:, layer]))[:, 1]
        eer = compute_eer(labels["val"], scores)
        rows.append({"layer": layer + 1, "val_eer": eer})
        print(f"layer {layer+1:2d}  val EER {eer*100:6.2f}%", flush=True)

    table = pd.DataFrame(rows).sort_values("val_eer")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    best = int(table.iloc[0]["layer"])
    print(f"\nbest layer: {best}  (val EER {table.iloc[0]['val_eer']*100:.2f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
