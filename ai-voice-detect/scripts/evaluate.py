"""Score the test split and report the numbers that actually matter.

Prints three things, in increasing order of honesty:

  overall        every test clip pooled
  seen           generators that appeared in training
  UNSEEN         generators withheld entirely -- the headline number

The gap between `seen` and `UNSEEN` is the measurement this whole project
exists to produce. A strong pooled figure carried entirely by the seen
generators would be the same mistake the predecessor project made when it
reported 0.972 AUC without ever running a cross-dataset test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import apply_temperature, load_policy
from src.models.head import VoiceClassifierHead
from src.preprocessing.dataset import EmbeddingDataset, collate_padded
from src.preprocessing.embeddings import EmbeddingCache, pick_device
from src.preprocessing.splits import DEFAULT_HELD_OUT_SOURCES
from src.training.metrics import compute_metrics, per_source_metrics
from torch.utils.data import DataLoader


def _fmt(name, m):
    return (f"{name:<22} EER {m['eer']*100:6.2f}%   AUC {m['auc']:.4f}   "
            f"acc {m['accuracy']:.4f}   n={m['n']:4d} (ai {m['n_ai']}, human {m['n_human']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/run/best.pt")
    parser.add_argument("--policy", default="outputs/run/policy.json")
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--held-out", nargs="*", default=list(DEFAULT_HELD_OUT_SOURCES))
    parser.add_argument("--out", default="outputs/run/test_results.json")
    args = parser.parse_args()

    device = pick_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = VoiceClassifierHead(
        input_dim=768, proj_dim=cfg["proj_dim"],
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    manifest = pd.read_csv(args.manifest)
    frame = manifest[manifest["split"] == args.split].reset_index(drop=True)
    cache = EmbeddingCache(args.cache_dir, args.layer)
    dataset = EmbeddingDataset(frame, cache, variants=["clean"], max_frames=cfg["max_frames"])
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_padded)

    logits, labels = [], []
    with torch.no_grad():
        for features, mask, target in loader:
            logits.append(model(features.to(device), mask.to(device)).cpu().numpy())
            labels.append(target.numpy())
    logits = np.concatenate(logits)
    labels = np.concatenate(labels)
    sources = np.array(dataset.sources())

    policy = load_policy(args.policy) if Path(args.policy).exists() else {"temperature": 1.0, "threshold": 0.5}
    probabilities = apply_temperature(logits, policy["temperature"])
    threshold = policy["threshold"]

    overall = compute_metrics(labels, probabilities, threshold)
    held_out = set(args.held_out)
    human = labels == 0
    seen_mask = human | ~np.isin(sources, list(held_out))
    unseen_mask = human | np.isin(sources, list(held_out))

    seen = compute_metrics(labels[seen_mask], probabilities[seen_mask], threshold)
    unseen = compute_metrics(labels[unseen_mask], probabilities[unseen_mask], threshold)

    print(f"\ntemperature {policy['temperature']:.4f}   threshold {threshold:.4f}\n")
    print(_fmt("overall", overall))
    print(_fmt("seen generators", seen))
    print(_fmt(f"UNSEEN {sorted(held_out)}", unseen))

    print("\nper generator (human rows included in each):")
    for source, metrics in sorted(per_source_metrics(labels, probabilities, sources).items(),
                                  key=lambda kv: kv[1]["eer"]):
        tag = "  <- UNSEEN" if source in held_out else ""
        print(f"  {_fmt(source, metrics)}{tag}")

    results = {"overall": overall, "seen": seen, "unseen": unseen,
               "per_source": per_source_metrics(labels, probabilities, sources),
               "policy": policy}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
