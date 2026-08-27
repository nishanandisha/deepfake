"""Fit temperature and the operating threshold on the calibration split."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import build_policy, save_policy
from src.models.head import VoiceClassifierHead
from src.preprocessing.dataset import EmbeddingDataset, collate_padded
from src.preprocessing.embeddings import EmbeddingCache, pick_device
from src.training.train import collect_logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/run/best.pt")
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out", default="outputs/run/policy.json")
    args = parser.parse_args()

    device = pick_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = VoiceClassifierHead(
        input_dim=768, proj_dim=cfg["proj_dim"],
        hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    manifest = pd.read_csv(args.manifest)
    frame = manifest[manifest["split"] == "calibration"]
    cache = EmbeddingCache(args.cache_dir, args.layer)
    dataset = EmbeddingDataset(frame, cache, variants=["clean"], max_frames=cfg["max_frames"])
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_padded)

    logits, labels = collect_logits(model, loader, device)
    policy, _ = build_policy(logits, labels)

    print(f"calibration clips  {policy['n_calibration']}")
    print(f"temperature        {policy['temperature']:.4f}")
    print(f"threshold          {policy['threshold']:.4f}")
    print(f"ECE  before {policy['ece_before']:.4f} -> after {policy['ece_after']:.4f}")
    save_policy(policy, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
