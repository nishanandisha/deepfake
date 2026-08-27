"""Train the classifier head. See src/training/train.py for the loop."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import TrainConfig, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--out-dir", default="outputs/run")
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(TrainConfig(
        manifest=args.manifest, cache_dir=args.cache_dir, layer=args.layer,
        out_dir=args.out_dir, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, dropout=args.dropout, seed=args.seed,
    ))
