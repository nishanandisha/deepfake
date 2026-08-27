"""Does the model detect synthesis, or just recording conditions?

The human half of this corpus is 933 chunks of 14 YouTube videos; the AI half
is clean TTS output. "Compressed and noisy = human, clean = AI" separates
those two groups perfectly without learning anything about synthesis -- and it
would transfer effortlessly to any *new* TTS system, which would make a small
seen/unseen gap look like generalisation when it is nothing of the kind.

The probe: take AI clips the model scores confidently and push them through
the channel degradations that characterise the human class. Synthesis
artifacts should survive a codec; a channel shortcut should not. If p(AI)
collapses, the low EER is an artifact of the corpus.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import apply_temperature, load_policy
from src.preprocessing.augment import augment
from src.preprocessing.embeddings import WavLMFrontend, load_audio, pick_device
from src.models.head import VoiceClassifierHead

LAYER = 12
device = pick_device()
state = torch.load("outputs/run/best.pt", map_location=device, weights_only=False)
cfg = state["config"]
model = VoiceClassifierHead(768, cfg["proj_dim"], 128, cfg["hidden_dim"], cfg["dropout"]).to(device)
model.load_state_dict(state["model"]); model.eval()
frontend = WavLMFrontend(device=device, layer=LAYER)
policy = load_policy("outputs/run/policy.json")

@torch.no_grad()
def p_ai(signal):
    f = torch.from_numpy(frontend.embed(signal)).unsqueeze(0).to(device)
    return float(apply_temperature(np.array([model(f).item()]), policy["temperature"])[0])

manifest = pd.read_csv("data/splits/manifest.csv")
test = manifest[manifest["split"] == "test"]
rng = np.random.default_rng(0)

print(f"threshold = {policy['threshold']:.4f}  (>= means AI)\n")
for label_name, label in [("AI", 1), ("HUMAN", 0)]:
    frame = test[test["label"] == label]
    frame = frame.iloc[rng.choice(len(frame), min(40, len(frame)), replace=False)]
    clean, degraded = [], []
    for row in frame.itertuples(index=False):
        signal = load_audio(row.path)
        clean.append(p_ai(signal))
        degraded.append(p_ai(augment(signal, seed=int(rng.integers(1 << 30)))))
    clean, degraded = np.array(clean), np.array(degraded)
    thr = policy["threshold"]
    correct = (lambda a: (a >= thr) if label == 1 else (a < thr))
    print(f"{label_name} clips (n={len(frame)})")
    print(f"  clean     mean p(AI) {clean.mean():.4f}   correct {correct(clean).mean()*100:5.1f}%")
    print(f"  degraded  mean p(AI) {degraded.mean():.4f}   correct {correct(degraded).mean()*100:5.1f}%")
    print(f"  -> shift  {degraded.mean()-clean.mean():+.4f}\n")
