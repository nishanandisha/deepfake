"""Stage 7, part 4: the moderator view.

Given one sample, produces a single self-contained HTML report with:
  - the calibrated authenticity score and approve/flag/block decision
  - the modality split (coarse SHAP) with its numeric attributions
  - the top-k named acoustic descriptors and their SHAP values
  - saliency-overlaid frames for the most-implicated frames

Deliberately a static HTML file rather than a UI: Stage 9 (optional) wraps
the same underlying explanation functions in an interactive app, so the
report generator stays a thin presentation layer over src/explain/*.
"""

import base64
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def _frame_to_data_uri(frame_rgb: np.ndarray) -> str:
    """PNG-encode a frame as a base64 data URI so the report is a single
    self-contained file with no sidecar images."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def _decision_color(decision: str) -> str:
    return {"approve": "#1a7f37", "flag": "#9a6700", "block": "#cf222e"}.get(decision, "#57606a")


def build_report_html(
    sample_id: str,
    authenticity_score: float,
    decision: str,
    modality_split: Dict[str, float],
    top_acoustic_features: List[tuple],
    saliency_frames: List[np.ndarray],
    frame_indices: List[int],
    ground_truth_label: Optional[str] = None,
    ground_truth_modality: Optional[str] = None,
) -> str:
    visual_pct = modality_split["visual_share"] * 100
    acoustic_pct = modality_split["acoustic_share"] * 100

    feature_rows = "\n".join(
        f"<tr><td>{name}</td><td class='num'>{value:+.4f}</td></tr>"
        for name, value in top_acoustic_features
    )

    frame_cells = "\n".join(
        f"<figure><img src='{_frame_to_data_uri(frame)}' alt='frame {idx}'>"
        f"<figcaption>frame {idx}</figcaption></figure>"
        for frame, idx in zip(saliency_frames, frame_indices)
    )

    ground_truth_block = ""
    if ground_truth_label is not None:
        ground_truth_block = (
            "<p class='muted'>Ground truth: "
            f"<strong>{ground_truth_label}</strong>"
            + (f" (manipulated: {ground_truth_modality})" if ground_truth_modality else "")
            + "</p>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Moderator review -- {sample_id}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem auto;
          max-width: 900px; color: #1f2328; }}
  .decision {{ font-size: 1.5rem; font-weight: 700; color: {_decision_color(decision)}; }}
  .bar {{ display: flex; height: 28px; border-radius: 4px; overflow: hidden; margin: .5rem 0; }}
  .bar .visual {{ background: #0969da; width: {visual_pct:.1f}%; }}
  .bar .acoustic {{ background: #8250df; width: {acoustic_pct:.1f}%; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: .5rem; }}
  th, td {{ border-bottom: 1px solid #d0d7de; padding: .4rem .6rem; text-align: left; }}
  td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  figure {{ display: inline-block; margin: .5rem; text-align: center; }}
  figure img {{ width: 200px; border-radius: 4px; }}
  figcaption {{ font-size: .8rem; color: #57606a; }}
  .muted {{ color: #57606a; font-size: .9rem; }}
</style></head><body>
<h1>Moderator review</h1>
<p class="muted">Sample: <code>{sample_id}</code></p>

<h2>Decision</h2>
<p class="decision">{decision.upper()}</p>
<p>Calibrated authenticity score: <strong>{authenticity_score:.4f}</strong>
   (1.0 = certainly authentic, 0.0 = certainly manipulated)</p>
{ground_truth_block}

<h2>Modality split (coarse SHAP)</h2>
<div class="bar"><div class="visual"></div><div class="acoustic"></div></div>
<p>Visual <strong>{visual_pct:.1f}%</strong> &middot;
   Acoustic <strong>{acoustic_pct:.1f}%</strong></p>
<p class="muted">SHAP values: visual {modality_split['shap_visual']:+.4f},
   acoustic {modality_split['shap_acoustic']:+.4f}
   (base {modality_split['base_value']:+.4f})</p>

<h2>Top acoustic descriptors (fine SHAP)</h2>
<table><thead><tr><th>Descriptor</th><th class="num">SHAP value</th></tr></thead>
<tbody>
{feature_rows}
</tbody></table>
<p class="muted">Positive values push toward "manipulated".</p>

<h2>Most-implicated frames (Grad-CAM)</h2>
{frame_cells}
</body></html>
"""


def write_report(output_path: str, html: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
