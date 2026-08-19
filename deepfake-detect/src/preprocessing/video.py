"""Frame sampling + face detection/alignment for the visual branch.

Face detection uses OpenCV's bundled Haar cascade (no extra dependency).
This is a coarse bounding-box crop, not landmark-based alignment -- if
detection quality turns out to matter for Stage 2 results, swap
`detect_face_bbox` for a landmark-based aligner (e.g. mediapipe) without
touching the rest of the pipeline, since callers only depend on
`align_face` returning a fixed-size RGB array.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def sample_frames(video_path: str, frame_rate: float) -> List[np.ndarray]:
    """Uniformly sample frames from a video at `frame_rate` fps, returned as
    a list of RGB uint8 arrays in their native resolution (unaligned)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or frame_rate
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(round(native_fps / frame_rate)), 1)

    frames = []
    for frame_idx in range(0, max(frame_count, 1), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def detect_face_bbox(frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Returns (x, y, w, h) of the largest detected face, or None."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    return tuple(max(faces, key=lambda box: box[2] * box[3]))


def align_face(frame_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Crop to the detected face (falling back to a center crop when no
    face is found) and resize to (size, size)."""
    height, width = frame_rgb.shape[:2]
    bbox = detect_face_bbox(frame_rgb)

    if bbox is not None:
        x, y, w, h = bbox
        # Pad the box by 25% so the crop includes forehead/chin, common for
        # face-alignment pipelines that then resize to a square input.
        pad_x, pad_y = int(w * 0.25), int(h * 0.25)
        x0, y0 = max(x - pad_x, 0), max(y - pad_y, 0)
        x1, y1 = min(x + w + pad_x, width), min(y + h + pad_y, height)
    else:
        side = min(height, width)
        x0, y0 = (width - side) // 2, (height - side) // 2
        x1, y1 = x0 + side, y0 + side

    crop = frame_rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_video(video_path: str, frame_rate: float, size: int = 224) -> np.ndarray:
    """Full video -> [T, size, size, 3] uint8 aligned-face frame tensor."""
    frames = sample_frames(video_path, frame_rate)
    aligned = [align_face(frame, size=size) for frame in frames]
    return np.stack(aligned) if aligned else np.zeros((0, size, size, 3), dtype=np.uint8)
