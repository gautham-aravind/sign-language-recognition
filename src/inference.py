"""Webcam-side logic that can be tested without a camera or TensorFlow: cropping, preprocessing and smoothing.

Preprocessing here must stay equivalent to training: RGB, 128x128, scaled to 0-1.
"""
from collections import Counter, deque
from typing import NamedTuple, Optional

import cv2
import numpy as np

from src.data import CLASSES

IMAGE_SIZE = 128      # model input is IMAGE_SIZE x IMAGE_SIZE x 3
THRESHOLD = 0.45      # a frame whose top probability is below this votes "uncertain"
WINDOW = 7            # majority vote over the last N frames
BOX = 300             # side of the square capture box, in pixels

UNCERTAIN = -1        # vote value for a frame below the threshold


def box_origin(frame_width, frame_height, box=BOX):
    """Top-left corner (x0, y0) of a centred square box."""
    return (frame_width - box) // 2, (frame_height - box) // 2


def crop_center(frame, box=BOX):
    """Centred square crop of a BGR frame (as returned by cv2.VideoCapture)."""
    h, w = frame.shape[:2]
    x0, y0 = box_origin(w, h, box)
    return frame[y0:y0 + box, x0:x0 + box]


def preprocess_bgr_crop(crop_bgr, size=IMAGE_SIZE):
    """BGR uint8 crop -> float32 RGB array of shape (size, size, 3) in [0, 1].

    OpenCV delivers BGR but the model was trained on RGB, so the channels are swapped first.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size)).astype("float32") / 255.0


def predict_probs(model, crop_bgr, size=IMAGE_SIZE):
    """Class probabilities (length 36) for one BGR crop."""
    x = preprocess_bgr_crop(crop_bgr, size)
    return model.predict(x[None], verbose=0)[0]


class Smoothed(NamedTuple):
    """Result of one smoothing step. `index` and `confidence` are None when the display should be '?'."""
    index: Optional[int]
    confidence: Optional[float]

    @property
    def label(self):
        return "?" if self.index is None else CLASSES[self.index]


class MajorityVoteSmoother:
    """Majority vote over the last `window` frames.

    Each frame votes for its top class if that class's probability is >= `threshold`, otherwise it votes
    "uncertain". The winner is shown only if it is a real class holding a strict majority of the window
    (window // 2 + 1 votes). The reported confidence is the mean probability of the frames that voted for
    the winning class, so the label and the number always describe the same class.
    """

    def __init__(self, window=WINDOW, threshold=THRESHOLD):
        self.window = window
        self.threshold = threshold
        self._recent = deque(maxlen=window)   # (vote, probability of that frame's top class)

    def update(self, probs):
        best = int(np.argmax(probs))
        confident = probs[best] >= self.threshold   # compared as-is, exactly like the original script
        self._recent.append((best if confident else UNCERTAIN, float(probs[best])))

        vote, count = Counter(v for v, _ in self._recent).most_common(1)[0]
        if vote == UNCERTAIN or count < self.window // 2 + 1:
            return Smoothed(None, None)
        confidence = float(np.mean([p for v, p in self._recent if v == vote]))
        return Smoothed(vote, confidence)
