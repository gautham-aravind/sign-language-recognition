"""Real-time static ASL demo: crop a fixed box from the webcam, classify it, smooth over a few frames.

Run from anywhere:  python webcam/webcam.py   (add --model PATH to use a different .keras file)
The default model is models/augstrong_seed44.keras in this repository, found relative to this file.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so `from src...` works wherever the script is launched from

from src.data import CLASSES                                              # noqa: E402
from src.inference import (BOX, THRESHOLD, WINDOW, MajorityVoteSmoother,  # noqa: E402
                           box_origin, crop_center, predict_probs)

DEFAULT_MODEL = REPO_ROOT / "models" / "augstrong_seed44.keras"


def load_model(model_path):
    """Load the saved Keras model. TensorFlow is imported here so importing this file stays light."""
    import tensorflow as tf
    from tensorflow.keras import layers

    class WebcamStyleAugment(layers.Layer):
        # Training-only augmentation layer; identity at prediction time, so a pass-through stub is exact.
        def call(self, x, training=None):
            return x

    return tf.keras.models.load_model(model_path, custom_objects={"WebcamStyleAugment": WebcamStyleAugment})


def draw_overlay(frame, result, probs, fps, frame_w, x0, y0):
    """Mirror the frame for display and draw the box, smoothed label, FPS and current top-3."""
    shown = cv2.flip(frame, 1)   # mirror for display only; the model sees the unflipped crop
    sx0 = frame_w - (x0 + BOX)
    cv2.rectangle(shown, (sx0, y0), (sx0 + BOX, y0 + BOX), (0, 255, 0), 2)

    # The label and number come from the same smoothed result: confidence is the mean probability over the
    # frames that voted for this label. When the vote is uncertain, only "?" is shown.
    text = "?" if result.index is None else f"{result.label}  (avg {result.confidence:.2f})"
    cv2.putText(shown, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.putText(shown, f"{fps:.1f} FPS", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Raw single-frame probabilities, labelled as such: these may disagree with the smoothed label.
    top3 = np.argsort(probs)[::-1][:3]
    cv2.putText(shown, "now: " + "  ".join(f"{CLASSES[i]}:{probs[i]:.2f}" for i in top3),
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return shown


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real-time static ASL recognition demo")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"path to a .keras model (default: {DEFAULT_MODEL})")
    parser.add_argument("--camera", type=int, default=0, help="camera index (try 1 if a virtual camera takes slot 0)")
    args = parser.parse_args(argv)

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}\nRun from a full clone of the repository, or pass --model PATH.")
    model = load_model(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera} (try --camera 1 if a virtual camera is taking slot 0)")

    smoother = MajorityVoteSmoother(WINDOW, THRESHOLD)
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        x0, y0 = box_origin(w, h, BOX)

        probs = predict_probs(model, crop_center(frame, BOX))
        result = smoother.update(probs)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        cv2.imshow("Sign recognition (q to quit)", draw_overlay(frame, result, probs, fps, w, x0, y0))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
