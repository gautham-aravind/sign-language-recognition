import time
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

MODEL_PATH = "augstrong_seed44.keras"   # copy from ../models/
CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
THRESHOLD = 0.45   # below this confidence the frame votes "?"
WINDOW = 7         # majority vote over the last N frames
BOX = 300          # side of the square capture box, in pixels


class WebcamStyleAugment(layers.Layer):
    # Training-only augmentation layer; identity at prediction time, so a pass-through stub is exact.
    def call(self, x, training=None):
        return x


model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"WebcamStyleAugment": WebcamStyleAugment})

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open camera 0 (try index 1 if a virtual camera is taking slot 0)")

recent = deque(maxlen=WINDOW)
prev = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    x0, y0 = (w - BOX) // 2, (h - BOX) // 2
    crop = frame[y0:y0 + BOX, x0:x0 + BOX]

    # Same preprocessing as training: OpenCV gives BGR, the model was trained on RGB.
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (128, 128)).astype("float32") / 255.0
    probs = model.predict(x[None], verbose=0)[0]
    best = int(np.argmax(probs))
    recent.append(best if probs[best] >= THRESHOLD else -1)

    vote, count = Counter(recent).most_common(1)[0]
    label = CLASSES[vote] if vote >= 0 and count >= WINDOW // 2 + 1 else "?"

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
    prev = now

    # Mirror the display only; the model sees the unflipped crop.
    shown = cv2.flip(frame, 1)
    sx0 = w - (x0 + BOX)
    cv2.rectangle(shown, (sx0, y0), (sx0 + BOX, y0 + BOX), (0, 255, 0), 2)
    cv2.putText(shown, f"{label}  ({probs[best]:.2f})", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.putText(shown, f"{fps:.1f} FPS", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    top3 = np.argsort(probs)[::-1][:3]
    cv2.putText(shown, "  ".join(f"{CLASSES[i]}:{probs[i]:.2f}" for i in top3),
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("Sign recognition (q to quit)", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
