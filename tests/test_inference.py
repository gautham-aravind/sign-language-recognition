"""Cropping, preprocessing and smoothing used by the webcam demo. No camera, GPU or TensorFlow needed."""
import unittest

import numpy as np

from src.data import CLASSES
from src.inference import (BOX, IMAGE_SIZE, THRESHOLD, WINDOW, MajorityVoteSmoother, box_origin, crop_center,
                           predict_probs, preprocess_bgr_crop)


def probs_for(index, p):
    """A 36-way probability vector whose top class is `index` with probability `p`."""
    v = np.full(len(CLASSES), (1.0 - p) / (len(CLASSES) - 1), dtype="float32")
    v[index] = p
    return v


A, B, ZERO = CLASSES.index("A"), CLASSES.index("B"), CLASSES.index("0")


class CropTests(unittest.TestCase):
    def test_crop_is_centred_square(self):
        frame = np.arange(480 * 640 * 3, dtype=np.uint32).reshape(480, 640, 3).astype(np.uint8)
        crop = crop_center(frame)
        self.assertEqual(crop.shape, (BOX, BOX, 3))
        x0, y0 = box_origin(640, 480)
        self.assertEqual((x0, y0), (170, 90))
        np.testing.assert_array_equal(crop, frame[90:390, 170:470])


class PreprocessTests(unittest.TestCase):
    def test_shape_dtype_range(self):
        crop = np.random.default_rng(0).integers(0, 256, (BOX, BOX, 3), dtype=np.uint8)
        x = preprocess_bgr_crop(crop)
        self.assertEqual(x.shape, (IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertEqual(x.dtype, np.float32)
        self.assertGreaterEqual(float(x.min()), 0.0)
        self.assertLessEqual(float(x.max()), 1.0)

    def test_bgr_is_converted_to_rgb(self):
        crop = np.zeros((BOX, BOX, 3), dtype=np.uint8)
        crop[..., 0] = 255                      # pure blue in OpenCV's BGR order
        x = preprocess_bgr_crop(crop)
        self.assertAlmostEqual(float(x[..., 2].mean()), 1.0)   # blue must end up in the last (RGB) channel
        self.assertAlmostEqual(float(x[..., 0].mean()), 0.0)


class FakeModel:
    """Stands in for the Keras model: records its input and returns a fixed probability vector."""

    def __init__(self, probs):
        self.probs = probs
        self.seen = None

    def predict(self, batch, verbose=0):
        self.seen = batch
        return self.probs[None]


class PredictTests(unittest.TestCase):
    def test_predict_probs_uses_batch_of_one_and_returns_36_probabilities(self):
        model = FakeModel(probs_for(A, 0.9))
        crop = np.full((BOX, BOX, 3), 128, dtype=np.uint8)
        probs = predict_probs(model, crop)
        self.assertEqual(model.seen.shape, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertLessEqual(float(model.seen.max()), 1.0)
        self.assertEqual(probs.shape, (len(CLASSES),))
        self.assertEqual(CLASSES[int(np.argmax(probs))], "A")


class SmootherTests(unittest.TestCase):
    def feed(self, smoother, frames):
        result = None
        for index, p in frames:
            result = smoother.update(probs_for(index, p))
        return result

    def test_defaults_match_documented_behaviour(self):
        self.assertEqual((WINDOW, THRESHOLD), (7, 0.45))

    def test_needs_a_majority_before_showing_a_label(self):
        s = MajorityVoteSmoother()
        self.assertIsNone(self.feed(s, [(A, 0.9)] * 3).index)      # 3 of 7 is not enough
        result = self.feed(s, [(A, 0.9)])                          # 4th vote reaches window // 2 + 1
        self.assertEqual(result.label, "A")

    def test_low_confidence_frames_vote_uncertain(self):
        s = MajorityVoteSmoother()
        result = self.feed(s, [(A, 0.30)] * 7)
        self.assertIsNone(result.index)
        self.assertIsNone(result.confidence)
        self.assertEqual(result.label, "?")

    def test_threshold_is_inclusive(self):
        s = MajorityVoteSmoother()
        self.assertEqual(self.feed(s, [(A, 0.45)] * 4).label, "A")

    def test_uncertain_frames_can_block_a_label(self):
        s = MajorityVoteSmoother()
        result = self.feed(s, [(A, 0.9)] * 3 + [(A, 0.2)] * 4)     # 3 confident A vs 4 uncertain
        self.assertEqual(result.label, "?")

    def test_old_votes_leave_the_window(self):
        s = MajorityVoteSmoother()
        self.feed(s, [(A, 0.9)] * 7)
        result = self.feed(s, [(B, 0.9)] * 4)                      # A now holds only 3 of the last 7
        self.assertEqual(result.label, "B")

    def test_confidence_belongs_to_the_displayed_label(self):
        # Regression: the demo used to show the *current frame's* top probability next to the voted label.
        # Here the last frame is a confident B (0.99) but A holds the vote, so the number must describe A.
        s = MajorityVoteSmoother()
        frames = [(A, 0.9), (A, 0.8), (A, 0.7), (A, 0.6), (B, 0.99), (B, 0.99), (B, 0.99)]
        result = self.feed(s, frames)
        self.assertEqual(result.label, "A")
        self.assertAlmostEqual(result.confidence, (0.9 + 0.8 + 0.7 + 0.6) / 4, places=5)

    def test_confidence_ignores_uncertain_frames(self):
        s = MajorityVoteSmoother()
        result = self.feed(s, [(A, 0.3)] * 3 + [(A, 0.8)] * 4)
        self.assertEqual(result.label, "A")
        self.assertAlmostEqual(result.confidence, 0.8, places=5)


if __name__ == "__main__":
    unittest.main()
