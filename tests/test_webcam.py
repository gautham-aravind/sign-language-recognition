"""The webcam script must be importable without touching the camera, and must find its model by file location."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.inference import Smoothed

SCRIPT = Path(__file__).resolve().parent.parent / "webcam" / "webcam.py"


def load_webcam_module():
    spec = importlib.util.spec_from_file_location("webcam_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WebcamScriptTests(unittest.TestCase):
    def test_import_does_not_open_the_camera(self):
        with mock.patch("cv2.VideoCapture", side_effect=AssertionError("camera opened at import")):
            module = load_webcam_module()
        self.assertTrue(callable(module.main))

    def test_default_model_path_does_not_depend_on_working_directory(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                module = load_webcam_module()
            finally:
                os.chdir(previous)
        self.assertTrue(module.DEFAULT_MODEL.is_absolute())
        self.assertEqual(module.DEFAULT_MODEL.name, "augstrong_seed44.keras")
        self.assertTrue(module.DEFAULT_MODEL.exists())

    def test_missing_model_exits_with_a_message_before_opening_the_camera(self):
        module = load_webcam_module()
        with mock.patch("cv2.VideoCapture", side_effect=AssertionError("camera opened")):
            with self.assertRaises(SystemExit) as ctx:
                module.main(["--model", "does_not_exist.keras"])
        self.assertIn("Model not found", str(ctx.exception))

    def test_overlay_handles_both_confident_and_uncertain_results(self):
        module = load_webcam_module()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        probs = np.full(36, 1 / 36, dtype="float32")
        for result in (Smoothed(10, 0.8), Smoothed(None, None)):
            shown = module.draw_overlay(frame, result, probs, 12.0, 640, 170, 90)
            self.assertEqual(shown.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
