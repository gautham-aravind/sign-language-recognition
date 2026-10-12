"""Class order and participant-split logic. Uses temporary empty files; no dataset or TensorFlow needed."""
import tempfile
import unittest
from pathlib import Path

from src.data import CLASSES, collect_split, list_classes, participant_of


class ClassOrderTests(unittest.TestCase):
    def test_36_classes_digits_then_letters(self):
        expected = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.assertEqual(CLASSES, expected)
        self.assertEqual(len(CLASSES), 36)

    def test_matches_sorted_folder_names(self):
        # The notebook builds class indices from sorted folder names; that must equal the canonical order,
        # otherwise the webcam app would label signs differently from training.
        self.assertEqual(sorted(CLASSES), CLASSES)


class ParticipantSplitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        # 3 classes x participants P1, P2, P9, P10 x 2 images, all empty placeholder files.
        for cls in ("0", "A", "B"):
            (self.root / cls).mkdir()
            for p in ("P1", "P2", "P9", "P10"):
                for k in (1, 2):
                    (self.root / cls / f"{p}_{cls}_{k}.jpg").touch()
        (self.root / "A" / "notes.txt").touch()      # non-jpg files must be ignored
        self.classes = ["0", "A", "B"]

    def test_participant_of(self):
        self.assertEqual(participant_of("P5_A_433.jpg"), "P5")
        self.assertEqual(participant_of("/some/dir/P10_Z_7.jpg"), "P10")

    def test_split_matches_exact_participant_id(self):
        paths, _ = collect_split(self.root, self.classes, ["P1"])
        names = {Path(p).name for p in paths}
        self.assertTrue(all(n.startswith("P1_") for n in names))   # 'P1' must not match 'P10'
        self.assertEqual(len(paths), 3 * 2)

    def test_splits_are_disjoint_and_complete(self):
        train, _ = collect_split(self.root, self.classes, ["P1", "P2"])
        val, _ = collect_split(self.root, self.classes, ["P9"])
        test, _ = collect_split(self.root, self.classes, ["P10"])
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(val) & set(test))
        self.assertEqual(len(train) + len(val) + len(test), 3 * 4 * 2)

    def test_labels_follow_class_order(self):
        paths, labels = collect_split(self.root, self.classes, ["P9"])
        self.assertEqual(len(paths), len(labels))
        for path, label in zip(paths, labels):
            self.assertEqual(self.classes[label], Path(path).parent.name)
        self.assertEqual(sorted(set(labels)), [0, 1, 2])

    def test_deterministic_order(self):
        first = collect_split(self.root, self.classes, ["P1", "P2"])
        second = collect_split(self.root, self.classes, ["P1", "P2"])
        self.assertEqual(first, second)

    def test_list_classes_reads_sorted_folders(self):
        self.assertEqual(list_classes(self.root), ["0", "A", "B"])

    def test_list_classes_falls_back_to_canonical_list(self):
        self.assertEqual(list_classes(self.root / "missing"), CLASSES)


if __name__ == "__main__":
    unittest.main()
