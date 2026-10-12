"""Class list and participant-based splitting. Plain Python (no TensorFlow), so it is cheap to import and test."""
from pathlib import Path

# Class index = position in this list: digits 0-9, then letters A-Z. Training, evaluation and the webcam app
# must all use this order (it equals the sorted folder names of the dataset).
CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def list_classes(dataset_dir):
    """Class names from the dataset folders (sorted), or the canonical list if the dataset is not present."""
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        return sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    return list(CLASSES)


def participant_of(image_path):
    """Participant id from a filename: 'P5_A_433.jpg' -> 'P5'. No image loading needed."""
    return Path(image_path).stem.split("_")[0]


def collect_split(dataset_dir, classes, participants):
    """Image paths and integer labels for the given participants.

    Matching is on the exact participant id, so 'P1' does not pick up 'P10'. Files are visited class by class in
    sorted filename order, which keeps the result deterministic.
    """
    dataset_dir = Path(dataset_dir)
    class_to_index = {name: i for i, name in enumerate(classes)}
    paths, labels = [], []
    for class_name in classes:
        for img_path in sorted((dataset_dir / class_name).glob("*.jpg")):
            if participant_of(img_path) in participants:
                paths.append(str(img_path))
                labels.append(class_to_index[class_name])
    return paths, labels

