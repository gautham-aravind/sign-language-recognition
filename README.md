# Static ASL Sign Recognition

Recognition of 36 static ASL signs (digits 0-9, letters A-Z) from a single image, built as a solo rebuild of an
earlier undergraduate team project. This version focuses on a **systematic, honest evaluation**: a participant-independent
split, a fixed baseline, one variable changed at a time, several seeds per experiment, and a real-time webcam demo.

> The original project was a three-person team project (Python/OpenCV, Motion History Images, CNN). My part was mainly the
> Python/image-processing side. This repo is a new, static-image implementation, not a copy of that work.
> Dynamic gestures (MHI) are out of scope for now.
>
> Built as a solo project: I designed the experimental setup and made the modelling decisions; I used an AI assistant
> (Claude) for code drafting, debugging and explanations, and I ran, checked and interpreted all experiments myself.

## Dataset

[ASL-HG on Mendeley Data](https://data.mendeley.com/datasets/j4y5w2c8w9/1), file `ASL_Raw_Images.zip`
(36 classes, 1,000 images each, 10 participants P1-P10, 300x300 RGB JPEG). **Not included here**; download it and extract to
`dataset/asl_dataset/<class>/P<k>_<class>_<id>.jpg`.

Split by participant, so nobody appears in two sets:

| Split | Participants | Images |
|---|---|---|
| Train | P1-P8 | 28,800 |
| Validation | P9 | 3,600 |
| Test | P10 | 3,600 |

## Model

Small CNN (about 426k parameters): four `Conv2D(3x3)+MaxPool` blocks (32/64/128/256 filters), global average pooling,
`Dense(128)`, `Dropout(0.3)`, 36-way softmax. Input: RGB resized to 128x128, scaled to 0-1. Adam (1e-3), batch size 32,
early stopping on validation loss (patience 5).

## Results

Validation = P9. Each cell is a mean over seeds with the range; one change per row relative to the baseline.

| Experiment | Seeds | P9 val accuracy |
|---|---|---|
| Baseline (raw RGB 128x128) | 3 | 0.751 (0.630-0.859) |
| + augmentation (shift, zoom, rotate, brightness, contrast) | 3 | **0.923** (0.908-0.932) |
| + augmentation, 192x192 input | 1 | 0.893 |
| + strong augmentation (blur, noise, saturation, colour cast) | 3 | 0.873 (0.837-0.896) |
| + augmentation + grayscale | 3 | 0.918 (0.898-0.953) |

**Held-out test (P10), evaluated once on the chosen augmented model: 83.25% accuracy.**

What the experiments showed:
- **Seed variance is large for the baseline** (63-86% with identical code), so single runs are unreliable.
- **Augmentation** was the biggest improvement and made training much steadier across seeds.
- **Higher resolution and grayscale gave no evidence of benefit** on P9.
- **Errors are concentrated in look-alike signs** (for example 2/V, 6/W, R/U, M/E), and which side a model picks can flip
  per participant. A single held-out person is therefore a noisy estimate.
- **Static images cannot capture motion signs** (J, Z), which is where a dynamic approach (such as MHI) would matter.

### Limitations

- One validation and one test participant; results would be more trustworthy rotated over several held-out people.
- Three seeds per experiment, so no precise confidence intervals.
- The P9 numbers are slightly optimistic because P9 was used to pick the best epoch and model.
- The 192x192 result is a single seed.

## Real-time webcam demo

`webcam/webcam.py` crops a fixed square box, feeds it to the model, and shows the predicted sign with confidence, a
majority vote over the last 7 frames, a confidence threshold (below it the app shows `?`) and the FPS.

Informal checks on my own webcam (one person, one camera): with a **cluttered background**, early models mostly
answered "H" for every sign. After training with webcam-style augmentation (`augstrong`) and using a **plain
background** plus a 0.45 threshold, A, L, 5, O and Y were all recognised. This is a handful of signs, not a measured
accuracy. With the 0.45 threshold, an empty box still showed "?" (no false letter). The dataset-to-webcam gap (camera quality, lighting, background) is the main open problem. Speed was
about 12 FPS on CPU.

```
cd webcam
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python webcam.py
```

Models in `models/`: `augmented_seed43.keras` (final P10-tested model) and `augstrong_seed44.keras` (used in the demo).
`augstrong` contains a custom layer, handled by the stub class in `webcam.py`.

## Repository layout

```
notebooks/
  01_dataset_exploration.ipynb   raw lab notebook (all experiments, with outputs)
  03_final_experiments.ipynb     cleaned, safe-to-run-all version
experiments/                     per-run settings and per-epoch history (JSON)
models/                          the two published models
webcam/                          real-time demo
```

## Reproducing

```
pip install -r requirements.txt
```

Open `notebooks/03_final_experiments.ipynb`. With `RUN_TRAINING = False` it reads saved results instead of retraining.
Training the experiments takes hours on a laptop GPU (GTX 1650 Ti, 4 GB). On WSL2 the pip-installed CUDA libraries need
`LD_LIBRARY_PATH` to include the `nvidia/*/lib` folders inside the virtual environment.

## Future work

Rotate held-out participants; skin-colour segmentation that keeps the hand's pixels; collect a small in-domain webcam
set; measure per-sign webcam accuracy; dynamic gestures with Motion History Images.
