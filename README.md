# Smiling or Not — Binary Facial Expression Classification

A convolutional neural network (CNN) trained **from scratch** to tell whether a person is smiling in a face photo (`smile` vs `non_smile`).

**Author:** Maximilien Muller — academic project, ESIEE Paris, course *IA et Deep Learning* (2025–2026)

## Results

| Metric | Value |
|---|---|
| Dataset | 1,203 labelled face images (600 smile / 603 non-smile) |
| Split | 80 % train (962) / 20 % validation (241), fixed seed |
| Validation accuracy | **94–96 %** depending on the run (96.3 % in the run saved in the notebook) |
| Model size | 1.27 M parameters (3 convolutional blocks) |
| Training | 40 epochs, Adam (lr = 1e-4), batch size 32, input 64×64 RGB |

> **Caveat.** The best checkpoint is selected on the same validation set that is used to report accuracy, and there is no separate test set. The figures above are therefore slightly optimistic. With only ~1,200 images, results also vary from run to run.

## Approach

1. **Data** — [Smiling or Not Face Data](https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data) (Kaggle). The near-equal class balance avoids a biased classifier. The unlabelled `test/` folder is not used.
2. **Split** — files are physically copied into `train/` and `val/` (fixed `seed=42`) instead of using Keras `validation_split`, which can give inconsistent splits across two generators.
3. **Augmentation (train only)** — rotation ±20°, shifts 10 %, zoom 10 %, horizontal flip. The validation set is only rescaled, so evaluation stays unbiased.
4. **Model** — 3 × [`Conv2D` → `BatchNormalization` → `MaxPooling2D` → `Dropout`] with 32 → 64 → 128 filters, then `Dense(256)`, `Dropout(0.5)` and a sigmoid output. Dropout grows with depth (0.25 → 0.25 → 0.4 → 0.5) to limit overfitting.
5. **Training** — binary cross-entropy, `ModelCheckpoint` on validation accuracy.
6. **Inference** — resize a photo to 64×64, normalise, predict. Score > 0.5 means *smiling*.

## Run it

The notebook was written for **Google Colab**.

1. Open `smiling_or_not_cnn.ipynb` in Colab.
2. Create a Kaggle API token (Kaggle → Settings → *Create New Token*) and upload your own `kaggle.json` when the notebook asks.
3. Run all cells.

Local run: `pip install -r requirements.txt`, then replace the Colab-specific upload cells (`google.colab.files`) with local paths.

## Limitations and next steps

- Small dataset: no cross-validation, and a single train/validation split.
- A held-out test set would give an honest final score.
- Transfer learning (e.g. MobileNetV2) and closed-mouth or varied smiles are natural improvements.
- The model is trained on one dataset only; it has not been tested on other lighting conditions, ages or camera angles.

## Tech stack

Python · TensorFlow / Keras · NumPy · Matplotlib
