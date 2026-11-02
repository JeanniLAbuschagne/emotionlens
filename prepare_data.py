"""
prepare_data.py
---------------
Loads the FER2013 dataset and turns it into numpy arrays ready for Keras.

FER2013 ships as a single CSV with three columns:
    emotion   -> integer label 0..6  (see config.EMOTIONS for the order)
    pixels    -> 2304 space-separated grayscale values (48*48)
    Usage     -> 'Training' | 'PublicTest' | 'PrivateTest'

We respect the official split so our reported test accuracy is comparable to
published benchmarks:
    Training     -> train set
    PublicTest   -> validation set
    PrivateTest  -> held-out test set (only touched by evaluate.py)
"""

import sys
import numpy as np
import pandas as pd

# allow `python training/prepare_data.py` from the project root
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import FER2013_CSV, IMG_SIZE, NUM_CLASSES


def _pixels_to_array(pixel_strings):
    """Convert the space-separated pixel strings into a normalized 4D tensor."""
    # np.fromstring is deprecated; split + np.array is the supported equivalent.
    data = np.array(
        [np.array(p.split(), dtype="float32") for p in pixel_strings]
    )
    data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    data /= 255.0  # scale pixels to [0, 1] so training is stable
    return data


def load_fer2013():
    """Return (x_train, y_train), (x_val, y_val), (x_test, y_test)."""
    if not FER2013_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {FER2013_CSV}.\n"
            "Download fer2013.csv from Kaggle "
            "(https://www.kaggle.com/datasets/deadskull7/fer2013) "
            "and place it in the data/ folder."
        )

    df = pd.read_csv(FER2013_CSV)

    # Some Kaggle mirrors ship the 'Usage' column (official train/val/test
    # split); others provide only emotion + pixels. Handle both so the loader
    # works regardless of which mirror you downloaded.
    if "Usage" not in df.columns:
        from sklearn.model_selection import train_test_split
        x_all = _pixels_to_array(df["pixels"].values)
        y_all = np.eye(NUM_CLASSES, dtype="float32")[df["emotion"].values]
        x_tmp, x_test, y_tmp, y_test = train_test_split(
            x_all, y_all, test_size=0.10, stratify=df["emotion"].values,
            random_state=42)
        strat = np.argmax(y_tmp, axis=1)
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_tmp, y_tmp, test_size=0.111, stratify=strat, random_state=42)
        print(f"Loaded FER2013 (self-split): {len(x_tr)} train / "
              f"{len(x_val)} val / {len(x_test)} test images")
        return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)

    splits = {}
    for usage in ["Training", "PublicTest", "PrivateTest"]:
        subset = df[df["Usage"] == usage]
        x = _pixels_to_array(subset["pixels"].values)
        # one-hot encode the integer labels for categorical cross-entropy
        y = np.eye(NUM_CLASSES, dtype="float32")[subset["emotion"].values]
        splits[usage] = (x, y)

    print(
        f"Loaded FER2013: "
        f"{len(splits['Training'][0])} train / "
        f"{len(splits['PublicTest'][0])} val / "
        f"{len(splits['PrivateTest'][0])} test images"
    )
    return splits["Training"], splits["PublicTest"], splits["PrivateTest"]


if __name__ == "__main__":
    # quick sanity check
    (xtr, ytr), (xv, yv), (xte, yte) = load_fer2013()
    print("train tensor shape:", xtr.shape, "labels:", ytr.shape)
