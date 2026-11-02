"""
evaluate.py
-----------
The VALIDATION component. Quality validation is mandatory for this task, so it
lives in its own script and produces evidence a human can read:

  1. Overall accuracy on the held-out PrivateTest split (never seen in
     training) -> the honest headline number.
  2. A per-class precision / recall / F1 report -> reveals which emotions the
     model handles well and which it confuses (FER2013 famously confuses
     'fear' with 'surprise' and 'sad' with 'neutral').
  3. A confusion-matrix heatmap saved to docs/confusion_matrix.png.
  4. Training-curve plots saved to docs/training_curves.png so we can show the
     model converged and was not badly overfitting.
  5. Comparison against a trivial majority-class baseline, so the reader can
     see the model learned real signal rather than guessing the common class.

Run from the project root (after training):
    python training/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_PATH, HISTORY_PATH, DOCS_DIR, EMOTIONS
from training.prepare_data import load_fer2013


def main():
    import matplotlib
    matplotlib.use("Agg")          # headless: write files, no display needed
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tensorflow.keras.models import load_model
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)

    DOCS_DIR.mkdir(exist_ok=True)
    _, _, (x_test, y_test) = load_fer2013()
    y_true = np.argmax(y_test, axis=1)

    model = load_model(MODEL_PATH)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    # ---- headline metric ----------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    print(f"\nHeld-out test accuracy: {acc:.4f}")

    # ---- trivial baseline for context --------------------------------------
    majority = np.bincount(y_true).argmax()
    baseline_acc = accuracy_score(y_true, np.full_like(y_true, majority))
    print(f"Majority-class baseline ('{EMOTIONS[majority]}'): {baseline_acc:.4f}")

    # ---- per-class report ---------------------------------------------------
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS,
                                digits=3, zero_division=0))

    # ---- confusion matrix figure -------------------------------------------
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="magma",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, cbar=True)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Normalised confusion matrix (test acc = {acc:.1%})")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved {DOCS_DIR / 'confusion_matrix.png'}")

    # ---- training curves ----------------------------------------------------
    if HISTORY_PATH.exists():
        hist = json.loads(HISTORY_PATH.read_text())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(hist["accuracy"], label="train")
        ax1.plot(hist["val_accuracy"], label="val")
        ax1.set_title("Accuracy"); ax1.set_xlabel("epoch"); ax1.legend()
        ax2.plot(hist["loss"], label="train")
        ax2.plot(hist["val_loss"], label="val")
        ax2.set_title("Loss"); ax2.set_xlabel("epoch"); ax2.legend()
        plt.tight_layout()
        plt.savefig(DOCS_DIR / "training_curves.png", dpi=150)
        print(f"Saved {DOCS_DIR / 'training_curves.png'}")

    # ---- OPTIONAL: cross-check against a pretrained model ------------------
    # Uncomment if you installed deepface (pip install deepface). This gives a
    # second, independent opinion on a few samples as an extra sanity check.
    #
    # from deepface import DeepFace
    # print("\nDeepFace cross-check skipped unless enabled.")


if __name__ == "__main__":
    main()
