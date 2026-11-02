"""
train.py
--------
Trains the emotion-detection CNN on FER2013 and saves the model + training
history.

Run from the project root:
    python training/train.py

Design notes (the "why"):
  * Data augmentation (small rotations/shifts/flips) multiplies the effective
    training set and reduces overfitting on FER2013's ~29k training images.
  * class_weight counters the heavy imbalance (e.g. 'disgust' has ~50x fewer
    samples than 'happy'); without it the model ignores rare classes.
  * EarlyStopping + ReduceLROnPlateau stop us wasting epochs and gently lower
    the learning rate when validation loss plateaus.
  * The best model (by validation accuracy) is checkpointed, so an unlucky
    last epoch cannot ruin the saved weights.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH,
                    HISTORY_PATH, MODELS_DIR, EMOTIONS)
from backend.model import build_model
from training.prepare_data import load_fer2013


def main():
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                            ReduceLROnPlateau)
    from sklearn.utils.class_weight import compute_class_weight

    (x_train, y_train), (x_val, y_val), _ = load_fer2013()

    # ---- class weights to handle imbalance ---------------------------------
    y_int = np.argmax(y_train, axis=1)
    weights = compute_class_weight("balanced", classes=np.arange(len(EMOTIONS)),
                                   y=y_int)
    class_weight = {i: w for i, w in enumerate(weights)}
    print("class weights:", {EMOTIONS[i]: round(w, 2)
                             for i, w in class_weight.items()})

    # ---- augmentation -------------------------------------------------------
    datagen = ImageDataGenerator(
        rotation_range=12, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True,
    )
    datagen.fit(x_train)

    # ---- model --------------------------------------------------------------
    model = build_model()
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    MODELS_DIR.mkdir(exist_ok=True)
    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_val, y_val),
        epochs=EPOCHS, class_weight=class_weight, callbacks=callbacks,
    )

    # persist history so evaluate.py / the report can plot the curves
    with open(HISTORY_PATH, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)

    print(f"\nTraining complete. Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
