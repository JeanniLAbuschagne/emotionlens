"""
model.py  (backend/)
--------------------
Defines the CNN architecture and a thin wrapper that loads the trained model
and runs predictions. Both training/train.py and the API import from here so
the architecture is defined in exactly one place.

The network is a compact VGG-style CNN: three convolutional blocks of
increasing width, batch-normalisation for stable training, max-pooling to
shrink the feature maps, and dropout to fight overfitting (FER2013 is noisy
and the classes are imbalanced, so regularisation matters). It is small
enough to train on a CPU in a couple of hours and reaches ~65% test accuracy,
which is competitive for FER2013.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import IMG_SIZE, NUM_CLASSES, EMOTIONS, VALENCE_MAP, MODEL_PATH


def build_model():
    """Construct and return an *untrained* Keras model."""
    # imported lazily so importing this module is cheap when TF is not needed
    from tensorflow.keras import layers, models, regularizers

    l2 = regularizers.l2(1e-4)
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same",
                          kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same",
                          kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.3)(x)
        return x

    x = conv_block(inp, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inp, out, name="emotion_cnn")


class EmotionClassifier:
    """Loads a trained model once and exposes a simple predict() method.

    Used by the FastAPI backend so the heavy model load happens a single time
    at server start-up rather than on every request.
    """

    def __init__(self, model_path=MODEL_PATH):
        from tensorflow.keras.models import load_model
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. "
                "Run `python training/train.py` first."
            )
        self.model = load_model(model_path)

    def predict(self, face_48x48):
        """Classify one preprocessed 48x48 grayscale face.

        Parameters
        ----------
        face_48x48 : np.ndarray, shape (48, 48), values in [0, 1]

        Returns
        -------
        dict with the dominant emotion, its valence, the confidence, a flag
        for low-confidence predictions, and the full probability distribution.
        """
        batch = face_48x48.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype("float32")
        probs = self.model.predict(batch, verbose=0)[0]

        top = int(np.argmax(probs))
        emotion = EMOTIONS[top]
        confidence = float(probs[top])

        from config import CONFIDENCE_THRESHOLD
        return {
            "emotion": emotion,
            "valence": VALENCE_MAP[emotion],
            "confidence": confidence,
            "uncertain": confidence < CONFIDENCE_THRESHOLD,
            "probabilities": {
                EMOTIONS[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
        }
