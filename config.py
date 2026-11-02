"""
config.py
---------
Central configuration shared by every component (training, evaluation,
backend). Keeping all the "magic numbers" and paths in one place means the
pipeline is reproducible and easy to retune without hunting through files.
"""

from pathlib import Path

# --- Project paths -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"           # FER2013 csv lives here
MODELS_DIR = ROOT_DIR / "models"       # trained .h5 model + label map
DOCS_DIR = ROOT_DIR / "docs"           # generated plots (curves, confusion matrix)

FER2013_CSV = DATA_DIR / "fer2013.csv"
MODEL_PATH = MODELS_DIR / "emotion_cnn.h5"
HISTORY_PATH = MODELS_DIR / "history.json"

# --- Image / model parameters ------------------------------------------------
# FER2013 images are 48x48 grayscale. We keep that native resolution so we do
# not have to upsample noisy faces.
IMG_SIZE = 48
NUM_CLASSES = 7

# Canonical FER2013 label order (the integer label in the csv maps to this).
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Group the 7 categorical emotions into 3 valence levels so the tool also
# satisfies the "at least three states/levels of valence" requirement and can
# report a simpler positive/neutral/negative reading next to the fine label.
VALENCE_MAP = {
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sad": "negative",
    "happy": "positive",
    "surprise": "positive",
    "neutral": "neutral",
}

# --- Training hyperparameters ------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.10          # fraction of the training data held out for validation

# Predictions below this softmax confidence are flagged as "uncertain" at
# inference time. This is the runtime half of our validation strategy.
CONFIDENCE_THRESHOLD = 0.40
