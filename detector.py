"""
detector.py  (backend/)
-----------------------
The preprocessing + face-detection component. It turns an arbitrary uploaded
photo into the exact input the CNN expects: a single 48x48 grayscale,
intensity-normalised face.

We use OpenCV's Haar cascade detector. It is fast, ships with OpenCV (no extra
download), and runs comfortably on a CPU, which keeps the demo responsive. If
no face is found we fall back to centre-cropping the whole frame so the tool
still returns *something* rather than failing silently.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import IMG_SIZE

# Haar cascade bundled with the OpenCV install.
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_and_preprocess(image_bgr):
    """Detect the largest face and return the model-ready 48x48 face + box.

    Parameters
    ----------
    image_bgr : np.ndarray
        Image as read by OpenCV (BGR colour order).

    Returns
    -------
    (face_48x48, box) where
        face_48x48 : np.ndarray shape (48, 48), float32 in [0, 1]
        box        : (x, y, w, h) of the detected face, or None if we fell back
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = _CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    if len(faces) > 0:
        # pick the largest face (most likely the subject of the photo)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y + h, x:x + w]
        box = (int(x), int(y), int(w), int(h))
    else:
        # fallback: centre square crop of the whole image
        h0, w0 = gray.shape
        side = min(h0, w0)
        cx, cy = w0 // 2, h0 // 2
        face = gray[cy - side // 2:cy + side // 2,
                    cx - side // 2:cx + side // 2]
        box = None

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    return face, box


def bytes_to_image(raw_bytes):
    """Decode raw uploaded bytes into an OpenCV BGR image."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a readable image.")
    return img
