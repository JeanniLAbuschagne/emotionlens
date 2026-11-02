# EmotionLens : Facial Emotion Detection

Detects the emotional state of a person from a photo of their face and reports
both a **categorical emotion** (7 classes) and a **valence level**
(positive / neutral / negative), together with a per-class confidence
breakdown. Built for the "Emotion detection in images" task.

The system classifies into the seven FER2013 emotions : **angry, disgust,
fear, happy, sad, surprise, neutral** : which comfortably satisfies the
requirement of at least three emotional states.

---

## 1. Architecture overview

```
 Browser UI  ──HTTP POST (image)──►  FastAPI gateway  ──►  Face detector (OpenCV)
 (frontend/)  ◄──── JSON result ────  (backend/api.py)        │  crop + 48×48 grayscale
                                            │                 ▼
                                            └────────►  CNN classifier (TensorFlow)
                                                          softmax over 7 emotions
```

Five components, each in its own module:

| Component | File | Responsibility |
|-----------|------|----------------|
| UI / client | `frontend/` | Upload image, draw result + probability bars |
| API gateway | `backend/api.py` | REST endpoint, orchestrates the pipeline |
| Preprocessing + detection | `backend/detector.py` | Detect face, crop, grayscale, resize, normalise |
| Classifier | `backend/model.py` | CNN inference → emotion + valence + confidence |
| Training | `training/train.py` | Train the CNN on FER2013 |
| **Validation** | `training/evaluate.py` | Accuracy, per-class F1, confusion matrix |

Browser ↔ server communication is plain **HTTP + JSON**, so the UI and the
model are decoupled and can be replaced independently.

---

## 2. Installation

### Prerequisites
- Python **3.10 or 3.11**
- ~2 GB free disk (TensorFlow + dataset)
- A CPU is sufficient; a GPU only speeds up training.

### Steps

```bash
# 1. create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. download the dataset
#    Get fer2013.csv from Kaggle:
#    https://www.kaggle.com/datasets/deadskull7/fer2013
#    and place it at:  data/fer2013.csv
```

---

## 3. Usage

### a) Train the model
```bash
python training/train.py
```
Saves the best model to `models/emotion_cnn.h5` and the training history to
`models/history.json`. On a CPU expect roughly 1–3 hours; on a GPU, minutes.

### b) Validate the model (required quality check)
```bash
python training/evaluate.py
```
Prints held-out test accuracy and a per-class report, and writes
`docs/confusion_matrix.png` and `docs/training_curves.png`.

### c) Run the application
```bash
uvicorn backend.api:app --reload --port 8000
```
Open <http://localhost:8000>, drop in a face photo, and read the result.
Interactive API docs are auto-generated at <http://localhost:8000/docs>.

### Quick API test (no UI)
```bash
curl -X POST -F "file=@some_face.jpg" http://localhost:8000/predict
```

---

## 4. How classification and validation work

**Classification.** The uploaded image is converted to grayscale; OpenCV's Haar
cascade locates the largest face; the face is cropped, resized to 48×48 and
scaled to [0, 1]. The CNN outputs a softmax probability for each of the seven
emotions. The highest probability is the predicted emotion; we also map it to a
valence level and flag predictions below a confidence threshold as uncertain.

**Validation.** `evaluate.py` measures accuracy on the official held-out
PrivateTest split (never used in training), reports per-class precision/recall/
F1, and renders a confusion matrix so class confusions (e.g. fear↔surprise) are
visible. It also compares against a majority-class baseline to confirm the
model learned real signal. At inference time the confidence threshold provides
a second, runtime layer of validation.

---

## 5. Project layout

```
emotionlens/
├── config.py              # shared paths + hyperparameters
├── requirements.txt
├── backend/
│   ├── api.py             # FastAPI gateway
│   ├── detector.py        # face detection + preprocessing
│   └── model.py           # CNN definition + inference wrapper
├── training/
│   ├── prepare_data.py    # load + split FER2013
│   ├── train.py           # train the CNN
│   └── evaluate.py        # validation: metrics + confusion matrix
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── data/                  # put fer2013.csv here
├── models/                # trained model lands here
└── docs/                  # generated plots
```

---

## 6. Frameworks used (with links)

- [Python](https://www.python.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/)
- [FER2013 dataset](https://www.kaggle.com/datasets/deadskull7/fer2013)

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not find data/fer2013.csv` | Download the dataset and place it in `data/`. |
| `No trained model at models/emotion_cnn.h5` | Run `python training/train.py` first. |
| `503` from `/predict` | The model isn't trained yet — train it. |
| Webcam/photo gives "no face" box | Use a clear, front-facing, well-lit photo. |
| TensorFlow install fails | Ensure Python is 3.10/3.11; upgrade pip. |

---

## 8. Limitations & ethics

FER2013 is noisy and imbalanced (very few `disgust` samples), so accuracy on
`disgust` and `fear` is lower than on `happy`. The tool reports a single facial
expression, which is **not** a reliable read of a person's true internal
emotion. It is intended for aggregate, anonymised marketing analysis, not for
judging individuals.
