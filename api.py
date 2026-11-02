"""
api.py  (backend/)
------------------
The API-gateway component. A small FastAPI service that ties the pipeline
together and exposes it over HTTP:

    POST /predict   multipart image  ->  JSON {emotion, valence, confidence,...}
    GET  /health    liveness probe
    GET  /          serves the browser UI (frontend/index.html)

Communication contract between the browser and this service is plain
HTTP + JSON, so the UI and the model are fully decoupled and either side can
be replaced independently. The model is loaded ONCE at start-up.

Run from the project root:
    uvicorn backend.api:app --reload --port 8000
then open http://localhost:8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.detector import detect_and_preprocess, bytes_to_image
from backend.model import EmotionClassifier

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

app = FastAPI(title="EmotionLens API", version="1.0")

# allow the browser UI (and graders testing from elsewhere) to call the API
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded lazily on first use so the server can start even before training,
# and so importing this module (e.g. for tests) does not require the model.
_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = EmotionClassifier()
    return _classifier


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receive an image, run the full pipeline, return the emotion result."""
    raw = await file.read()
    try:
        image = bytes_to_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    face, box = detect_and_preprocess(image)

    try:
        result = get_classifier().predict(face)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result["face_detected"] = box is not None
    result["box"] = box  # (x, y, w, h) for drawing on the UI, or null
    return result


# serve static assets (app.js, style.css) and the index page
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
