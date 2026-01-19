import os
import io
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# =====================================================
# HARD LIMIT FIXES (CRITICAL FOR RAILWAY)
# =====================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI()

# =====================================================
# CONSTANTS
# =====================================================
EMOTION_LABELS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

YOLOV8_MODEL_PATH = "yolov8n-face-lindevs.pt"
EMOTION_MODEL_PATH = "FER_dinamic_LSTM_SAVEE.pt"

DEVICE = torch.device("cpu")

# =====================================================
# GLOBAL MODELS (LAZY LOADED)
# =====================================================
face_detector = None
emotion_model = None
feature_mapper = None
models_loaded = False

# =====================================================
# MODEL DEFINITION
# =====================================================
class EmotionLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(512, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 7)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        return self.fc(x)

# =====================================================
# MODEL LOADER (RUNS ON FIRST REQUEST ONLY)
# =====================================================
def load_models():
    global face_detector, emotion_model, feature_mapper, models_loaded

    if models_loaded:
        return

    print("Loading models...")

    from ultralytics import YOLO  # DELAYED IMPORT (IMPORTANT)

    face_detector = YOLO(YOLOV8_MODEL_PATH)

    emotion_model = EmotionLSTMModel().to(DEVICE)
    emotion_model.load_state_dict(
        torch.load(EMOTION_MODEL_PATH, map_location=DEVICE)
    )
    emotion_model.eval()

    feature_mapper = nn.Linear(48 * 48, 512).to(DEVICE)

    models_loaded = True
    print("Models loaded successfully")

# =====================================================
# HEALTH CHECK (FAST BOOT)
# =====================================================
@app.get("/")
def health():
    return {"status": "ok"}

# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    load_models()

    try:
        # ---------------------------------------------
        # READ IMAGE
        # ---------------------------------------------
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # ---------------------------------------------
        # FACE DETECTION
        # ---------------------------------------------
        results = face_detector(image)

        largest_face = None
        max_area = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_face = (x1, y1, x2, y2)

        if largest_face is None:
            return {"error": "No face detected"}

        x1, y1, x2, y2 = largest_face
        face = image[y1:y2, x1:x2]

        # ---------------------------------------------
        # PREPROCESS
        # ---------------------------------------------
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        face = torch.tensor(face, dtype=torch.float32, device=DEVICE) / 255.0
        face = face.view(1, -1)  # (1, 2304)

        # ---------------------------------------------
        # FEATURE MAP → LSTM
        # ---------------------------------------------
        mapped = feature_mapper(face)
        model_input = mapped.unsqueeze(1)  # (1,1,512)

        # ---------------------------------------------
        # PREDICT
        # ---------------------------------------------
        with torch.no_grad():
            logits = emotion_model(model_input)
            probs = F.softmax(logits, dim=1)
            emotion = EMOTION_LABELS[
                torch.argmax(probs, dim=1).item()
            ]

        return {"emotion": emotion}

    except Exception as e:
        return {"error": str(e)}

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
