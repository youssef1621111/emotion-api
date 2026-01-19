import os
import io
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI()

# =====================================================
# CONSTANTS
# =====================================================
EMOTION_LABELS = [
    'Angry', 'Disgust', 'Fear',
    'Happy', 'Sad', 'Surprise', 'Neutral'
]

YOLOV8_MODEL_PATH = "yolov8n-face-lindevs.pt"
EMOTION_MODEL_PATH = "FER_dinamic_LSTM_SAVEE.pt"

# =====================================================
# MODEL DEFINITION (MATCHES STATE_DICT)
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
# LOAD MODELS ONCE (CRITICAL)
# =====================================================
print("Loading YOLO face model...")
face_detector = YOLO(YOLOV8_MODEL_PATH)

print("Loading emotion LSTM model...")
emotion_model = EmotionLSTMModel()
emotion_model.load_state_dict(
    torch.load(EMOTION_MODEL_PATH, map_location="cpu")
)
emotion_model.eval()

# Temporary feature mapper (same as notebook)
feature_mapper = nn.Linear(48 * 48, 512)

print("Models loaded successfully")

# =====================================================
# API ENDPOINT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # ---------------------------------------------
        # READ IMAGE FROM WEBSITE
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
        # PREPROCESS (NOTEBOOK LOGIC)
        # ---------------------------------------------
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        face = torch.tensor(face).float() / 255.0
        face = face.view(1, -1)  # (1, 2304)

        # ---------------------------------------------
        # FEATURE MAPPING + LSTM INPUT
        # ---------------------------------------------
        mapped = feature_mapper(face)
        model_input = mapped.unsqueeze(1)  # (1, 1, 512)

        # ---------------------------------------------
        # EMOTION PREDICTION
        # ---------------------------------------------
        with torch.no_grad():
            logits = emotion_model(model_input)
            probs = F.softmax(logits, dim=1)
            emotion = EMOTION_LABELS[torch.argmax(probs).item()]

        return {"emotion": emotion}

    except Exception as e:
        return {"error": str(e)}

# =====================================================
# RENDER ENTRY POINT (CRITICAL)
# =====================================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        workers=1
    )
