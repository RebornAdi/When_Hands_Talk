# app/api.py
import os
import io
import base64
import cv2
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import mediapipe as mp
from config import MODEL_DIR, MODEL_NAME, SCALER_NAME, GESTURE_LABELS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS, SMOOTHING_ALPHA, VOTE_WINDOW

app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------
# Load scaler + model
# -----------------------
scaler = joblib.load(os.path.join(MODEL_DIR, SCALER_NAME))

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = scaler.mean_.shape[0]
model = MLP(input_dim, len(GESTURE_LABELS))
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_NAME), map_location=device))
model.to(device)
model.eval()

# -----------------------
# MediaPipe setup
# -----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=MAX_NUM_HANDS,
                       min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                       min_tracking_confidence=MIN_TRACKING_CONFIDENCE)

# small helper to decode base64 jpeg
def decode_base64_image(base64_str):
    header, encoded = base64_str.split(',', 1) if ',' in base64_str else (None, base64_str)
    data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(data)).convert('RGB')
    arr = np.array(image)[:, :, ::-1]  # RGB -> BGR (cv2)
    return arr

def extract_landmarks_from_bgr(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    lm = np.array([[p.x, p.y, p.z] for p in hand.landmark]).flatten()
    if lm.shape != (63,):
        return None
    return lm

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    """
    Expects JSON: { "image": "data:image/jpeg;base64,..." }
    Returns: { label, confidence }
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "no image provided"}), 400

    try:
        b64 = data["image"]
        img = decode_base64_image(b64)
        lm = extract_landmarks_from_bgr(img)
        if lm is None:
            return jsonify({"label": "NoHand", "confidence": 0.0})

        # scale and predict
        feat = scaler.transform([lm])[0]
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            label = GESTURE_LABELS.get(idx, "Unknown")
            conf = float(probs[idx])

        return jsonify({"label": label, "confidence": conf})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # optionally set FLASK_ENV=development for debugging
    app.run(host="0.0.0.0", port=5000, debug=True)
