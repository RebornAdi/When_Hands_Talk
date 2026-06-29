"""
config.py
Central configuration for the whole pipeline. Every script imports from here
so you only ever need to change settings in one place.
"""

import string

# -----------------------------------------------------------------------------
# Gesture labels: A-Z plus two control gestures used for sentence assembly.
# NOTE: J and Z are motion-based in real ASL. This system classifies a single
# static landmark snapshot, so J/Z here are treated as static approximations
# (a common simplification in fingerspelling-recognition projects). Be upfront
# about this scope when describing the project.
# -----------------------------------------------------------------------------
GESTURE_LABELS = {i: letter for i, letter in enumerate(string.ascii_uppercase)}
GESTURE_LABELS[26] = "SPACE"   # suggested gesture: open flat palm, held still
GESTURE_LABELS[27] = "DELETE"  # suggested gesture: closed fist

NUM_GESTURES = len(GESTURE_LABELS)  # 28

SAMPLES_PER_GESTURE = 300  # bump to 400-500 for visually similar letters (M/N/S, A/E/T)

# -----------------------------------------------------------------------------
# Training parameters
# -----------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 10

# -----------------------------------------------------------------------------
# MediaPipe parameters
# -----------------------------------------------------------------------------
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 1

HAND_DETECTION = {
    "max_num_hands": MAX_NUM_HANDS,
    "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
    "min_tracking_confidence": MIN_TRACKING_CONFIDENCE,
    "model_complexity": 1,
}

CAMERA = {
    "width": 960,
    "height": 720,
    "flip_horizontal": True,
}

MIRROR = {
    "width": 480,
    "height": 480,
    "background_color": (20, 20, 20),
    "landmark_style": {"color": (0, 255, 170), "thickness": 2, "radius": 3},
    "connection_style": {"color": (255, 255, 255), "thickness": 2},
}

# -----------------------------------------------------------------------------
# Prediction smoothing (per-frame jitter reduction)
# -----------------------------------------------------------------------------
SMOOTHING_ALPHA = 0.6   # EWMA factor applied to raw landmarks
VOTE_WINDOW = 7          # majority vote window over recent predictions

# -----------------------------------------------------------------------------
# Sentence assembly (sign -> text)
# -----------------------------------------------------------------------------
STABLE_FRAMES_REQUIRED = 4     # consecutive matching predictions needed before a letter commits
COMMIT_COOLDOWN_SECONDS = 1.0  # min gap before the same letter can commit twice in a row
SPACE_LABEL = "SPACE"
DELETE_LABEL = "DELETE"
NO_HAND_LABEL = "NoHand"

# -----------------------------------------------------------------------------
# Model / artifact paths
# -----------------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_NAME = "gesture_model.pt"
SCALER_NAME = "gesture_scaler.pkl"
METRICS_NAME = "metrics.json"
CONFUSION_MATRIX_NAME = "confusion_matrix.png"
LABEL_MAP_NAME = "label_map.json"
