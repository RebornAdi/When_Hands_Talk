GESTURE_LABELS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

NUM_GESTURES = 4
SAMPLES_PER_GESTURE = 300  # recommended for accuracy

# Train parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 7

# MediaPipe parameters
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 1

# Smoothing parameters
SMOOTHING_ALPHA = 0.6
VOTE_WINDOW = 7

# Model paths
MODEL_DIR = "models"
MODEL_NAME = "gesture_model.pt"
SCALER_NAME = "gesture_scaler.pkl"
