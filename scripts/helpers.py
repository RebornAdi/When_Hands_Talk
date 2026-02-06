import cv2
import numpy as np
import mediapipe as mp
from config import HAND_DETECTION, CAMERA, MIRROR

def initialize_hands():
    """Initialize MediaPipe Hands with config settings"""
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=HAND_DETECTION['max_num_hands'],
        min_detection_confidence=HAND_DETECTION['min_detection_confidence'],
        min_tracking_confidence=HAND_DETECTION['min_tracking_confidence'],
        model_complexity=HAND_DETECTION['model_complexity']
    )

def initialize_camera():
    """Set up camera with configured settings"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
    return cap

def create_mirror_canvas():
    """Create styled canvas for mirror hand"""
    canvas = np.zeros((MIRROR['height'], MIRROR['width'], 3), dtype=np.uint8)
    canvas[:] = MIRROR['background_color']
    return canvas

def draw_stylized_hand(canvas, landmarks):
    """Draw hand with custom styling"""
    mp.solutions.drawing_utils.draw_landmarks(
        canvas,
        landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=MIRROR['landmark_style']['color'],
            thickness=MIRROR['landmark_style']['thickness'],
            circle_radius=MIRROR['landmark_style']['radius']
        ),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=MIRROR['connection_style']['color'],
            thickness=MIRROR['connection_style']['thickness']
        )
    )
    return canvas

def extract_landmarks(hand_landmarks):
    """Extract (x, y, z) coordinates from hand landmarks"""
    return [
        [landmark.x, landmark.y, landmark.z]
        for landmark in hand_landmarks.landmark
    ]

def flip_frame(frame):
    """Flip frame horizontally if configured"""
    return cv2.flip(frame, 1) if CAMERA['flip_horizontal'] else frame