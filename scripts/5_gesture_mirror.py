import cv2
import numpy as np
import mediapipe as mp
from helpers import (
    initialize_hands,
    initialize_camera,
    create_mirror_canvas,
    flip_frame
)

def generate_cartoon_hand():
    """Create a cartoon hand image programmatically"""
    size = 300
    img = np.zeros((size, size, 4), dtype=np.uint8)  # RGBA image
    
    # Palm (circle)
    cv2.circle(img, (size//2, size//2), size//3, (255, 220, 180, 255), -1)
    
    # Fingers 
    # Thumb
    cv2.ellipse(img, (size//2 - size//5, size//2 + size//8), 
               (size//6, size//8), 45, 0, 180, (255, 200, 150, 255), -1)
    # Index
    cv2.rectangle(img, (size//2 + size//10, size//4), 
                 (size//2 + size//5, size//2), (255, 220, 180, 255), -1)
    # Middle
    cv2.rectangle(img, (size//2 + size//8, size//6), 
                 (size//2 + size//4, size//2), (255, 220, 180, 255), -1)
    # Ring
    cv2.rectangle(img, (size//2 + size//12, size//5), 
                 (size//2 + size//6, size//2), (255, 220, 180, 255), -1)
    # Pinky
    cv2.rectangle(img, (size//2, size//4), 
                 (size//2 + size//10, size//2), (255, 220, 180, 255), -1)
    
    # Smooth edges
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.resize(img, (200, 200))
    cv2.putText()
    return img

# Generate the hand image once at startup
cartoon_hand = generate_cartoon_hand()

def draw_cartoon_hand(canvas, landmarks):
    """Draw the generated cartoon hand that mimics real hand movements"""
    try:
        # Calculate hand center and rotation
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        dx = middle_tip.x - wrist.x
        dy = middle_tip.y - wrist.y
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Dynamic sizing based on hand width
        hand_width = abs(landmarks.landmark[5].x - landmarks.landmark[17].x)
        scale = min(max(hand_width * 3.5, 0.5), 1.2)  # Constrained range
        
        # Transform hand image
        h, w = cartoon_hand.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(cartoon_hand, new_size)
        
        # Rotate around center
        M = cv2.getRotationMatrix2D((new_size[0]//2, new_size[1]//2), angle, 1)
        rotated = cv2.warpAffine(resized, M, new_size)
        
        # Position on canvas
        canvas_h, canvas_w = canvas.shape[:2]
        x = int(wrist.x * canvas_w - new_size[0]//2)
        y = int(wrist.y * canvas_h - new_size[1]//2)
        
        # Draw with bounds checking and transparency
        y_start, y_end = max(0, y), min(canvas_h, y + new_size[1])
        x_start, x_end = max(0, x), min(canvas_w, x + new_size[0])
        
        if y_end > y_start and x_end > x_start:
            img_h, img_w = y_end-y_start, x_end-x_start
            alpha = rotated[:img_h, :img_w, 3]/255.0
            for c in range(3):
                canvas[y_start:y_end, x_start:x_end, c] = \
                    canvas[y_start:y_end, x_start:x_end, c]*(1-alpha) + \
                    rotated[:img_h, :img_w, c]*alpha
                    
        return canvas
        
    except Exception as e:
        print(f"Hand drawing error: {e}")
        return canvas

# Rest of your existing main() function remains the same
def main():
    hands = initialize_hands()
    cap = initialize_camera()
    
    cv2.namedWindow('Your Hand', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cartoon Mirror', cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = flip_frame(frame)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mirror_img = create_mirror_canvas()
        
        if results.multi_hand_landmarks:
            # Original hand view
            overlay = frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                overlay,
                results.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS
            )
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            # Cartoon mirror view
            mirror_img = draw_cartoon_hand(
                mirror_img,
                results.multi_hand_landmarks[0]
            )
        
        cv2.imshow('Your Hand', frame)
        cv2.imshow('Cartoon Mirror', mirror_img)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()