<<<<<<< HEAD
# Hand Gesture Recognition System  

## 📌 Overview  
This project is a real-time hand gesture recognition system that uses MediaPipe for hand tracking and Scikit-learn (SVM) for gesture classification. It can detect custom hand gestures via a webcam and display the recognized gesture in real-time.  

### Features  
✅ Real-time hand detection using MediaPipe  
✅ Custom gesture training (record your own gestures)  
✅ Machine learning model (SVM classifier)  
✅ Simple & modular code for easy customization  

---

## ⚙️ Setup & Installation  

### 1. Clone the Repository  
```bash
git clone https://github.com/RebornAdi/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Project Structure  
```
hand-gesture-recognition/
├── data/               # Stores recorded gesture samples
├── models/             # Stores trained ML models
├── scripts/            # Main scripts (run in order)
│   ├── 1_hand_detection.py
│   ├── 2_data_collection.py
│   ├── 3_model_training.py
│   |── 4_gesture_recognition.py
|   |── helpers.py
|   └── config.py  
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 How to Run  

### Step 1: Test Hand Detection  
```bash
python scripts/1_hand_detection.py
```
- Checks if your webcam works and detects hand landmarks.  
- Press ESC to exit.  

### Step 2: Record Gesture Data  
```bash
python scripts/2_data_collection.py
```
- Follow on-screen prompts to record gestures.  
- Default gestures:  
  - `0`: Open Hand  
  - `1`: Fist  
  - `2`: Thumbs Up  
  - `3`: Peace Sign  
  - `4`: OK Sign  

### Step 3: Train the Model  
```bash
python scripts/3_model_training.py
```
- Trains an SVM classifier on recorded gestures.  
- Saves model to `models/gesture_model.pkl`.  

### Step 4: Real-Time Gesture Recognition  
```bash
python scripts/4_gesture_recognition.py
```
- Detects gestures in real-time and displays predictions.  
- Press ESC to exit.  

---

## 🔧 Customization  

### 1. Add/Modify Gestures  
Edit `config.py` to change:  
```python
GESTURE_LABELS = {
    0: "Open Hand",
    1: "Fist",
    2: "Thumbs Up",
    3: "Peace Sign",
    4: "OK Sign",
    5: "New Gesture"
}
NUM_GESTURES = 6 
```

### 2. Improve Accuracy  
- Increase `SAMPLES_PER_GESTURE` (default: `50`) in `config.py`.  
- Try different classifiers (`RandomForest`, `KNN`) in `3_model_training.py`.  

### 3. Troubleshooting  
- Webcam not working? Check `cv2.VideoCapture(0)` (try `1` if using an external camera).  
- Import errors? Run scripts from the project root directory.  

---

## 📜 License  
This project is open-source under the MIT License.  

---

## 💡 Future Improvements  
- Add mouse control using gestures  
- Support multiple hand tracking  
- Deploy as a web application

---
# Hand-Gesture-Recognition-
# Hand-Gesture-Recognition-
=======
# Hand Gesture Recognition System  

## 📌 Overview  
This project is a real-time hand gesture recognition system that uses MediaPipe for hand tracking and Scikit-learn (SVM) for gesture classification. It can detect custom hand gestures via a webcam and display the recognized gesture in real-time.  

### Features  
✅ Real-time hand detection using MediaPipe  
✅ Custom gesture training (record your own gestures)  
✅ Machine learning model (SVM classifier)  
✅ Simple & modular code for easy customization  

---

## ⚙️ Setup & Installation  

### 1. Clone the Repository  
```bash
git clone https://github.com/RebornAdi/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Project Structure  
```
hand-gesture-recognition/
├── data/               # Stores recorded gesture samples
├── models/             # Stores trained ML models
├── scripts/            # Main scripts (run in order)
│   ├── 1_hand_detection.py
│   ├── 2_data_collection.py
│   ├── 3_model_training.py
│   |── 4_gesture_recognition.py
|   |── helpers.py
|   └── config.py  
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 How to Run  

### Step 1: Test Hand Detection  
```bash
python scripts/1_hand_detection.py
```
- Checks if your webcam works and detects hand landmarks.  
- Press ESC to exit.  

### Step 2: Record Gesture Data  
```bash
python scripts/2_data_collection.py
```
- Follow on-screen prompts to record gestures.  
- Default gestures:  
  - `0`: Open Hand  
  - `1`: Fist  
  - `2`: Thumbs Up  
  - `3`: Peace Sign  
  - `4`: OK Sign  

### Step 3: Train the Model  
```bash
python scripts/3_model_training.py
```
- Trains an SVM classifier on recorded gestures.  
- Saves model to `models/gesture_model.pkl`.  

### Step 4: Real-Time Gesture Recognition  
```bash
python scripts/4_gesture_recognition.py
```
- Detects gestures in real-time and displays predictions.  
- Press ESC to exit.  

---

## 🔧 Customization  

### 1. Add/Modify Gestures  
Edit `config.py` to change:  
```python
GESTURE_LABELS = {
    0: "Open Hand",
    1: "Fist",
    2: "Thumbs Up",
    3: "Peace Sign",
    4: "OK Sign",
    5: "New Gesture"
}
NUM_GESTURES = 6 
```

### 2. Improve Accuracy  
- Increase `SAMPLES_PER_GESTURE` (default: `50`) in `config.py`.  
- Try different classifiers (`RandomForest`, `KNN`) in `3_model_training.py`.  

### 3. Troubleshooting  
- Webcam not working? Check `cv2.VideoCapture(0)` (try `1` if using an external camera).  
- Import errors? Run scripts from the project root directory.  

---

## 📜 License  
This project is open-source under the MIT License.  

---

## 💡 Future Improvements  
- Add mouse control using gestures  
- Support multiple hand tracking  
- Deploy as a web application

---
>>>>>>> 682cb936ccc2b9b460890a589d8a4df024a7c330
