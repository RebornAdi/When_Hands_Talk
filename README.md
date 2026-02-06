
# 👐 When Hands Talk

### Real-Time Hand Sign Recognition for Alphabets

## 📌 Overview

**When Hands Talk** is a real-time hand sign recognition system designed to assist **deaf and speech-impaired individuals** by recognizing hand sign **alphabets** through a webcam.

The system uses **MediaPipe** for precise hand landmark detection and a **Support Vector Machine (SVM)** classifier built with **Scikit-learn** to classify hand signs in real time. The recognized alphabet is displayed instantly on the screen, enabling basic assistive communication.

---

## ✨ Features

✅ Real-time hand sign detection via webcam
✅ Recognizes alphabets
✅ Designed for deaf & speech-impaired communication
✅ Custom data collection and training
✅ Machine learning–based classification (SVM)
✅ Modular and easy-to-extend codebase

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/RebornAdi/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
hand-gesture-recognition/
├── data/               # Collected samples for A, B, C, D
├── models/             # Trained ML models
├── scripts/            # Core execution scripts
│   ├── 1_hand_detection.py
│   ├── 2_data_collection.py
│   ├── 3_model_training.py
│   ├── 4_sign_recognition.py
│   ├── helpers.py
│   └── config.py  
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1: Test Hand Detection

```bash
python scripts/1_hand_detection.py
```

* Verifies webcam access
* Displays real-time hand landmarks
* Press **ESC** to exit

---

### Step 2: Collect Hand Sign Data

```bash
python scripts/2_data_collection.py
```

* Follow on-screen instructions
* Record samples for each alphabet

**Supported Alphabets:**

* `0` – A
* `1` – B
* `2` – C
* `3` – D

---

### Step 3: Train the Model

```bash
python scripts/3_model_training.py
```

* Trains an SVM classifier on the collected data
* Saves the trained model to:

  ```
  models/gesture_model.pkl
  ```

---

### Step 4: Real-Time Alphabet Recognition

```bash
python scripts/4_sign_recognition.py
```

* Detects and classifies hand signs in real time
* Displays the recognized alphabet (A–D)
* Press **ESC** to exit

---

## 🔧 Customization

### ➕ Modify Recognized Alphabets

Edit `config.py`:

```python
GESTURE_LABELS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}
NUM_GESTURES = 4
```

---

### 🎯 Improve Accuracy

* Increase `SAMPLES_PER_GESTURE` in `config.py`
* Maintain consistent lighting while recording
* Keep the hand centered in the camera frame

---

## 🛠 Troubleshooting

* **Webcam not detected?**
  Try `cv2.VideoCapture(1)` instead of `0`
* **Import errors?**
  Run scripts from the project root directory

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 💡 Future Enhancements

* Extend recognition to full alphabet (A–Z)
* Add **text-to-speech** output
* Support **Indian Sign Language (ISL)**
* Sentence-level sign recognition

---

### 👐 *When Hands Talk, Silence Speaks.*

If you want, I can now:

* Write a **perfect abstract** for your report
* Prepare **viva explanation points**
* Upgrade this to **A–Z without breaking structure**
