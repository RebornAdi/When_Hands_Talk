👐 When Hands Talk
Real-Time Hand Sign Alphabet Recognition System
📌 Project Overview

When Hands Talk is a real-time computer vision–based hand sign recognition system developed to assist deaf and speech-impaired individuals in basic communication. The application recognizes hand sign alphabets through a webcam and instantly converts them into readable text.

The system leverages MediaPipe for high-precision hand landmark detection and a Support Vector Machine (SVM) classifier implemented using Scikit-learn for gesture classification. By combining machine learning with real-time video processing, the project demonstrates an accessible and scalable assistive technology solution.

🎯 Objectives

Enable real-time recognition of hand sign alphabets

Support assistive communication technologies

Provide a modular pipeline for data collection, training, and deployment

Demonstrate practical application of computer vision and machine learning

✨ Key Features

✅ Real-time hand tracking using webcam input

✅ Alphabet-level hand sign recognition

✅ Machine Learning–based classification (SVM)

✅ Custom dataset creation pipeline

✅ Modular and extensible architecture

✅ Lightweight and hardware-efficient implementation

🧠 Technology Stack
Category	Technologies Used
Programming Language	Python
Computer Vision	OpenCV, MediaPipe
Machine Learning	Scikit-learn (SVM)
Numerical Computing	NumPy
Model Storage	Pickle
Hardware	Standard Webcam

📁 Project Architecture
hand-gesture-recognition/
│
├── data/                # Dataset samples (A–D gestures)
├── models/              # Saved trained models
├── scripts/             # Core pipeline scripts
│   ├── 1_hand_detection.py
│   ├── 2_data_collection.py
│   ├── 3_model_training.py
│   ├── 4_sign_recognition.py
│   ├── helpers.py
│   └── config.py
│
├── requirements.txt
└── README.md

🔮 Future Enhancements

Full alphabet recognition (A–Z)

Text-to-Speech (TTS) integration

Indian Sign Language (ISL) support

Word and sentence-level recognition

Deep learning–based gesture classification (CNN/LSTM)

Mobile or web deployment
