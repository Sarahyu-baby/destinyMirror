
# Destiny Mirror — AI Face Analyzer & Fortune Predictor 

## Overview

Destiny Mirror is an advanced facial analysis and fortune-prediction application built using Kivy, OpenCV, MediaPipe, and Machine Learning models (XGBoost + LightGBM).

The app captures a real-time webcam image, extracts facial landmarks, computes geometric ratios, predicts fortune indicators (Love, Wealth, Health, Social, etc.), and displays results through an interactive UI.


---
## Libraries Used

Kivy  
OpenCV  
MediaPipe  
NumPy  
Pandas  
Scikit-learn  
XGBoost  
LightGBM  
Joblib  

---
# Project Structure

```
DestinyMirror/
│
├── captures/                         # User-captured photos
├── celebrity_faces/                  # Raw celebrity face images
├── fortune_results/                  # Saved prediction results
├── processed_samples/                # Intermediate processed data (optional)
│
├── celebrity_face_features.csv       # Extracted facial features from celebrities
├── celebrity_labels.csv              # Raw labels for love/wealth/health/personality
├── destiny_labels.csv                # Cleaned & standardized label file
├── merged_celebrity_data.csv         # Final dataset (features + labels)
│
├── destiny_brain.pkl                 # Trained ML models packaged as AI brain
│
├── readme.md                         # Documentation file
│
├── eye_feature_extractor.py          # Extracts geometric eye-related features
├── face_analyzer.py                  # MediaPipe landmark detection + feature calculation
├── face_visualizer.py                # Debug tool: draw face mesh & ratios overlay
│
├── batch_process_faces.py            # Batch runs analyzers → generates features CSV
├── merge.py                          # Merges feature CSV with labels CSV
│
├── love_model.py                     # Dedicated love prediction model
├── othermodels.py                    # Wealth/Health/Personality models
├── train_and_save.py                 # Trains all models → exports destiny_brain.pkl
│
├── destiny_predictor.py              # Loads destiny_brain.pkl → performs prediction
├── screens.py                        # Kivy UI screens (Main, Result, Camera)
├── destinyMirror.py                  # Main application launcher
│
└── __pycache__/                      # Python cache                   
```

---

# Full Pipeline Overview

## 1. Feature Extraction (Celebrities)

```
celebrity_faces/  →  batch_process_faces.py
                         ├── eye_feature_extractor.py
                         ├── face_analyzer.py
                         └── face_visualizer.py
        ▼
celebrity_face_features.csv
```

---

## 2. Dataset Merge

```
celebrity_face_features.csv
        +
celebrity_labels.csv
        ▼
merge.py
        ▼
merged_celebrity_data.csv
```

---

## 3. Model Training

```
merged_celebrity_data.csv
        ▼
train_and_save.py
        ├── love_model.py
        └── othermodels.py
        ▼
destiny_brain.pkl
```

---

## 4. Runtime Prediction Pipeline

```
User Photo → face_analyzer.py + eye_feature_extractor.py + face_visualizer.py
        ▼
Feature Vector → destiny_predictor.py
        ▼
Model Output (Love / Wealth / Health / Personality...)
        ▼
Kivy UI → destinyMirror.py
```

---

# Installation
Step 1 — Install Python 3.10  
Step 2 — Install libraries:
```
pip install mediapipe opencv-python numpy pandas scikit-learn xgboost lightgbm kivy
```
Step 3 — macOS Fix for LightGBM (if needed):
```
brew install libomp
```
or:
```
conda install -c conda-forge libomp
```
---
## Usage Guide
Step 1 — Launch Application

python destinyMirror.py

Step 2 — Capture & Predict
1. Align your face: Position your face within the frame. Face the camera directly, keep your mouth naturally closed, and wear light makeup or no makeup.  
2. Press "CAPTURE & PREDICT"  
3. AI predictions appear as cards  

Step 3 — VIP Mode
VIP subscription unlocks:
- Tier-2 metrics
- Premium descriptions
- VIP-exclusive content labels

Step 4 — Save (VIP Only)
Press SAVE (VIP ONLY).
After confirming $5 charge, results are saved to:
- captures/
- fortune_results/

Step 5 — Go back for another prediction
- Click BACK to return to the camera interface to take another photo.
---
# Train the Models (Optional)

```
python batch_process_faces.py
python merge.py
python train_and_save.py
```

---
# Github Link
https://github.com/Sarahyu-baby/destinyMirror 

---
