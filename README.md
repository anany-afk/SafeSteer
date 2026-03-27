# Driver Drowsiness Detection
### MediaPipe FaceMesh + CNN / Bidirectional LSTM Hybrid

A real-time driver drowsiness detection system that uses facial landmark geometry and a temporal deep learning model to detect drowsiness, eye closure, and yawning through a webcam feed.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 99.6% |
| Precision (Drowsy) | 0.99 |
| Recall (Drowsy) | 1.00 |
| F1-Score | 1.00 |
| ROC-AUC | 0.9999 |

---

## How It Works

### 1. MediaPipe FaceMesh
Every video frame is passed through Google's MediaPipe FaceMesh pipeline, which maps 478 3D facial landmarks onto the detected face. From these landmarks, 20 geometric features are extracted per frame including:

- **EAR** (Eye Aspect Ratio) — measures how open or closed the eyes are
- **MAR** (Mouth Aspect Ratio) — measures mouth opening for yawn detection
- **Head Pose** — pitch, yaw, and roll angles via solvePnP
- **PERCLOS** — percentage of eye closure over a 4-second window
- **Lid Droop** — upper eyelid sag as an early fatigue indicator
- **Eye and Mouth Area** — polygon areas for additional drowsiness context

### 2. CNN / BiLSTM Hybrid Model
A sliding window of 16 consecutive frames (≈0.5s at 30fps) is stacked into a (16, 20) input sequence and passed through:

- **Dilated Temporal CNN** — detects local patterns across time at multiple scales using exponentially increasing dilation (1, 2, 4)
- **Bidirectional LSTM** — models how drowsiness patterns evolve forward and backward in time
- **Attention Mechanism** — learns which frames in the window are most informative for the final prediction
- **Classifier Head** — outputs a drowsy/alert probability

### 3. Real-Time Demo
The live demo includes:
- **Personal calibration** — measures your individual EAR and MAR baseline over 3 seconds before monitoring begins
- **Adaptive threshold** — slowly updates your EAR baseline during the session to account for lighting changes
- **Yawn counter** — counts yawns only when the mouth stays open for 20+ consecutive frames, filtering out talking
- **Blink counter** — counts blinks between 2–10 frames, ignoring noise and drowsy episodes
- **PERCLOS monitoring** — independent eye closure metric as a secondary trigger
- **CLAHE preprocessing** — enhances low-light frames before landmark detection
- **Beep alert** — plays 3 audible beeps on drowsiness detection (Windows)
- **Session summary** — prints duration, yawn count, and blink count on exit

---

## Datasets Used

| Dataset | Description | Role |
|---------|-------------|------|
| [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) | Real driver images in drowsy and alert states | Primary training data |
| [CEW (Closed Eyes in the Wild)](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html) | Diverse closed-eye images across ethnicities | Improves small/Asian eye detection |

---

## Project Structure

```
Driver Drowsiness/
├── model.py                          # Main script
├── Driver Drowsiness Dataset (DDD)/  # DDD dataset (not included)
│   ├── Drowsy/
│   └── Non Drowsy/
├── CEW/                              # CEW dataset (not included)
└── training_output/                  # Generated after training
    ├── best_model.pth
    ├── drowsiness_model_full.pth
    ├── scaler.pkl
    ├── training_curves.png
    ├── test_confusion_matrix.png
    └── history.json
```

---

## Installation

```bash
pip install mediapipe==0.10.13 torch torchvision opencv-python scikit-learn matplotlib tqdm joblib
```

---

## Usage

**Train the model:**
```bash
python model.py --mode train
```

**Run live webcam demo:**
```bash
python model.py --mode demo
```

**Custom dataset paths:**
```bash
python model.py --mode train --ddd_path "path/to/DDD" --cew_path "path/to/CEW"
```

**All CLI options:**
```
--mode        train / demo / evaluate
--ddd_path    path to DDD dataset folder
--cew_path    path to CEW dataset folder
--output      path to save training outputs
--epochs      number of training epochs (default: 40)
--batch       batch size (default: 32)
--lr          learning rate (default: 3e-4)
--seq_len     sequence length in frames (default: 16)
```

---

## Architecture

```
Input: (16, 20)  ← 16 frames × 20 geometric features
        │
        ▼
Dilated Temporal CNN
  Block 1 (dilation=1) → 64 channels
  Block 2 (dilation=2) → 128 channels
  Block 3 (dilation=4) → 256 channels
        │
        ▼
Bidirectional LSTM (2 layers, hidden=256)
  Forward + Backward → 512-dim output per frame
        │
        ▼
Attention Pooling
  Weighted sum across 16 frames → 512-dim context
        │
        ▼
Classifier Head
  512 → 128 → 64 → 2 (Non-Drowsy / Drowsy)
```

---

## Why Not a Raw CNN?

A standard CNN classifies each frame independently with no memory of what came before. Drowsiness is a gradual process — eyes slowly droop, blink rate increases, head begins to nod. This system instead extracts interpretable geometric features using MediaPipe and feeds them into a temporal model that understands patterns over time, making it invariant to lighting conditions, skin tone, and camera quality.

---

## Dependencies

- `mediapipe` — facial landmark detection
- `torch` / `torchvision` — model training and inference
- `opencv-python` — video capture and preprocessing
- `scikit-learn` — data splitting, scaling, and evaluation metrics
- `matplotlib` — training curve and confusion matrix plots
- `tqdm` — progress bars during feature extraction
- `joblib` — scaler serialization
