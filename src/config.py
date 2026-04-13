import os
import torch

CFG = {
    # paths to datasets/files
    "ddd_path":    os.path.join("data", "raw", "Driver Drowsiness Dataset (DDD)"),
    "cew_path":    os.path.join("data", "raw", "CEW"),
    "output_dir":  "models",
    "reports_dir": os.path.join("models", "reports"),

    # feature extraction
    "img_size":        (224, 224),
    "sequence_len":    16,
    "ear_threshold":   0.25,
    "mar_threshold":   0.60,

    # training
    "epochs":          40,
    "batch_size":      32,
    "lr":              3e-4,
    "weight_decay":    1e-4,
    "dropout":         0.4,
    "val_split":       0.15,
    "test_split":      0.10,
    "patience":        8,
    "num_workers":     2,

    # dimensions
    "landmark_features": 20,
    "cnn_channels":  [64, 128, 256],
    "lstm_hidden":   256,
    "lstm_layers":   2,
    "fc_hidden":     128,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(CFG["output_dir"], "face_landmarker.task")

# Mediapipe constants
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 291, 13, 14, 17, 0, 267, 37]
MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]
NOSE_TIP  = 1
CHIN      = 152
HEAD_POSE_POINTS = [1, 152, 226, 446, 57, 287]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".BMP"}
