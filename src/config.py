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

    # Performance Tuning
    "performance_mode":  "AUTO",   # AUTO, LOW, BALANCED, HIGH
    "camera_res":        (640, 480),
    "camera_fps":        30,
    "inference_interval": 1,        # Run AI every N frames
    "landmark_interval":  1,        # Run MediaPipe every N frames
    "solve_pnp_interval": 2,        # Throttled head pose
    "jpeg_quality":      85,
    "max_queue_size":    1,        # Buffer size (1 = latest frame only)
    "benchmark_samples": 15,
    "warmup_cycles":     10,
    "log_metrics":       True,

    # Drooping Redesign (Fatigue Accumulator)
    "droop_neutral_zone":   15.0,   # Pitch dead-zone (Lowered)
    "droop_max_offset":     35.0,   # Reach max rate faster
    "droop_yaw_limit":      25.0,   # Ignore droop if looking sideways
    
    # Eye Monitoring (Final Repair)
    "eye_closure_warning":  1.2,    # Seconds of sustained closure for warning
    "eye_closure_urgent":   2.5,    # Seconds of sustained closure for urgent alert
    "mar_threshold":        0.50,   # Adaptive sensitivity for yawns
    
    # Interactive Calibration (Personal Bio-Profile)
    "calib_min_samples":    80,     # Reduced for faster capture
    "ear_warning_factor":   0.72,   # 72% of baseline
    "ear_urgent_factor":    0.60,   # 60% of baseline
    
    # Sideways/Diagonal Tuning
    "droop_roll_neutral":   8.0,    # Roll dead-zone (Lowered for sensitivity)
    "droop_roll_max":       35.0,   # Reach max lateral rate faster
    "droop_diagonal_bonus": 1.5,    # Multiplier for combined pitch+roll
    
    "droop_inc_mild":       220.0,  # Accumulator points per second (mild)
    "droop_inc_heavy":      450.0,  # Accumulator points per second (heavy)
    "droop_dec_upright":    -600.0, # Faster reduction when sitting up
    "droop_dec_looking":    -250.0, # Reduction per second when checking mirrors
    "droop_alert_threshold": 800.0, # 0-1000 scale
    "droop_recovery_level":  450.0, # Level to stop alert (hysteresis)
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
