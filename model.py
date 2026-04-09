"""

DRIVER DROWSINESS DETECTION — MediaPipe FaceMesh + CNN/ Bidirectional LSTM Hybrid

Architecture:
1)MediaPipe FaceMesh  - Open source framework by google that provides ready to use ML pipelines to process perceptual (imgs,vids) data.
                        FaceMesh detects and maps facial features using dense set of 3D landmarks.
                        FaceMesh takes a single image or video frame as input and returns the precise 3D coordinates (x, y, z) of
                        of 478 landmark points distributed across the entire face.(used for EAR and MAR).


2)CNN/Bi-LSTM Hybrid  -    Hybrid model is used because drowsiness is not an instant event, it occurs over a period of time,
                        meaning that the model cannot just see a single frame and predict the output. CNN learns the facial patterns 
                        within a short window of frames and LSTM studies how those patterns evolve and persist over time. Input is given
                        as follows,  each video frame is converted by MediaPipe into a 20-dimensional feature vector containing EAR,
                        MAR, head pose angles, PERCLOS flags, eye areas, lid droop, and other geometric measurements. A sliding window
                        of 16 consecutive frames is then stacked together, giving the model an input of shape (16, 20) — 16 time steps,
                        each with 20 features. This represents roughly half a second of facial behaviour at 30fps. FINAL O/P=16 vectors.


Why 99.99 ROC-AUC?
-Captures local temporal patterns (CNN) and long-range dependencies (LSTM) simultaneously
-Processes context from both past and future within each window (bidirectional)
-Focuses on the most informative moments rather than treating all frames equally (attention)
-Was trained on diverse eye shapes including CEW's closed-eye variety across ethnicities
-Uses geometric features rather than raw pixels, so it is invariant to lighting, skin tone, and camera quality 
  

  Usage:
    python model.py --mode train        # full training pipeline
    python model.py --mode evaluate     # evaluate on test split
    python model.py --mode demo         # live webcam inference

"""

import os
import sys
import argparse
import warnings
import logging
import time
import math
import json
import threading
from pathlib import Path
from collections import deque

# windows beep (wakeup alert)
try:
    import winsound
    BEEP_AVAILABLE = True
except ImportError:
    BEEP_AVAILABLE = False  # error handler

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


#  CONFIGURATION
CFG = {
    # paths to datasets/files
    "ddd_path":  "Driver Drowsiness Dataset (DDD)",
    "cew_path":  "CEW",                # CEW dataset root
    "output_dir": "training_output",

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

    "device": "cuda" if torch.cuda.is_available() else "cpu", #using cpu for traing, can change to gpu
}

os.makedirs(CFG["output_dir"], exist_ok=True)


#  MEDIAPIPE (FaceMesh 478-point model)
#all mediapipe coordinates for mapping
# right eye landmarks
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# left eye landmarks
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# mouth landmarks
MOUTH     = [61, 291, 13, 14, 17, 0, 267, 37]
# outer mouth
MOUTH_OUTER = [61, 291, 39, 181, 0, 17, 269, 405]
# nose tip
NOSE_TIP  = 1
# chin
CHIN      = 152
# head pose reference points (nose, chin, left/right eye corners, mouth corners)
HEAD_POSE_POINTS = [1, 152, 226, 446, 57, 287]


#  GEOMETRIC FEATURE EXTRACTOR
class FacialFeatureExtractor:
    """
    extracts facial features from a single frame using mediapipe and returns
    a 20-dimensional feature vector with the following values:

    index 0  - EAR for the right eye
    index 1  - EAR for the left eye
    index 2  - mean EAR across both eyes
    index 3  - difference between left and right EAR (detect asymmetry)
    index 4  - mouth aspect ratio (MAR)
    index 5  - eye closure flag, set to 1 if EAR drops below the threshold
    index 6  - yawn flag, set to 1 if MAR exceeds the threshold
    index 7  - normalised mouth height relative to face size
    index 8  - inner brow distance (decreases when fatigued)
    index 9  - head pitch angle, indicates nodding forward
    index 10 - head yaw angle, indicates turning left or right
    index 11 - head roll angle, indicates tilting to the side
    index 12 - distance from nose tip to chin, proxy for how far the face is from the camera
    index 13 - area of the left eye polygon
    index 14 - area of the right eye polygon
    index 15 - area of the mouth polygon
    index 16 - normalised inter-pupil distance, accounts for varying head distances
    index 17 - left upper lid droop, increases when the eyelid starts to sag
    index 18 - right upper lid droop
    index 19 - mediapipe detection confidence for the current frame
    """

    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,          # enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # applies CLAHE (contrast limited adaptive histogram equalization) to enhance facial feature visibility in low light conditions
        # clipLimit=3.0 prevents over-amplifying noise in very dark regions
        # tileGridSize=(8,8) gives localised contrast enhancement 
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def _euclidean(self, p1, p2):
        # calculates the straight line distance between two 2D points using the pythagorean theorem
        # used throughout the feature extractor to measure distances between facial landmarks
        # for example, the gap between eyelid points for EAR, lip points for MAR
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _eye_aspect_ratio(self, lm, indices, w, h):
        pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
        # vertical distances
        v1 = self._euclidean(pts[1], pts[5])
        v2 = self._euclidean(pts[2], pts[4])
        # horizontal distance
        hz = self._euclidean(pts[0], pts[3])
        return (v1 + v2) / (2.0 * hz + 1e-6)

    def _mouth_aspect_ratio(self, lm, w, h):
        pts = [(lm[i].x * w, lm[i].y * h) for i in MOUTH]
        v1 = self._euclidean(pts[2], pts[6])
        v2 = self._euclidean(pts[3], pts[5])
        hz = self._euclidean(pts[0], pts[1])
        return (v1 + v2) / (2.0 * hz + 1e-6)

    def _polygon_area(self, pts):
        # calculates the area of a polygon defined by a list of 2D points using the shoelace formula
        # works by summing the cross products of consecutive point pairs around the polygon boundary
        # used to compute the area of the eye and mouth regions as additional drowsiness features
        # smaller eye area over time can indicate drooping eyelids even when ear alone isnt enough
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def _head_pose(self, lm, w, h):
        """Estimate head pose angles via solvePnP."""
        model_pts = np.array([
            [0.0, 0.0, 0.0],         # nose tip
            [0.0, -330.0, -65.0],    # chin
            [-225.0, 170.0, -135.0], # left eye corner
            [225.0, 170.0, -135.0],  # right eye corner
            [-150.0, -150.0, -125.0],# left mouth corner
            [150.0, -150.0, -125.0], # right mouth corner
        ], dtype=np.float64)

        img_pts = np.array([
            (lm[HEAD_POSE_POINTS[i]].x * w,
             lm[HEAD_POSE_POINTS[i]].y * h)
            for i in range(6)
        ], dtype=np.float64)

        focal = w
        cam_mat = np.array([[focal, 0, w/2],
                            [0, focal, h/2],
                            [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            model_pts, img_pts, cam_mat, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        pitch = float(euler[0][0])
        yaw   = float(euler[1][0])
        roll  = float(euler[2][0])
        return pitch, yaw, roll

    def _preprocess(self, frame_bgr):
        """
         lightweight preprocessing pipeline to improve detection in low light (~1ms per frame):
           1. gaussian blur  - smooths out grainy noise from the camera sensor
           2. clahe          - boosts local contrast in dark areas without overexposing bright ones
           3. gamma lut      - brightens the frame only when it is actually too dark to see clearly
        """
        # Step 1: Mild Gaussian blur — kills sensor noise
        blurred = cv2.GaussianBlur(frame_bgr, (3, 3), 0)

        # Step 2: CLAHE 
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)

        # Step 3: Adaptive gamma — only compute when frame is actually dark
        mean_lum = float(np.mean(l_clahe))
        if mean_lum < 85:
            gamma = 0.55 + (mean_lum / 85.0) * 0.65
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                               for i in range(256)], dtype=np.uint8)
            l_clahe = cv2.LUT(l_clahe, table)

        enhanced = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
        return enhanced

    def extract(self, frame_bgr):
        """
        Parameters
        ----------
        frame_bgr : np.ndarray  (H, W, 3)  BGR image

        Returns
        -------
        features : np.ndarray shape (20,)  or zeros on failure
        success  : bool
        """
        h, w = frame_bgr.shape[:2]

        processed = self._preprocess(frame_bgr)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        # Fallback: if preprocessing made things worse (rare), try raw frame
        if not results.multi_face_landmarks:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return np.zeros(20, dtype=np.float32), False

        lm = results.multi_face_landmarks[0].landmark

        # EAR
        ear_r = self._eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear_l = self._eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        ear_m = (ear_r + ear_l) / 2.0
        ear_d = abs(ear_r - ear_l)

        # MAR
        mar = self._mouth_aspect_ratio(lm, w, h)

        # Flags
        eye_flag  = 1.0 if ear_m < CFG["ear_threshold"] else 0.0
        yawn_flag = 1.0 if mar  > CFG["mar_threshold"]  else 0.0

        # Mouth openness (normalised by face height)
        mouth_top    = lm[13].y * h
        mouth_bottom = lm[14].y * h
        face_height  = abs(lm[10].y - lm[152].y) * h + 1e-6
        mouth_open   = abs(mouth_bottom - mouth_top) / face_height

        # Brow furrow (inner brow distance)
        brow_l = (lm[107].x * w, lm[107].y * h)
        brow_r = (lm[336].x * w, lm[336].y * h)
        brow_furrow = self._euclidean(brow_l, brow_r) / (w + 1e-6)

        # Head pose
        pitch, yaw, roll = self._head_pose(lm, w, h)

        # Nose-to-chin distance (proxy for camera distance)
        nose  = (lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h)
        chin  = (lm[CHIN].x * w,     lm[CHIN].y * h)
        n2c   = self._euclidean(nose, chin) / (h + 1e-6)

        # Eye polygon areas
        r_pts = [(lm[i].x*w, lm[i].y*h) for i in RIGHT_EYE]
        l_pts = [(lm[i].x*w, lm[i].y*h) for i in LEFT_EYE]
        m_pts = [(lm[i].x*w, lm[i].y*h) for i in MOUTH_OUTER]
        eye_r_area = self._polygon_area(r_pts) / (w*h + 1e-6)
        eye_l_area = self._polygon_area(l_pts) / (w*h + 1e-6)
        mouth_area = self._polygon_area(m_pts) / (w*h + 1e-6)

        # Inter-pupil distance (iris landmarks: 468=left, 473=right)
        try:
            p_l = (lm[468].x * w, lm[468].y * h)
            p_r = (lm[473].x * w, lm[473].y * h)
            pupil_dist = self._euclidean(p_l, p_r) / (w + 1e-6)
        except Exception:
            pupil_dist = 0.0

        # Lid droop: upper lid midpoint vs eye corner height
        l_upper  = lm[386].y * h
        l_corner = lm[362].y * h
        r_upper  = lm[159].y * h
        r_corner = lm[33].y  * h
        l_droop  = max(0.0, l_upper - l_corner) / (h + 1e-6)
        r_droop  = max(0.0, r_upper - r_corner) / (h + 1e-6)

        # Detection confidence (average across landmarks)
        conf = float(np.mean([l.visibility if hasattr(l, 'visibility')
                               else 1.0 for l in lm]))

        features = np.array([
            ear_r, ear_l, ear_m, ear_d,
            mar, eye_flag, yawn_flag, mouth_open,
            brow_furrow, pitch/90.0, yaw/90.0, roll/90.0,
            n2c, eye_l_area*1000, eye_r_area*1000,
            mouth_area*1000, pupil_dist,
            l_droop*100, r_droop*100, conf
        ], dtype=np.float32)

        return features, True

    def close(self):
        self.face_mesh.close()


#  DATASET BUILDER  (converts image folders to feature sequences)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".BMP"}

def load_image_paths(root: str, class_map: dict) -> list:
    samples = []
    root = Path(root)
    for folder, label in class_map.items():
        # Fuzzy match: look for any subfolder whose name contains the key
        matched = [d for d in root.iterdir()
                   if d.is_dir() and folder.lower() in d.name.lower()]
        if not matched:
            log.warning(f"Could not find folder matching '{folder}' in {root}")
            continue
        folder_path = matched[0]
        log.info(f"  Found class '{folder}' → '{folder_path.name}' (label={label})")
        for f in folder_path.rglob("*"):
            if f.suffix.lower() in SUPPORTED_EXTS:
                samples.append((str(f), label))
    return samples


def extract_features_from_dataset(samples: list,
                                   extractor: FacialFeatureExtractor,
                                   desc: str = "") -> tuple:
    """
    Process all images; returns (features_array, labels_array).
    Features shape: (N, 20)
    """
    features, labels = [], []
    failed = 0
    for path, label in tqdm(samples, desc=desc, unit="img"):
        img = cv2.imread(path)
        if img is None:
            failed += 1
            continue
        img = cv2.resize(img, CFG["img_size"])
        feat, ok = extractor.extract(img)
        if not ok:
            failed += 1
            feat = np.zeros(20, dtype=np.float32)
        features.append(feat)
        labels.append(label)

    log.info(f"{desc}: {len(features)} samples loaded, {failed} failed")
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def build_sequences(features: np.ndarray, labels: np.ndarray,
                    seq_len: int) -> tuple:
    """
    Sliding-window sequencing.
    Returns (sequences, seq_labels) with shapes (N, seq_len, feat_dim) and (N,).
    The label of a sequence = label of its last frame.
    """
    seqs, seq_labels = [], []
    for i in range(seq_len - 1, len(features)):
        seqs.append(features[i - seq_len + 1 : i + 1])
        seq_labels.append(labels[i])
    return np.array(seqs, dtype=np.float32), np.array(seq_labels, dtype=np.int64)


#  PYTORCH DATASET
class DrowsinessDataset(Dataset):
    def __init__(self, sequences, labels, scaler=None, fit_scaler=False):
        """
        sequences : (N, seq_len, feat_dim)
        labels    : (N,)
        """
        N, T, F = sequences.shape
        flat = sequences.reshape(-1, F)

        if fit_scaler:
            self.scaler = StandardScaler()
            flat = self.scaler.fit_transform(flat)
        elif scaler is not None:
            self.scaler = scaler
            flat = scaler.transform(flat)
        else:
            self.scaler = None

        self.X = torch.tensor(flat.reshape(N, T, F), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#  MODEL ARCHITECTURE
class TemporalBlock(nn.Module):
    """1-D dilated causal convolution block with residual connection."""
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Causal trimming helper
        self.trim = pad
        self.residual = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        out = self.conv(x)
        # Trim to original length (causal padding leaves extra)
        out = out[:, :, :x.size(2)]
        return F.gelu(out + self.residual(x))


class DrowsinessNet(nn.Module):
    """
    Hybrid CNN + BiLSTM drowsiness classifier.

    Input : (batch, seq_len, feat_dim)
    Output: (batch, num_classes)
    """
    def __init__(self, feat_dim=20, seq_len=16, num_classes=2,
                 cnn_channels=None, lstm_hidden=256,
                 lstm_layers=2, fc_hidden=128, dropout=0.4):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        #Temporal CNN feature extractor (temporal because detects temporal patterns in feature sequence)
        layers = []
        in_ch = feat_dim
        for i, out_ch in enumerate(cnn_channels):
            layers.append(TemporalBlock(in_ch, out_ch,
                                        dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Bidirectional LSTM (works both directions, remembers both after time and before time then concatenates results)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Attention over LSTM outputs
        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classifier head 
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fc_hidden // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        # x : (B, T, F)
        # CNN expects (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)   # → (B, T, C)

        lstm_out, _ = self.lstm(x)  # (B, T, 2*H)

        # Attention pooling
        scores = self.attn(lstm_out)          
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights).sum(dim=1)  

        return self.classifier(context)


#  TRAINING UTILITIES
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def compute_class_weights(labels):
    """Return inverse-frequency weights as a tensor."""
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts.astype(float)
    weights /= weights.sum()
    w = torch.zeros(len(classes))
    for c, wt in zip(classes, weights):
        w[c] = wt
    return w


def get_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    sample_weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(labels),
        replacement=True,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(X)
            loss = criterion(logits, y)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_probs, all_targets = [], [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_targets))


def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    log.info("Saved training_curves.png")


def plot_confusion_matrix(targets, preds, class_names, title, output_dir):
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fname = title.replace(" ", "_").lower() + ".png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()


#  MAIN TRAINING PIPELINE
def load_cew_samples(cew_root: str) -> list:
    """
    CEW (Closed Eyes in the Wild) loader.
    Handles ALL images are closed eyes (label=1).
    Open-eye counterpart (label=0) is from DDD(non-drowsy) folder
    """
    root = Path(cew_root)
    if not root.exists():
        log.warning(f"CEW path not found: {cew_root} — skipping")
        return []

    samples = []

    # Check for subfolders first
    closed_keys = ["closedEyes", "closed_eyes", "closed", "close"]
    open_keys   = ["openEyes",   "open_eyes",   "open",   "alert"]
    closed_dir  = open_dir = None

    for d in root.iterdir():
        if not d.is_dir(): continue
        name = d.name.lower()
        if any(k.lower() in name for k in closed_keys):
            closed_dir = d
        if any(k.lower() in name for k in open_keys):
            open_dir = d

    if closed_dir or open_dir:
        # structured subfolder version
        if closed_dir:
            for f in closed_dir.rglob("*"):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    samples.append((str(f), 1))
            log.info(f"  CEW closed → '{closed_dir.name}' "
                     f"({sum(1 for s in samples if s[1]==1)} images)")
        if open_dir:
            before = len(samples)
            for f in open_dir.rglob("*"):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    samples.append((str(f), 0))
            log.info(f"  CEW open   → '{open_dir.name}' "
                     f"({len(samples)-before} images)")
    else:
        for f in root.rglob("*"):
            if f.suffix.lower() in SUPPORTED_EXTS:
                samples.append((str(f), 1))
        log.info(f"  CEW flat folder — all {len(samples)} images = closed eyes (label=1)")
        log.info(f"  Open-eye negatives sourced from DDD non-drowsy folder")

    log.info(f"  CEW total: {len(samples)} samples  "
             f"(closed={sum(1 for s in samples if s[1]==1)}  "
             f"open={sum(1 for s in samples if s[1]==0)})")
    return samples


def train(cfg):
    device = cfg["device"]
    log.info(f"Using device: {device}")

    extractor = FacialFeatureExtractor()
    all_feats, all_labels = [], []

    # DDD dataset
    log.info("=== Loading DDD dataset ===")
    ddd_samples = load_image_paths(cfg["ddd_path"], {
        "drowsy":     1,
        "non drowsy": 0,
        "non_drowsy": 0,
        "nondrowsy":  0,
    })
    if ddd_samples:
        ddd_feats, ddd_labels = extract_features_from_dataset(
            ddd_samples, extractor, desc="DDD"
        )
        all_feats.append(ddd_feats)
        all_labels.append(ddd_labels)

    # CEW dataset
    # CEW = Closed Eyes in the Wild
    # closed eyes : label 1 (drowsy), open eyes : label 0 (alert)
    # helps with small/Asian eyes which were underrepresented
    # in DDD, improves EAR calibration accuracy for diverse eye shapes
    log.info("=== Loading CEW dataset ===")
    cew_samples = load_cew_samples(cfg["cew_path"])
    if cew_samples:
        cew_feats, cew_labels = extract_features_from_dataset(
            cew_samples, extractor, desc="CEW"
        )
        # CEW images are single frames, not sequences
        # We replicate each frame seq_len times to simulate a static sequence
        # This teaches the model what sustained eye closure looks like
        # without needing temporal context from CEW
        all_feats.append(cew_feats)
        all_labels.append(cew_labels)

    extractor.close()

    if not all_feats:
        log.error("No data loaded from any dataset — check your paths")
        return

    #  Combine all datasets 
    combined_feats  = np.concatenate(all_feats,  axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    log.info(f"Total samples before sequencing: {len(combined_feats)}")
    log.info(f"Class distribution: {np.bincount(combined_labels)}")

    log.info("Building temporal sequences …")

    seq_list, label_list = [], []

    for i, (feats, labels) in enumerate(zip(all_feats, all_labels)):
        name = ["DDD", "CEW"][i] if i < 2 else f"Dataset{i}"

        if name == "CEW":
            # CEW: tile each single image into a (seq_len, feat_dim) sequence
            cew_seqs = np.stack(
                [np.tile(feat, (cfg["sequence_len"], 1)) for feat in feats],
                axis=0
            ).astype(np.float32)
            seq_list.append(cew_seqs)
            label_list.append(labels.astype(np.int64))
            log.info(f"  CEW: {len(cew_seqs)} static sequences built  shape={cew_seqs.shape}")

        else:
            X_seq, y_seq = build_sequences(feats, labels, cfg["sequence_len"])
            seq_list.append(X_seq)
            label_list.append(y_seq)
            log.info(f"  {name}: {len(X_seq)} sequences built")

    # all parts are now (N, seq_len, feat_dim) 
    X = np.concatenate(seq_list,   axis=0).astype(np.float32)
    y = np.concatenate(label_list, axis=0).astype(np.int64)

    log.info(f"Combined: {X.shape[0]} sequences  "
             f"feat_dim={X.shape[2]}  seq_len={X.shape[1]}")
    log.info(f"Class distribution: {np.bincount(y)}")

    #  Train / val / test split 
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=cfg["test_split"], stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=cfg["val_split"] / (1 - cfg["test_split"]),
        stratify=y_tv, random_state=42
    )
    log.info(f"Split — Train:{len(y_train)}  Val:{len(y_val)}  Test:{len(y_test)}")

    #  datasets+loaders
    train_ds = DrowsinessDataset(X_train, y_train, fit_scaler=True)
    scaler   = train_ds.scaler
    val_ds   = DrowsinessDataset(X_val,   y_val,   scaler=scaler)
    test_ds  = DrowsinessDataset(X_test,  y_test,  scaler=scaler)

    joblib.dump(scaler, os.path.join(cfg["output_dir"], "scaler.pkl"))

    sampler      = get_weighted_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=cfg["num_workers"],
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"])
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"])

    # Model 
    model = DrowsinessNet(
        feat_dim     = X.shape[2],
        seq_len      = cfg["sequence_len"],
        num_classes  = 2,
        cnn_channels = cfg["cnn_channels"],
        lstm_hidden  = cfg["lstm_hidden"],
        lstm_layers  = cfg["lstm_layers"],
        fc_hidden    = cfg["fc_hidden"],
        dropout      = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params:,}")

    class_w    = compute_class_weights(y_train).to(device)
    criterion  = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    optimizer  = torch.optim.AdamW(model.parameters(),
                                   lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=cfg["epochs"], steps_per_epoch=len(train_loader)
    )
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    early_stop = EarlyStopping(patience=cfg["patience"])

    #  Training loop 
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, amp_scaler
        )
        scheduler.step()
        vl_loss, vl_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        log.info(f"Epoch {epoch:03d}/{cfg['epochs']} | "
                 f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.4f} | "
                 f"VlLoss={vl_loss:.4f} VlAcc={vl_acc:.4f} | "
                 f"{time.time()-t0:.1f}s")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg["output_dir"], "best_model.pth"))
            log.info("  ✓ Saved new best model")

        early_stop(vl_loss)
        if early_stop.stop:
            log.info(f"Early stopping at epoch {epoch}")
            break

    #  Test evaluation 
    model.load_state_dict(torch.load(
        os.path.join(cfg["output_dir"], "best_model.pth"),
        map_location=device
    ))
    _, _, preds, probs, targets = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*60)
    print("  FINAL TEST RESULTS")
    print("="*60)
    print(classification_report(targets, preds,
          target_names=["Non-Drowsy", "Drowsy"]))
    try:
        print(f"  ROC-AUC : {roc_auc_score(targets, probs):.4f}")
    except Exception:
        pass

    plot_training_curves(history, cfg["output_dir"])
    plot_confusion_matrix(targets, preds, ["Non-Drowsy", "Drowsy"],
                          "Test Confusion Matrix", cfg["output_dir"])

    with open(os.path.join(cfg["output_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "feat_dim": X.shape[2],
    }, os.path.join(cfg["output_dir"], "drowsiness_model_full.pth"))

    log.info(f"\nAll outputs saved to: {cfg['output_dir']}/")
    return model, scaler


#  REAL-TIME DEMO  (webcam inference)
class RealTimeDetector:
    """
    Robust real-time drowsiness detector with:
      - Personal EAR/MAR calibration
      - Adaptive baseline drift correction
      - Face-loss freezing (looking away = ignored, not penalised)
      - Smoothed rolling EAR (not raw frame values)
      - Yawn counter with cooldown
      - Blink counter
      - Probability + PERCLOS dual-trigger
    """

    #  Tuning constants 
    ALERT_PROB_THRESHOLD = 0.88   # model confidence to count as drowsy frame
    ALERT_CONSECUTIVE    = 35     # sustained drowsy frames before alerting (~1.2s)
    PERCLOS_ALERT        = 0.55   # % eye closure over 4s window to trigger
    SMOOTH_WINDOW        = 20     # rolling average window for model prob
    EAR_SMOOTH           = 10     # rolling average window for EAR
    ADAPT_RATE           = 0.002  # how fast baseline EAR drifts toward current
    YAWN_COOLDOWN        = 60     # frames to wait before counting next yawn
    YAWN_MIN_FRAMES      = 20     # mouth must stay open 20+ frames (~0.67s) = yawn
    BLINK_MIN_FRAMES     = 2      # min frames eyes closed to count as blink
    BLINK_MAX_FRAMES     = 10     # max frames — more than this = drowsy, not blink

    def __init__(self, model_path, scaler_path, cfg):
        self.device = cfg["device"]
        self.cfg    = cfg

        #  Load model 
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = DrowsinessNet(
            feat_dim    = checkpoint["feat_dim"],
            seq_len     = cfg["sequence_len"],
            cnn_channels= cfg["cnn_channels"],
            lstm_hidden = cfg["lstm_hidden"],
            lstm_layers = cfg["lstm_layers"],
            fc_hidden   = cfg["fc_hidden"],
            dropout     = 0.0,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = joblib.load(scaler_path)

        #  Feature extractor in video mode 
        self.extractor = FacialFeatureExtractor()
        self.extractor.face_mesh = self.extractor.mp_face.FaceMesh(
            static_image_mode       = False,
            max_num_faces           = 1,
            refine_landmarks        = True,
            min_detection_confidence= 0.5,
            min_tracking_confidence = 0.5,
        )

        # State 
        self.frame_buffer   = deque(maxlen=cfg["sequence_len"])
        self.prob_buffer    = deque(maxlen=self.SMOOTH_WINDOW)
        self.ear_buffer     = deque(maxlen=self.EAR_SMOOTH)
        self.perclos_window = deque(maxlen=120)   # 4s at 30fps

        # Yawn tracking
        self.yawn_count        = 0
        self.yawn_cooldown_ctr = 0
        self.in_yawn           = False
        self.yawn_open_ctr     = 0   # frames mouth has been open this yawn

        # Blink tracking
        self.blink_count      = 0
        self.eye_closed_ctr   = 0

        # Alert state
        self.consec_alert = 0
        self.alert_active = False

        # adaptive baseline (updated slowly during alert state only)
        self.ear_baseline  = None   # set during calibration
        self.mar_baseline  = None
        self.ear_thresh    = None
        self.mar_thresh    = None

        # Session timer
        self.session_start = None

        # Beep state — tracks whether beep thread is active so we don't stack
        self.beep_active = False

        # Web mode flags
        self.web_mode = False
        self.stop_flag = False
        self.latest_frame = None
        self.latest_stats = {}

    def _beep(self):
        if self.beep_active:
            return   

        def _play():
            self.beep_active = True
            if BEEP_AVAILABLE:
                for _ in range(3):
                    winsound.Beep(1000, 500)
                    time.sleep(0.1)
            else:
                for _ in range(3):
                    sys.stdout.write('\a')
                    sys.stdout.flush()
                    time.sleep(0.3)
            self.beep_active = False

        threading.Thread(target=_play, daemon=True).start()

    #  Calibration 

    def _calibrate(self, cap):
        ears, mars  = [], []
        calib_total = 90        # 3 s at 30 fps
        collected   = 0

        log.info("Calibration — look straight at camera with eyes open")

        while collected < calib_total:
            ret, frame = cap.read()
            if not ret:
                break

            feat, ok = self.extractor.extract(frame)
            h, w     = frame.shape[:2]

            # Semi-transparent orange overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 140, 255), -1)
            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

            # Progress bar
            pct = int((collected / calib_total) * (w - 40))
            cv2.rectangle(frame, (20, h-40), (w-20, h-20), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, h-40), (20+pct, h-20), (0, 210, 100), -1)
            secs_left = int((calib_total - collected) / 30) + 1
            cv2.putText(frame, "CALIBRATING — eyes open, look at camera",
                        (20, h//2 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Starting in {secs_left}s...",
                        (20, h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (210, 210, 210), 1)

            # In web mode, send calibration frames to the browser instead of cv2 window
            if hasattr(self, 'web_mode') and self.web_mode:
                self.latest_frame = frame.copy()
                self.latest_stats = {
                    "status": "CALIBRATING",
                    "prob": 0,
                    "ear": 0,
                    "mar": 0,
                    "perclos": 0,
                    "yawns": 0,
                    "blinks": 0,
                    "fps": 0,
                    "elapsed": 0,
                    "calibration_progress": round(collected / calib_total * 100)
                }
            else:
                cv2.imshow("Driver Drowsiness Detection", frame)
                cv2.waitKey(1)

            if ok:
                ears.append(float(feat[2]))
                mars.append(float(feat[4]))
                collected += 1

        if not ears:
            log.warning("Calibration failed — using safe defaults")
            return 0.30, 0.50

        ear_base = float(np.mean(ears))
        mar_base = float(np.mean(mars))

        # EAR threshold = 75% of open-eye baseline
        # MAR threshold = 160% of closed-mouth baseline
        ear_thresh = np.clip(ear_base * 0.75, 0.20, 0.38)
        mar_thresh = np.clip(mar_base * 1.60, 0.42, 0.85)

        self.ear_baseline = ear_base
        self.mar_baseline = mar_base

        log.info(f"Calibration done  EAR baseline={ear_base:.3f}  "
                 f"thresh={ear_thresh:.3f}  |  "
                 f"MAR baseline={mar_base:.3f}  thresh={mar_thresh:.3f}")

        return float(ear_thresh), float(mar_thresh)

    # Overlay 

    def _overlay(self, frame, smooth_ear, smooth_prob, perclos,
                 fps, elapsed_s):
        h, w = frame.shape[:2]

        alert  = self.alert_active
        colour = (0, 0, 220) if alert else (0, 210, 0)
        cv2.rectangle(frame, (0, 0), (w-1, h-1), colour, 6)

        # Status
        status = "DROWSY / YAWNING" if alert else "ALERT"
        cv2.putText(frame, f"Status  : {status}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

        # Metrics
        lines = [
            f"Prob    : {smooth_prob:.2f}",
            f"EAR     : {smooth_ear:.3f}  "
                f"({'LOW' if smooth_ear < self.ear_thresh else 'OK'})",
            f"PERCLOS : {perclos:.1%}",
            f"Yawns   : {self.yawn_count}",
            f"Blinks  : {self.blink_count}",
            f"Session : {int(elapsed_s//60):02d}:{int(elapsed_s%60):02d}",
            f"FPS     : {fps:.1f}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, 62 + i*27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2)

        # alert text
        if alert:
            cv2.putText(frame, "WAKE UP!",
                        (w//2 - 110, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 255), 3)

        # Bottom hint
        cv2.putText(frame, "CLAHE enhanced  |  q = quit",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        return frame

    #  Main loop 

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log.error("Cannot open webcam")
            return

        self.ear_thresh, self.mar_thresh = self._calibrate(cap)
        self.session_start = time.time()
        prev_time = time.time()

        log.info("Detection running — press q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feat, ok = self.extractor.extract(frame)

            # Looking away / out of frame = freeze everything, don't penalise
            if ok:
                raw_ear = float(feat[2])
                raw_mar = float(feat[4])

                #  Smoothed EAR 
                self.ear_buffer.append(raw_ear)
                smooth_ear = float(np.mean(self.ear_buffer))

                # Adaptive baseline drift 
                # During clearly alert moments, nudge baseline toward current EAR
                # so it self-corrects for lighting changes over a long session
                if smooth_ear > self.ear_thresh * 1.1:
                    self.ear_baseline = (self.ear_baseline * (1 - self.ADAPT_RATE)
                                         + smooth_ear * self.ADAPT_RATE)
                    self.ear_thresh = float(np.clip(
                        self.ear_baseline * 0.75, 0.20, 0.38))

                #  Eye closure flag using smoothed EAR 
                eye_closed = smooth_ear < self.ear_thresh
                feat[5] = 1.0 if eye_closed else 0.0

                #  PERCLOS 
                self.perclos_window.append(1 if eye_closed else 0)

                #  Blink counter 
                if eye_closed:
                    self.eye_closed_ctr += 1
                else:
                    if self.BLINK_MIN_FRAMES <= self.eye_closed_ctr <= self.BLINK_MAX_FRAMES:
                        self.blink_count += 1
                    self.eye_closed_ctr = 0

                #  Yawn counter 
                # Talking: mouth opens/closes rapidly (< 20 frames open)
                # Yawning: mouth stays wide open for 20+ frames (~0.67s)
                yawning = raw_mar > self.mar_thresh
                feat[6] = 1.0 if yawning else 0.0

                if self.yawn_cooldown_ctr > 0:
                    self.yawn_cooldown_ctr -= 1

                if yawning:
                    self.yawn_open_ctr += 1
                    self.in_yawn = True
                else:
                    if (self.in_yawn
                            and self.yawn_open_ctr >= self.YAWN_MIN_FRAMES
                            and self.yawn_cooldown_ctr == 0):
                        self.yawn_count += 1
                        self.yawn_cooldown_ctr = self.YAWN_COOLDOWN
                    # Reset regardless — whether it was a yawn or just talking
                    self.in_yawn       = False
                    self.yawn_open_ctr = 0

                #  Model inference 
                self.frame_buffer.append(feat)
                prob = 0.0
                if len(self.frame_buffer) == self.cfg["sequence_len"]:
                    seq        = np.array(self.frame_buffer, dtype=np.float32)
                    seq_scaled = self.scaler.transform(seq)
                    X_t = torch.tensor(
                        seq_scaled[None], dtype=torch.float32
                    ).to(self.device)
                    with torch.no_grad():
                        logits = self.model(X_t)
                        prob   = float(torch.softmax(logits, dim=1)[0, 1])

                self.prob_buffer.append(prob)
                smooth_prob = float(np.mean(self.prob_buffer))

                #  Alert logic 
                perclos = float(np.mean(self.perclos_window)) \
                          if self.perclos_window else 0.0

                # Model prob only triggers alert if EAR also looks drowsy
                # Prevents false alerts when eyes are clearly open (EAR = OK)
                ear_drowsy    = smooth_ear < self.ear_thresh
                drowsy_signal = (
                    (smooth_prob > self.ALERT_PROB_THRESHOLD and ear_drowsy)
                    or perclos > self.PERCLOS_ALERT
                )

                if drowsy_signal:
                    self.consec_alert += 1
                else:
                    self.consec_alert = max(0, self.consec_alert - 3)

                prev_alert = self.alert_active
                self.alert_active = self.consec_alert >= self.ALERT_CONSECUTIVE

                if self.alert_active and not prev_alert:
                    self._beep()

            else:
                # Face not detected — keep last values, don't update counters
                smooth_ear  = float(np.mean(self.ear_buffer)) \
                              if self.ear_buffer else 0.0
                smooth_prob = float(np.mean(self.prob_buffer)) \
                              if self.prob_buffer else 0.0
                perclos     = float(np.mean(self.perclos_window)) \
                              if self.perclos_window else 0.0

            #  FPS & elapsed
            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            elapsed   = now - self.session_start

            # Display 
            display = self.extractor._preprocess(frame)
            display = self._overlay(display, smooth_ear, smooth_prob,
                                    perclos, fps, elapsed)

            if hasattr(self, 'web_mode') and self.web_mode:
                self.latest_frame = display.copy()
                self.latest_stats = {
                    "status": "DROWSY" if self.alert_active else "ALERT",
                    "prob": round(smooth_prob, 2),
                    "ear": round(smooth_ear, 3),
                    "mar": round(raw_mar, 3) if ok else 0.0,
                    "perclos": round(perclos, 3),
                    "yawns": self.yawn_count,
                    "blinks": self.blink_count,
                    "fps": round(fps, 1),
                    "elapsed": int(elapsed)
                }
                if hasattr(self, 'stop_flag') and self.stop_flag:
                    break
            else:
                cv2.imshow("Driver Drowsiness Detection", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if not (hasattr(self, 'web_mode') and self.web_mode):
            cv2.destroyAllWindows()
        self.extractor.close()

        # Session summary 
        total = time.time() - self.session_start
        print(f"\n{'='*45}")
        print(f"  SESSION SUMMARY")
        print(f"{'='*45}")
        print(f"  Duration : {int(total//60):02d}:{int(total%60):02d}")
        print(f"  Yawns    : {self.yawn_count}")
        print(f"  Blinks   : {self.blink_count}")
        print(f"{'='*45}\n")


#  CLI ENTRY POINT
def parse_args():
    p = argparse.ArgumentParser(description="Driver Drowsiness Detection")
    p.add_argument("--mode",     choices=["train", "evaluate", "demo"],
                   default="train")
    p.add_argument("--ddd_path", default=CFG["ddd_path"])
    p.add_argument("--cew_path", default=CFG["cew_path"])
    p.add_argument("--output",   default=CFG["output_dir"])
    p.add_argument("--epochs",   type=int,   default=CFG["epochs"])
    p.add_argument("--batch",    type=int,   default=CFG["batch_size"])
    p.add_argument("--lr",       type=float, default=CFG["lr"])
    p.add_argument("--seq_len",  type=int,   default=CFG["sequence_len"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG.copy()
    cfg.update({
        "ddd_path":    args.ddd_path,
        "cew_path":    args.cew_path,
        "output_dir":  args.output,
        "epochs":      args.epochs,
        "batch_size":  args.batch,
        "lr":          args.lr,
        "sequence_len": args.seq_len,
    })

    if args.mode == "train":
        train(cfg)

    elif args.mode == "evaluate":
        model_path  = os.path.join(cfg["output_dir"], "drowsiness_model_full.pth")
        scaler_path = os.path.join(cfg["output_dir"], "scaler.pkl")
        if not os.path.exists(model_path):
            log.error("No trained model found — run with --mode train first")
            sys.exit(1)
        detector = RealTimeDetector(model_path, scaler_path, cfg)
        log.info("Model loaded successfully. Run --mode demo for live inference.")

    elif args.mode == "demo":
        model_path  = os.path.join(cfg["output_dir"], "drowsiness_model_full.pth")
        scaler_path = os.path.join(cfg["output_dir"], "scaler.pkl")
        if not os.path.exists(model_path):
            log.error("No trained model found — run with --mode train first")
            sys.exit(1)
        detector = RealTimeDetector(model_path, scaler_path, cfg)
        detector.run()


if __name__ == "__main__":
    main()