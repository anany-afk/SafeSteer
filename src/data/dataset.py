import os
import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.config import CFG, SUPPORTED_EXTS
from src.data.extractor import FacialFeatureExtractor

log = logging.getLogger(__name__)

def load_image_paths(root: str, class_map: dict) -> list:
    samples = []
    root = Path(root)
    if not root.exists():
        log.warning(f"Root path not found: {root}")
        return []
    for folder, label in class_map.items():
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

def build_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int) -> tuple:
    seqs, seq_labels = [], []
    for i in range(seq_len - 1, len(features)):
        seqs.append(features[i - seq_len + 1 : i + 1])
        seq_labels.append(labels[i])
    return np.array(seqs, dtype=np.float32), np.array(seq_labels, dtype=np.int64)

def load_cew_samples(cew_root: str) -> list:
    root = Path(cew_root)
    if not root.exists():
        log.warning(f"CEW path not found: {cew_root} — skipping")
        return []
    samples = []
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
        if closed_dir:
            for f in closed_dir.rglob("*"):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    samples.append((str(f), 1))
        if open_dir:
            for f in open_dir.rglob("*"):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    samples.append((str(f), 0))
    else:
        for f in root.rglob("*"):
            if f.suffix.lower() in SUPPORTED_EXTS:
                samples.append((str(f), 1))
    return samples

class DrowsinessDataset(Dataset):
    def __init__(self, sequences, labels, scaler=None, fit_scaler=False):
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
