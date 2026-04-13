import os
import sys
import argparse
import logging
import joblib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import CFG
from src.data.extractor import FacialFeatureExtractor
from src.data.dataset import (load_image_paths, extract_features_from_dataset, 
                              build_sequences, load_cew_samples, DrowsinessDataset)
from src.models.architecture import DrowsinessNet
from src.core.trainer import (train_one_epoch, evaluate, compute_class_weights, 
                               get_weighted_sampler, plot_training_curves, plot_confusion_matrix)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Train Driver Drowsiness Detection Model")
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

    device = cfg["device"]
    log.info(f"Using device: {device}")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    extractor = FacialFeatureExtractor()
    all_feats, all_labels = [], []

    log.info("=== Loading DDD dataset ===")
    ddd_samples = load_image_paths(cfg["ddd_path"], {
        "drowsy": 1, "non drowsy": 0, "non_drowsy": 0, "nondrowsy": 0,
    })
    if ddd_samples:
        feats, labels = extract_features_from_dataset(ddd_samples, extractor, desc="DDD")
        all_feats.append(feats); all_labels.append(labels)

    log.info("=== Loading CEW dataset ===")
    cew_samples = load_cew_samples(cfg["cew_path"])
    if cew_samples:
        feats, labels = extract_features_from_dataset(cew_samples, extractor, desc="CEW")
        all_feats.append(feats); all_labels.append(labels)

    extractor.close()
    if not all_feats:
        log.error("No data loaded — check paths"); return

    seq_list, label_list = [], []
    for i, (feats, labels) in enumerate(zip(all_feats, all_labels)):
        name = ["DDD", "CEW"][i] if i < 2 else f"Dataset{i}"
        if name == "CEW":
            cew_seqs = np.stack([np.tile(feat, (cfg["sequence_len"], 1)) for feat in feats], axis=0).astype(np.float32)
            seq_list.append(cew_seqs); label_list.append(labels.astype(np.int64))
        else:
            X_s, y_s = build_sequences(feats, labels, cfg["sequence_len"])
            seq_list.append(X_s); label_list.append(y_s)

    X = np.concatenate(seq_list, axis=0).astype(np.float32)
    y = np.concatenate(label_list, axis=0).astype(np.int64)

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=cfg["test_split"], stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=cfg["val_split"]/(1-cfg["test_split"]), stratify=y_tv, random_state=42)

    train_ds = DrowsinessDataset(X_train, y_train, fit_scaler=True)
    scaler = train_ds.scaler
    val_ds = DrowsinessDataset(X_val, y_val, scaler=scaler)
    test_ds = DrowsinessDataset(X_test, y_test, scaler=scaler)
    joblib.dump(scaler, os.path.join(cfg["output_dir"], "scaler.pkl"))

    # Weighted Sampler Helper (Move to trainer.py if needed, keeping here for now)
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(y_train))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = DrowsinessNet(feat_dim=X.shape[2], seq_len=cfg["sequence_len"], num_classes=2, **{k:v for k,v in cfg.items() if k in ["cnn_channels", "lstm_hidden", "lstm_layers", "fc_hidden", "dropout"]}).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_train).to(device), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=len(train_loader))
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    
    from src.core.trainer import EarlyStopping
    early_stop = EarlyStopping(patience=cfg["patience"])
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_loss = float("inf")

    for epoch in range(1, cfg["epochs"]+1):
        tr_l, tr_a = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        scheduler.step()
        vl_l, vl_a, _, _, _ = evaluate(model, val_loader, criterion, device)
        for k, v in zip(history.keys(), [tr_l, vl_l, tr_a, vl_a]): history[k].append(v)
        log.info(f"Epoch {epoch:03d} | TrLoss={tr_l:.4f} TrAcc={tr_a:.4f} | VlLoss={vl_l:.4f} VlAcc={vl_a:.4f}")
        if vl_l < best_loss:
            best_loss = vl_l
            torch.save(model.state_dict(), os.path.join(cfg["output_dir"], "best_model.pth"))
        early_stop(vl_l)
        if early_stop.stop: break

    model.load_state_dict(torch.load(os.path.join(cfg["output_dir"], "best_model.pth"), map_location=device))
    _, _, preds, probs, targets = evaluate(model, test_loader, criterion, device)
    print("\nFINAL TEST RESULTS\n" + "-"*30)
    from sklearn.metrics import classification_report
    print(classification_report(targets, preds, target_names=["Non-Drowsy", "Drowsy"]))
    
    plot_training_curves(history, cfg["reports_dir"])
    plot_confusion_matrix(targets, preds, ["Non-Drowsy", "Drowsy"], "Test Confusion Matrix", cfg["reports_dir"])
    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "feat_dim": X.shape[2]}, os.path.join(cfg["output_dir"], "drowsiness_model_full.pth"))

if __name__ == "__main__":
    main()
