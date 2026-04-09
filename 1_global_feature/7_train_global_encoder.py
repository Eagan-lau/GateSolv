#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def matthews_corrcoef_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def best_threshold_by_mcc(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    thr_list = np.unique(prob)
    if thr_list.size == 0:
        return 0.5, 0.0
    best_thr = 0.5
    best_mcc = -1e9
    for thr in thr_list:
        pred = (prob >= thr).astype(int)
        mcc = matthews_corrcoef_binary(y_true, pred)
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_thr = float(thr)
    return best_thr, float(best_mcc)


class GlobalEncoder(nn.Module):
    def __init__(self, g_in: int, g_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(g_in),
            nn.Linear(g_in, g_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(g_hidden, g_hidden),
            nn.ReLU(),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


class GlobalClassifier(nn.Module):
    def __init__(self, g_in: int, g_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = GlobalEncoder(g_in=g_in, g_hidden=g_hidden, dropout=dropout)
        self.head = nn.Linear(g_hidden, 1)

    def forward(self, g: torch.Tensor):
        h = self.encoder(g)
        logits = self.head(h).squeeze(1)
        return logits, h


def load_merged(global_csv: str, label_csv: str):
    gdf = pd.read_csv(global_csv)
    ldf = pd.read_csv(label_csv)
    df = ldf.merge(gdf, on="PDB_ID", how="inner")
    if len(df) == 0:
        raise RuntimeError("No rows after merge on PDB_ID. Check IDs and file contents.")

    feature_cols = [c for c in df.columns if c not in ("PDB_ID", "y")]
    df = df.dropna(subset=feature_cols).copy()
    X = df[feature_cols].astype(np.float32).to_numpy()
    y = df["y"].astype(int).to_numpy()
    pids = df["PDB_ID"].astype(str).tolist()

    if np.any(~np.isfinite(X)):
        raise ValueError("Found NaN or Inf in global features. Fix upstream extraction first.")
    return df, X, y, pids, feature_cols


@torch.no_grad()
def predict_prob(model: GlobalClassifier, X: np.ndarray, device: torch.device, scaler: StandardScaler) -> np.ndarray:
    model.eval()
    Xs = scaler.transform(X).astype(np.float32)
    logits = model(torch.from_numpy(Xs).to(device))[0].detach().float().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-logits))
    return prob.astype(np.float64)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    mcc = matthews_corrcoef_binary(y_true, pred)
    precision, recall, f1 = precision_recall_f1(y_true, pred)
    return {
        "thr": float(threshold),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "prob_mean": float(prob.mean()),
        "prob_std": float(prob.std()),
    }


def save_predictions(path: str, pids, y_true, prob, threshold: float):
    pred = (prob >= threshold).astype(int)
    df = pd.DataFrame({
        "PDB_ID": list(pids),
        "y": np.asarray(y_true, dtype=int),
        "prob": np.asarray(prob, dtype=float),
        "pred": pred.astype(int),
    })
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Train a supervised global encoder and fix the final decision threshold on validation only."
    )
    parser.add_argument("--train_global_csv", required=True)
    parser.add_argument("--train_label_csv", required=True)
    parser.add_argument("--val_global_csv", required=True)
    parser.add_argument("--val_label_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--test_global_csv", default=None)
    parser.add_argument("--test_label_csv", default=None)

    parser.add_argument("--g_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    if device.type == "cuda" and (not args.no_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _, Xtr, ytr, pid_tr, feat_cols = load_merged(args.train_global_csv, args.train_label_csv)
    _, Xva, yva, pid_va, _ = load_merged(args.val_global_csv, args.val_label_csv)

    has_test = bool(args.test_global_csv and args.test_label_csv)
    if has_test:
        _, Xte, yte, pid_te, _ = load_merged(args.test_global_csv, args.test_label_csv)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)

    model = GlobalClassifier(g_in=Xtr_s.shape[1], g_hidden=args.g_hidden, dropout=args.dropout).to(device)

    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    Xtr_t = torch.from_numpy(Xtr_s)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    n = Xtr_t.shape[0]
    indices = np.arange(n)

    best_val_mcc = -1e9
    best_val_threshold = 0.5
    best_path_full = os.path.join(args.out_dir, "global_model_best.pt")
    best_path_encoder = os.path.join(args.out_dir, "global_encoder_best.pt")
    history = []
    bad_epochs = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        np.random.shuffle(indices)

        losses = []
        for start in range(0, n, args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            xb = Xtr_t[batch_idx].to(device)
            yb = ytr_t[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(xb)
                loss = criterion(logits, yb)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_prob = predict_prob(model, Xva, device, scaler)
        val_thr, val_mcc_at_best = best_threshold_by_mcc(yva, val_prob)
        val_metrics = compute_metrics(yva, val_prob, val_thr)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mcc": val_metrics["mcc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_thr": val_metrics["thr"],
            "val_prob_mean": val_metrics["prob_mean"],
            "val_prob_std": val_metrics["prob_std"],
        }
        history.append(record)

        print(
            f"[E{epoch:03d}] train_loss={train_loss:.4f} "
            f"VAL_MCC={val_metrics['mcc']:.4f} F1={val_metrics['f1']:.4f} "
            f"thr={val_metrics['thr']:.6f} prob_mean={val_metrics['prob_mean']:.3f}±{val_metrics['prob_std']:.3f}"
        )

        if val_mcc_at_best > best_val_mcc + 1e-6:
            best_val_mcc = float(val_mcc_at_best)
            best_val_threshold = float(val_thr)
            bad_epochs = 0

            common_payload = {
                "scaler_mean": scaler.mean_.astype(np.float32),
                "scaler_scale": scaler.scale_.astype(np.float32),
                "feat_cols": feat_cols,
                "epoch": epoch,
                "best_val_mcc": best_val_mcc,
                "best_val_threshold": best_val_threshold,
                "args": vars(args),
            }
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoder_state": model.encoder.state_dict(),
                    **common_payload,
                },
                best_path_full,
            )
            torch.save(
                {
                    "encoder_state": model.encoder.state_dict(),
                    **common_payload,
                },
                best_path_encoder,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[EARLY STOP] no improvement for {args.patience} epochs.")
                break

    history_csv = os.path.join(args.out_dir, "train_history.csv")
    pd.DataFrame(history).to_csv(history_csv, index=False)

    best_ckpt = torch.load(best_path_full, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    scaler.mean_ = best_ckpt["scaler_mean"].astype(np.float64)
    scaler.scale_ = best_ckpt["scaler_scale"].astype(np.float64)
    scaler.var_ = np.square(scaler.scale_)
    scaler.n_features_in_ = scaler.mean_.shape[0]

    final_threshold = float(best_ckpt["best_val_threshold"])

    val_prob = predict_prob(model, Xva, device, scaler)
    val_metrics_fixed = compute_metrics(yva, val_prob, final_threshold)
    save_predictions(os.path.join(args.out_dir, "val_predictions.csv"), pid_va, yva, val_prob, final_threshold)

    summary = {
        "best_val_mcc": float(best_ckpt["best_val_mcc"]),
        "best_val_threshold": final_threshold,
        "best_model_ckpt": best_path_full,
        "best_encoder_ckpt": best_path_encoder,
        "history_csv": history_csv,
        "n_features": int(Xtr_s.shape[1]),
        "val_metrics_fixed_threshold": val_metrics_fixed,
    }

    print(
        f"[BEST] VAL_MCC={val_metrics_fixed['mcc']:.4f} "
        f"F1={val_metrics_fixed['f1']:.4f} "
        f"precision={val_metrics_fixed['precision']:.4f} "
        f"recall={val_metrics_fixed['recall']:.4f} "
        f"thr={final_threshold:.6f}"
    )

    if has_test:
        test_prob = predict_prob(model, Xte, device, scaler)
        test_metrics_fixed = compute_metrics(yte, test_prob, final_threshold)
        summary["test_metrics_fixed_threshold"] = test_metrics_fixed
        save_predictions(os.path.join(args.out_dir, "test_predictions.csv"), pid_te, yte, test_prob, final_threshold)
        print(
            f"[TEST] MCC={test_metrics_fixed['mcc']:.4f} "
            f"F1={test_metrics_fixed['f1']:.4f} "
            f"precision={test_metrics_fixed['precision']:.4f} "
            f"recall={test_metrics_fixed['recall']:.4f} "
            f"thr={final_threshold:.6f}"
        )

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE] Global encoder training finished.")
    print(f"Best validation threshold (fixed for test): {final_threshold:.6f}")
    print(f"Encoder checkpoint: {best_path_encoder}")


if __name__ == "__main__":
    main()
