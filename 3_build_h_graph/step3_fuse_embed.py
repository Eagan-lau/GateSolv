#!/usr/bin/env python3
"""Step 3: gated fusion of graph and global embeddings.

This script fuses graph-level and global-level learned embeddings using a
lightweight gated module, trains a binary classifier, selects the best model on
validation, and evaluates test using the validation-selected threshold.

Key design choices:
- Validation threshold is selected on the validation split only and then fixed
  for test evaluation.
- Checkpoint selection is aligned with the requested model-selection metric.
- Graph/global CSV pairs are merged by intersection of PDB_ID to avoid silent
  inconsistencies while still allowing partially missing upstream samples.
- Pair-merge statistics and final threshold metadata are explicitly exported.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def configure_torch_accel(enable_tf32: bool = True) -> None:
    if torch.cuda.is_available() and enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def save_pdf(fig, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def find_prefix_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix='{prefix}'.")

    def _sort_key(col: str):
        try:
            return int(col.split("_")[-1])
        except Exception:
            return col

    return sorted(cols, key=_sort_key)


def safe_metric_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0


def safe_metric_prauc(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0


def best_threshold_by_mcc(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    thr_list = np.unique(prob)
    best_thr, best_mcc = 0.5, -1.0
    for thr in thr_list:
        pred = (prob >= thr).astype(int)
        mcc = matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else -1.0
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_thr = float(thr)
    return best_thr, best_mcc


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (prob >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0
    return {
        "MCC": float(mcc),
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "ROC_AUC": safe_metric_auc(y_true, prob),
        "PR_AUC": safe_metric_prauc(y_true, prob),
        "Thr": float(thr),
    }


def select_score(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name not in {"VAL_MCC", "VAL_PR_AUC", "VAL_ROC_AUC"}:
        raise ValueError(f"Unsupported selection metric: {metric_name}")
    return float(metrics[metric_name.replace("VAL_", "")])


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    pin_memory = device.type == "cuda"
    persistent_workers = pin_memory and (num_workers > 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )


@dataclass
class PairStats:
    graph_n: int
    global_n: int
    intersection_n: int
    graph_only_n: int
    global_only_n: int
    graph_only_preview: List[str]
    global_only_preview: List[str]

    def to_row(self, split: str) -> Dict[str, object]:
        return {
            "split": split,
            "graph_n": self.graph_n,
            "global_n": self.global_n,
            "intersection_n": self.intersection_n,
            "graph_only_n": self.graph_only_n,
            "global_only_n": self.global_only_n,
            "graph_only_preview": ", ".join(self.graph_only_preview),
            "global_only_preview": ", ".join(self.global_only_preview),
        }


def _validate_pair_tables(
    graph_df: pd.DataFrame,
    global_df: pd.DataFrame,
    graph_csv: str,
    global_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, PairStats]:
    required_cols = {"PDB_ID", "y"}
    if not required_cols.issubset(graph_df.columns):
        raise ValueError(f"Missing columns in graph CSV: {required_cols - set(graph_df.columns)}")
    if not required_cols.issubset(global_df.columns):
        raise ValueError(f"Missing columns in global CSV: {required_cols - set(global_df.columns)}")

    graph_df = graph_df.copy()
    global_df = global_df.copy()
    graph_df["PDB_ID"] = graph_df["PDB_ID"].astype(str)
    global_df["PDB_ID"] = global_df["PDB_ID"].astype(str)

    dup_graph = int(graph_df["PDB_ID"].duplicated().sum())
    dup_global = int(global_df["PDB_ID"].duplicated().sum())
    if dup_graph > 0:
        raise ValueError(f"Graph CSV contains {dup_graph} duplicated PDB_ID values: {graph_csv}")
    if dup_global > 0:
        raise ValueError(f"Global CSV contains {dup_global} duplicated PDB_ID values: {global_csv}")

    graph_ids = set(graph_df["PDB_ID"])
    global_ids = set(global_df["PDB_ID"])

    common_ids = graph_ids & global_ids
    missing_in_global = sorted(graph_ids - global_ids)
    missing_in_graph = sorted(global_ids - graph_ids)

    stats = PairStats(
        graph_n=len(graph_ids),
        global_n=len(global_ids),
        intersection_n=len(common_ids),
        graph_only_n=len(missing_in_global),
        global_only_n=len(missing_in_graph),
        graph_only_preview=missing_in_global[:10],
        global_only_preview=missing_in_graph[:10],
    )

    print(
        f"[INFO] Pair merge summary | graph={stats.graph_n} | global={stats.global_n} | "
        f"intersection={stats.intersection_n} | graph_only={stats.graph_only_n} | "
        f"global_only={stats.global_only_n}"
    )

    if len(common_ids) == 0:
        raise ValueError(
            "No overlapping PDB_ID values between graph/global CSVs. "
            f"graph_csv={graph_csv}, global_csv={global_csv}"
        )

    if missing_in_global:
        print(
            f"[WARN] {len(missing_in_global)} PDB_ID values exist only in graph CSV. "
            f"First 10: {missing_in_global[:10]}"
        )
    if missing_in_graph:
        print(
            f"[WARN] {len(missing_in_graph)} PDB_ID values exist only in global CSV. "
            f"First 10: {missing_in_graph[:10]}"
        )

    graph_df = graph_df[graph_df["PDB_ID"].isin(common_ids)].copy()
    global_df = global_df[global_df["PDB_ID"].isin(common_ids)].copy()

    merged_labels = graph_df[["PDB_ID", "y"]].merge(
        global_df[["PDB_ID", "y"]],
        on="PDB_ID",
        suffixes=("_graph", "_global"),
        how="inner",
        validate="one_to_one",
    )
    label_mismatch = merged_labels[merged_labels["y_graph"] != merged_labels["y_global"]]
    if not label_mismatch.empty:
        raise ValueError(
            f"Label mismatch detected for {len(label_mismatch)} PDB_ID values. "
            f"Preview: {label_mismatch.head(10).to_dict(orient='records')}"
        )

    return graph_df.reset_index(drop=True), global_df.reset_index(drop=True), stats


def load_pair_csv(
    h_graph_csv: str,
    h_global_csv: str,
    graph_prefix: str = "h_graph_",
    global_prefix: str = "h_global_",
) -> Tuple[pd.DataFrame, List[str], List[str], PairStats]:
    graph_df = pd.read_csv(h_graph_csv)
    global_df = pd.read_csv(h_global_csv)

    graph_df, global_df, stats = _validate_pair_tables(
        graph_df, global_df, h_graph_csv, h_global_csv
    )

    merged = graph_df.merge(
        global_df.drop(columns=["y"]),
        on="PDB_ID",
        how="inner",
        validate="one_to_one",
    )

    graph_cols = find_prefix_cols(merged, graph_prefix)
    global_cols = find_prefix_cols(merged, global_prefix)

    if len(graph_cols) != len(global_cols):
        raise ValueError(
            f"Embedding dimension mismatch: graph={len(graph_cols)} vs global={len(global_cols)}"
        )

    feat_cols = graph_cols + global_cols
    merged = merged.replace([np.inf, -np.inf], np.nan)

    bad_mask = merged[feat_cols].isna().any(axis=1)
    if bad_mask.any():
        bad_ids = merged.loc[bad_mask, "PDB_ID"].astype(str).tolist()
        raise ValueError(
            f"Found {int(bad_mask.sum())} samples with NaN/Inf features after merge. "
            f"First bad IDs: {bad_ids[:10]}"
        )

    if merged.empty:
        raise RuntimeError("Merged pair table is empty after taking intersection.")

    print(f"[INFO] Final merged pair table: n={len(merged)}")
    return merged.reset_index(drop=True), graph_cols, global_cols, stats


class PairEmbedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, graph_cols: List[str], global_cols: List[str]):
        self.ids = df["PDB_ID"].astype(str).tolist()
        self.y = df["y"].astype(int).to_numpy()
        self.hg = df[graph_cols].astype(np.float32).to_numpy()
        self.hl = df[global_cols].astype(np.float32).to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            self.ids[idx],
            torch.from_numpy(self.hg[idx]),
            torch.from_numpy(self.hl[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class ScalarGateFuse(nn.Module):
    def __init__(self, dim: int, use_ln: bool, dropout: float):
        super().__init__()
        self.ln_g = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.ln_l = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, hg: torch.Tensor, hl: torch.Tensor):
        hg2 = self.ln_g(hg)
        hl2 = self.ln_l(hl)
        gate = torch.sigmoid(self.gate(torch.cat([hg2, hl2], dim=-1)))
        hf = gate * hg2 + (1.0 - gate) * hl2
        return hf, gate


class VectorGateFuse(nn.Module):
    def __init__(self, dim: int, use_ln: bool, dropout: float):
        super().__init__()
        self.ln_g = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.ln_l = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, hg: torch.Tensor, hl: torch.Tensor):
        hg2 = self.ln_g(hg)
        hl2 = self.ln_l(hl)
        gate = torch.sigmoid(self.gate(torch.cat([hg2, hl2], dim=-1)))
        hf = gate * hg2 + (1.0 - gate) * hl2
        return hf, gate


class FusionModel(nn.Module):
    def __init__(self, dim: int, gate_type: str, use_ln: bool, dropout: float):
        super().__init__()
        if gate_type == "scalar":
            self.fuse = ScalarGateFuse(dim, use_ln=use_ln, dropout=dropout)
        elif gate_type == "vector":
            self.fuse = VectorGateFuse(dim, use_ln=use_ln, dropout=dropout)
        else:
            raise ValueError("gate_type must be 'scalar' or 'vector'.")
        self.classifier = nn.Linear(dim, 1)

    def forward(self, hg: torch.Tensor, hl: torch.Tensor):
        hf, gate = self.fuse(hg, hl)
        logit = self.classifier(hf).squeeze(-1)
        return logit, hf, gate


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.10
    max_epochs: int = 200
    patience: int = 25
    gate_type: str = "vector"
    use_ln: bool = True
    select_metric: str = "VAL_MCC"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    amp: bool = True
    tf32: bool = True


@torch.no_grad()
def predict_with_ckpt(
    ckpt_path: str,
    dataset: PairEmbedDataset,
    batch_size: int,
    device: str,
    num_workers: int,
    amp: bool,
    tf32: bool,
) -> Dict[str, np.ndarray]:
    device_obj = torch.device(device)
    configure_torch_accel(enable_tf32=tf32)
    checkpoint = torch.load(ckpt_path, map_location=device_obj)
    dim = int(checkpoint["dim"])
    ckpt_cfg = checkpoint.get("cfg", {})
    model = FusionModel(
        dim=dim,
        gate_type=ckpt_cfg.get("gate_type", "vector"),
        use_ln=bool(ckpt_cfg.get("use_ln", True)),
        dropout=float(ckpt_cfg.get("dropout", 0.10)),
    ).to(device_obj)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    loader = make_loader(dataset, batch_size, False, device_obj, num_workers)
    use_amp = bool(amp) and (device_obj.type == "cuda")

    ids, ys, logits, probs, gate_mean, gate_std = [], [], [], [], [], []
    fused_batches = []

    for pid, hg, hl, y in loader:
        hg = hg.to(device_obj, non_blocking=True)
        hl = hl.to(device_obj, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logit, hf, gate = model(hg, hl)

        logit_np = logit.detach().float().cpu().numpy()
        prob_np = torch.sigmoid(logit).detach().float().cpu().numpy()
        gate_np = gate.detach().float().cpu().numpy()
        hf_np = hf.detach().float().cpu().numpy()

        if gate_np.ndim == 2 and gate_np.shape[1] == 1:
            gm = gate_np[:, 0]
            gs = np.zeros_like(gm)
        else:
            gm = gate_np.mean(axis=1)
            gs = gate_np.std(axis=1)

        ids.extend(list(pid))
        ys.extend(y.numpy().astype(int).tolist())
        logits.extend(logit_np.tolist())
        probs.extend(prob_np.tolist())
        gate_mean.extend(gm.tolist())
        gate_std.extend(gs.tolist())
        fused_batches.append(hf_np)

    return {
        "PDB_ID": np.asarray(ids, dtype=str),
        "y": np.asarray(ys, dtype=int),
        "logit": np.asarray(logits, dtype=float),
        "prob": np.asarray(probs, dtype=float),
        "gate_mean": np.asarray(gate_mean, dtype=float),
        "gate_std": np.asarray(gate_std, dtype=float),
        "h_fused": np.vstack(fused_batches),
    }


def train_one_variant(
    name: str,
    cfg: TrainConfig,
    train_ds: PairEmbedDataset,
    val_ds: PairEmbedDataset,
    out_dir: str,
) -> Dict:
    ensure_dir(out_dir)
    device = torch.device(cfg.device)
    configure_torch_accel(enable_tf32=cfg.tf32)

    dim = int(train_ds.hg.shape[1])
    model = FusionModel(dim=dim, gate_type=cfg.gate_type, use_ln=cfg.use_ln, dropout=cfg.dropout).to(device)

    print(
        f"[INFO] Variant={name} | device={device} | "
        f"amp={cfg.amp and device.type == 'cuda'} | "
        f"tf32={cfg.tf32 and device.type == 'cuda'} | "
        f"select_metric={cfg.select_metric}"
    )
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    y_train = train_ds.y
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = make_loader(train_ds, cfg.batch_size, True, device, cfg.num_workers)
    val_loader = make_loader(val_ds, cfg.batch_size, False, device, cfg.num_workers)

    use_amp = bool(cfg.amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_path = os.path.join(out_dir, f"{name}.pt")
    best_score = -1e18
    best_epoch = 0
    best_thr = 0.5
    best_val_metrics: Optional[Dict[str, float]] = None
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mcc": [],
        "val_auc": [],
        "val_prauc": [],
        "val_thr_mcc": [],
        "selected_score": [],
    }

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses = []

        for _, hg, hl, y in train_loader:
            hg = hg.to(device, non_blocking=True)
            hl = hl.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logit, _, _ = model(hg, hl)
                loss = criterion(logit, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses, all_prob, all_y = [], [], []
        with torch.no_grad():
            for _, hg, hl, y in val_loader:
                hg = hg.to(device, non_blocking=True)
                hl = hl.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logit, _, _ = model(hg, hl)
                    loss = criterion(logit, y)
                val_losses.append(float(loss.detach().cpu().item()))
                all_prob.append(torch.sigmoid(logit).detach().float().cpu().numpy())
                all_y.append(y.detach().float().cpu().numpy())

        all_prob_np = np.concatenate(all_prob)
        all_y_np = np.concatenate(all_y).astype(int)

        finite_mask = np.isfinite(all_prob_np)
        if not np.all(finite_mask):
            bad_n = int((~finite_mask).sum())
            print(f"[WARN] Removed {bad_n} non-finite validation probabilities before scoring.")
            all_prob_np = all_prob_np[finite_mask]
            all_y_np = all_y_np[finite_mask]

        if all_prob_np.size == 0:
            raise RuntimeError("Validation probabilities are empty after filtering non-finite values.")

        thr_mcc, _ = best_threshold_by_mcc(all_y_np, all_prob_np)
        val_metrics = compute_metrics(all_y_np, all_prob_np, thr_mcc)
        score = select_score(val_metrics, cfg.select_metric)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        history["val_mcc"].append(val_metrics["MCC"])
        history["val_auc"].append(val_metrics["ROC_AUC"])
        history["val_prauc"].append(val_metrics["PR_AUC"])
        history["val_thr_mcc"].append(val_metrics["Thr"])
        history["selected_score"].append(score)

        if score > best_score + 1e-12:
            best_score = float(score)
            best_epoch = epoch
            best_thr = float(val_metrics["Thr"])
            best_val_metrics = dict(val_metrics)
            patience_counter = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "dim": dim,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "best_metric": cfg.select_metric,
                    "best_val_threshold": best_thr,
                    "best_val_metrics": best_val_metrics,
                },
                best_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    history_csv = os.path.join(out_dir, f"{name}_history.csv")
    pd.DataFrame(history).to_csv(history_csv, index=False)

    fig = plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} | loss curves")
    plt.legend()
    loss_pdf = os.path.join(out_dir, f"{name}_loss.pdf")
    save_pdf(fig, loss_pdf)

    return {
        "name": name,
        "best_ckpt": best_path,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_threshold": best_thr,
        "best_val_metrics": best_val_metrics,
        "history_csv": history_csv,
        "loss_pdf": loss_pdf,
        "cfg": asdict(cfg),
    }


def plot_roc_pr(y_true: np.ndarray, prob: np.ndarray, title_prefix: str, out_dir: str, tag: str) -> None:
    fig1 = plt.figure()
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title_prefix} | ROC")
    plt.legend()
    save_pdf(fig1, os.path.join(out_dir, f"{tag}_roc.pdf"))

    fig2 = plt.figure()
    if len(np.unique(y_true)) > 1:
        p, r, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        plt.plot(r, p, label=f"PR AUC={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} | PR")
    plt.legend()
    save_pdf(fig2, os.path.join(out_dir, f"{tag}_pr.pdf"))


def plot_gate_hist(gate_mean: np.ndarray, gate_std: np.ndarray, title_prefix: str, out_dir: str, tag: str) -> None:
    fig = plt.figure()
    plt.hist(gate_mean, bins=40, alpha=0.7, label="gate_mean")
    plt.hist(gate_std, bins=40, alpha=0.7, label="gate_std")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} | gate distribution")
    plt.legend()
    save_pdf(fig, os.path.join(out_dir, f"{tag}_gate.pdf"))


def export_variant(
    variant_name: str,
    ckpt_path: str,
    train_ds: PairEmbedDataset,
    val_ds: PairEmbedDataset,
    test_ds: PairEmbedDataset,
    batch_size: int,
    device: str,
    num_workers: int,
    amp: bool,
    tf32: bool,
    export_dir: str,
    val_selected_threshold: float,
) -> None:
    ensure_dir(export_dir)
    for split_name, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        out = predict_with_ckpt(ckpt_path, dataset, batch_size, device, num_workers, amp, tf32)
        dim = out["h_fused"].shape[1]
        df = pd.DataFrame(
            {
                "PDB_ID": out["PDB_ID"],
                "y": out["y"],
                "logit": out["logit"],
                "prob": out["prob"],
                "gate_mean": out["gate_mean"],
                "gate_std": out["gate_std"],
                "val_selected_threshold": float(val_selected_threshold),
                "pred_by_val_threshold": (out["prob"] >= float(val_selected_threshold)).astype(int),
            }
        )
        for j in range(dim):
            df[f"h_fused_{j}"] = out["h_fused"][:, j].astype(np.float32)
        df.to_csv(os.path.join(export_dir, f"{variant_name}.h_fused_{split_name}.csv"), index=False)

        aux = df[
            ["PDB_ID", "y", "prob", "gate_mean", "gate_std", "val_selected_threshold", "pred_by_val_threshold"]
        ].copy()
        aux.to_csv(os.path.join(export_dir, f"{variant_name}.fuse_aux_{split_name}.csv"), index=False)


def load_export_tables(export_dir: str, variant_name: str):
    val_df = pd.read_csv(os.path.join(export_dir, f"{variant_name}.h_fused_val.csv"))
    test_df = pd.read_csv(os.path.join(export_dir, f"{variant_name}.h_fused_test.csv"))
    pred_val = val_df[["PDB_ID", "y", "prob", "logit", "val_selected_threshold", "pred_by_val_threshold"]].copy()
    pred_test = test_df[["PDB_ID", "y", "prob", "logit", "val_selected_threshold", "pred_by_val_threshold"]].copy()
    gate_val = val_df[["PDB_ID", "y", "gate_mean", "gate_std"]].copy()
    gate_test = test_df[["PDB_ID", "y", "gate_mean", "gate_std"]].copy()
    return pred_val, gate_val, pred_test, gate_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3 gated fusion ablation (graph/global embeddings, GPU optimized: AMP + TF32)."
    )
    parser.add_argument("--h_graph_train", required=True)
    parser.add_argument("--h_global_train", required=True)
    parser.add_argument("--h_graph_val", required=True)
    parser.add_argument("--h_global_val", required=True)
    parser.add_argument("--h_graph_test", required=True)
    parser.add_argument("--h_global_test", required=True)
    parser.add_argument("--out_dir", default="step3_fuse_ablation_out")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--select_metric",
        default="VAL_MCC",
        choices=["VAL_MCC", "VAL_PR_AUC", "VAL_ROC_AUC"],
        help="Checkpoint and best-variant selection metric.",
    )
    parser.add_argument("--export_all_variants", action="store_true")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    print("[INFO] torch:", torch.__version__)
    print("[INFO] cuda available:", torch.cuda.is_available())
    print("[INFO] torch.version.cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    out_dir = args.out_dir
    fig_dir = os.path.join(out_dir, "fig_pdf")
    model_dir = os.path.join(out_dir, "models")
    export_dir = os.path.join(out_dir, "exports")
    for path in [out_dir, fig_dir, model_dir, export_dir]:
        ensure_dir(path)

    df_train, graph_cols, global_cols, train_stats = load_pair_csv(args.h_graph_train, args.h_global_train)
    df_val, _, _, val_stats = load_pair_csv(args.h_graph_val, args.h_global_val)
    df_test, _, _, test_stats = load_pair_csv(args.h_graph_test, args.h_global_test)

    pair_stats_df = pd.DataFrame(
        [
            train_stats.to_row("train"),
            val_stats.to_row("val"),
            test_stats.to_row("test"),
        ]
    )

    train_ds = PairEmbedDataset(df_train, graph_cols, global_cols)
    val_ds = PairEmbedDataset(df_val, graph_cols, global_cols)
    test_ds = PairEmbedDataset(df_test, graph_cols, global_cols)

    variants = [
        ("scalar_ln", "scalar", True),
        ("scalar_noln", "scalar", False),
        ("vector_ln", "vector", True),
        ("vector_noln", "vector", False),
    ]

    results_rows, configs_rows = [], []
    ckpt_map: Dict[str, str] = {}
    val_threshold_map: Dict[str, float] = {}

    for name, gate_type, use_ln in variants:
        cfg = TrainConfig(
            seed=args.seed,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            max_epochs=args.max_epochs,
            patience=args.patience,
            gate_type=gate_type,
            use_ln=use_ln,
            select_metric=args.select_metric,
            device=args.device,
            num_workers=args.num_workers,
            amp=(not args.no_amp),
            tf32=(not args.no_tf32),
        )

        run_dir = os.path.join(model_dir, name)
        ensure_dir(run_dir)

        train_info = train_one_variant(name, cfg, train_ds, val_ds, run_dir)
        ckpt_map[name] = train_info["best_ckpt"]
        val_threshold_map[name] = float(train_info["best_val_threshold"])

        configs_rows.append(
            {
                "variant": name,
                **train_info["cfg"],
                "best_ckpt": train_info["best_ckpt"],
                "best_epoch": train_info["best_epoch"],
                "best_score": train_info["best_score"],
                "best_val_threshold": train_info["best_val_threshold"],
            }
        )

        pred_val = predict_with_ckpt(
            train_info["best_ckpt"],
            val_ds,
            args.batch_size,
            cfg.device,
            cfg.num_workers,
            cfg.amp,
            cfg.tf32,
        )
        pred_test = predict_with_ckpt(
            train_info["best_ckpt"],
            test_ds,
            args.batch_size,
            cfg.device,
            cfg.num_workers,
            cfg.amp,
            cfg.tf32,
        )

        val_thr = float(train_info["best_val_threshold"])
        metrics_val = compute_metrics(pred_val["y"], pred_val["prob"], val_thr)
        metrics_test = compute_metrics(pred_test["y"], pred_test["prob"], val_thr)

        results_rows.append(
            {
                "variant": name,
                "VAL_MCC": metrics_val["MCC"],
                "VAL_F1": metrics_val["F1"],
                "VAL_ROC_AUC": metrics_val["ROC_AUC"],
                "VAL_PR_AUC": metrics_val["PR_AUC"],
                "VAL_Precision": metrics_val["Precision"],
                "VAL_Recall": metrics_val["Recall"],
                "VAL_Thr": metrics_val["Thr"],
                "TEST_MCC": metrics_test["MCC"],
                "TEST_F1": metrics_test["F1"],
                "TEST_ROC_AUC": metrics_test["ROC_AUC"],
                "TEST_PR_AUC": metrics_test["PR_AUC"],
                "TEST_Precision": metrics_test["Precision"],
                "TEST_Recall": metrics_test["Recall"],
            }
        )

        plot_roc_pr(pred_val["y"], pred_val["prob"], f"{name} (VAL)", fig_dir, f"{name}_val")
        plot_roc_pr(pred_test["y"], pred_test["prob"], f"{name} (TEST)", fig_dir, f"{name}_test")
        plot_gate_hist(pred_val["gate_mean"], pred_val["gate_std"], f"{name} (VAL)", fig_dir, f"{name}_val")
        plot_gate_hist(pred_test["gate_mean"], pred_test["gate_std"], f"{name} (TEST)", fig_dir, f"{name}_test")

        if args.export_all_variants:
            export_variant(
                name,
                train_info["best_ckpt"],
                train_ds,
                val_ds,
                test_ds,
                args.batch_size,
                cfg.device,
                cfg.num_workers,
                cfg.amp,
                cfg.tf32,
                export_dir,
                val_selected_threshold=val_thr,
            )

    metrics_df = pd.DataFrame(results_rows).sort_values(args.select_metric, ascending=False).reset_index(drop=True)
    configs_df = pd.DataFrame(configs_rows)

    best_variant = str(metrics_df.iloc[0]["variant"])
    best_ckpt = ckpt_map[best_variant]
    best_val_threshold = float(val_threshold_map[best_variant])

    export_variant(
        best_variant,
        best_ckpt,
        train_ds,
        val_ds,
        test_ds,
        args.batch_size,
        args.device,
        args.num_workers,
        (not args.no_amp),
        (not args.no_tf32),
        export_dir,
        val_selected_threshold=best_val_threshold,
    )

    excel_path = os.path.join(out_dir, "step3_fuse_ablation_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pair_stats_df.to_excel(writer, sheet_name="pair_merge_stats", index=False)
        configs_df.to_excel(writer, sheet_name="configs", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        pd.DataFrame(
            [
                {
                    "best_variant": best_variant,
                    "best_ckpt": best_ckpt,
                    "select_metric": args.select_metric,
                    "best_val_threshold": best_val_threshold,
                }
            ]
        ).to_excel(writer, sheet_name="best_variant", index=False)
        val_pred_df, val_gate_df, test_pred_df, test_gate_df = load_export_tables(export_dir, best_variant)
        val_pred_df.to_excel(writer, sheet_name="pred_val", index=False)
        test_pred_df.to_excel(writer, sheet_name="pred_test", index=False)
        val_gate_df.to_excel(writer, sheet_name="gate_stats_val", index=False)
        test_gate_df.to_excel(writer, sheet_name="gate_stats_test", index=False)
        pd.DataFrame([vars(args)]).to_excel(writer, sheet_name="run_args", index=False)

    meta = {
        "best_variant": best_variant,
        "best_ckpt": best_ckpt,
        "select_metric": args.select_metric,
        "best_val_threshold": best_val_threshold,
        "h_fused_train_csv": os.path.join(export_dir, f"{best_variant}.h_fused_train.csv"),
        "h_fused_val_csv": os.path.join(export_dir, f"{best_variant}.h_fused_val.csv"),
        "h_fused_test_csv": os.path.join(export_dir, f"{best_variant}.h_fused_test.csv"),
        "feature_cols": [f"h_fused_{i}" for i in range(train_ds.hg.shape[1])],
        "dim": int(train_ds.hg.shape[1]),
        "pair_merge_stats": {
            "train": asdict(train_stats),
            "val": asdict(val_stats),
            "test": asdict(test_stats),
        },
    }

    with open(os.path.join(out_dir, "meta_step3_best.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[DONE] Fusion ablation finished.")
    print(f"[BEST] {best_variant} by {args.select_metric}")
    print(f"[BEST] validation threshold fixed for test: {best_val_threshold:.6f}")
    print(f"[EXCEL] {excel_path}")
    print(f"[PDF]   {fig_dir}")
    print(f"[EXPORT]{export_dir}")


if __name__ == "__main__":
    main()
