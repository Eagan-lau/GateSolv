#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import random
import argparse
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_protein_id(x: str) -> str:
    x = str(x).strip()
    if x.endswith(".ef.pdb"):
        x = x[:-7]
    elif x.endswith(".pdb"):
        x = x[:-4]
    elif x.endswith(".ef"):
        x = x[:-3]
    return x


def matthews_corrcoef_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def best_threshold_by_mcc(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(prob)
    y_true = y_true[m]
    prob = prob[m]
    if prob.size == 0:
        return 0.5, 0.0
    thr_list = np.unique(prob)
    best_thr, best_mcc = 0.5, -1e9
    for t in thr_list:
        pred = (prob >= t).astype(int)
        mcc = matthews_corrcoef_binary(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(t)
    return best_thr, float(best_mcc)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def rbf(dist: torch.Tensor, D_min=0.0, D_max=20.0, D_count=32) -> torch.Tensor:
    centers = torch.linspace(D_min, D_max, D_count, device=dist.device)
    widths = (D_max - D_min) / D_count
    gamma = 1.0 / (widths ** 2 + 1e-8)
    dist = dist.view(-1, 1)
    return torch.exp(-gamma * (dist - centers) ** 2)


NODE_SUFFIX_CANDIDATES = [".node_features.npz", ".node_features.full.npz", ".ef.node_features.full.npz"]
EDGE_SUFFIX_CANDIDATES = [".edge_features.train.npz", ".ef.edge_features.train.npz"]


def resolve_npz_path(base_dir: str, pdb_id: str, suffix_candidates: List[str]) -> Optional[str]:
    for suf in suffix_candidates:
        p = os.path.join(base_dir, f"{pdb_id}{suf}")
        if os.path.exists(p):
            return p
    return None


class GraphDataset(Dataset):
    def __init__(self, node_dir: str, edge_dir: str, label_csv: str, strict_finite: bool = True):
        df = pd.read_csv(label_csv)
        if "PDB_ID" not in df.columns or "y" not in df.columns:
            raise ValueError("label_csv must contain columns: PDB_ID,y")
        rows = []
        missing = 0
        filtered = 0
        for _, r in df.iterrows():
            pid = normalize_protein_id(r["PDB_ID"])
            y = int(r["y"])
            node_npz = resolve_npz_path(node_dir, pid, NODE_SUFFIX_CANDIDATES)
            edge_npz = resolve_npz_path(edge_dir, pid, EDGE_SUFFIX_CANDIDATES)
            if node_npz is None or edge_npz is None:
                missing += 1
                continue
            if strict_finite:
                try:
                    n = np.load(node_npz, allow_pickle=True)
                    if "x_base" in n.files:
                        xb = n["x_base"].astype(np.float32)
                        xb62 = n["x_b62"].astype(np.float32)
                        xe = n["x_esm"].astype(np.float32)
                        if (not np.isfinite(xb).all()) or (not np.isfinite(xb62).all()) or (not np.isfinite(xe).all()):
                            filtered += 1
                            continue
                    else:
                        x = n["x"].astype(np.float32)
                        if not np.isfinite(x).all():
                            filtered += 1
                            continue
                    e = np.load(edge_npz, allow_pickle=True)
                    for k in ["edge_dist"]:
                        if k in e.files:
                            v = e[k].astype(np.float32)
                            if not np.isfinite(v).all():
                                filtered += 1
                                raise ValueError("non-finite edge feature")
                except Exception:
                    continue
            rows.append((pid, y, node_npz, edge_npz))
        if not rows:
            raise RuntimeError("No matched samples found.")
        self.rows = rows
        self.missing = missing
        self.filtered = filtered
        print(f"[INFO] Loaded {len(rows)} samples (missing={missing}, filtered_nonfinite={filtered}) from {label_csv}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        pid, y, node_npz, edge_npz = self.rows[idx]
        n = np.load(node_npz, allow_pickle=True)
        if "x_base" in n.files:
            x_base = n["x_base"].astype(np.float32)
            x_b62 = n["x_b62"].astype(np.float32)
            x_esm = n["x_esm"].astype(np.float32)
        else:
            x = n["x"].astype(np.float32)
            x_base = x[:, :12].astype(np.float32)
            x_b62 = x[:, 12:32].astype(np.float32)
            x_esm = x[:, 32:].astype(np.float32)

        e = np.load(edge_npz, allow_pickle=True)
        edge_index = e["edge_index"].astype(np.int64)
        edge_dist = e["edge_dist"].astype(np.float32)
        edge_seqbin = e["edge_seqbin"].astype(np.int64)
        edge_is_seq = e["edge_is_seq"].astype(np.float32)
        edge_inv_dist = e["edge_inv_dist"].astype(np.float32) if "edge_inv_dist" in e.files else (1.0 / (edge_dist + 1e-3)).astype(np.float32)

        L_node = x_base.shape[0]
        L_edge = int(e["L"]) if "L" in e.files else L_node
        if L_node != L_edge:
            raise ValueError(f"{pid}: L mismatch node={L_node}, edge={L_edge}")

        if not np.isfinite(x_base).all():
            raise ValueError(f"{pid}: non-finite x_base")
        if not np.isfinite(x_b62).all():
            raise ValueError(f"{pid}: non-finite x_b62")
        if not np.isfinite(x_esm).all():
            raise ValueError(f"{pid}: non-finite x_esm")
        if not np.isfinite(edge_dist).all():
            raise ValueError(f"{pid}: non-finite edge_dist")

        data = Data(
            x_base=torch.from_numpy(x_base),
            x_b62=torch.from_numpy(x_b62),
            x_esm=torch.from_numpy(x_esm),
            edge_index=torch.from_numpy(edge_index),
            edge_dist=torch.from_numpy(edge_dist),
            edge_inv_dist=torch.from_numpy(edge_inv_dist),
            edge_seqbin=torch.from_numpy(edge_seqbin),
            edge_is_seq=torch.from_numpy(edge_is_seq),
            y=torch.tensor([float(y)], dtype=torch.float32),
            pdb_id=pid,
        )
        data.num_nodes = L_node
        return data


class GINEEncoder(nn.Module):
    def __init__(
        self,
        base_dim=12,
        b62_dim=20,
        esm_dim=1280,
        base_hidden=64,
        b62_hidden=64,
        esm_hidden=256,
        node_hidden=256,
        edge_rbf_dim=32,
        seqbin_vocab=9,
        seqbin_emb=16,
        num_layers=5,
        dropout=0.1,
    ):
        super().__init__()
        self.base_mlp = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, base_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_hidden, base_hidden),
            nn.GELU(),
        )
        self.b62_mlp = nn.Sequential(
            nn.LayerNorm(b62_dim),
            nn.Linear(b62_dim, b62_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(b62_hidden, b62_hidden),
            nn.GELU(),
        )
        self.esm_mlp = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, esm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(esm_hidden, esm_hidden),
            nn.GELU(),
        )
        struct_dim = base_hidden + b62_hidden
        self.struct_mlp = nn.Sequential(
            nn.LayerNorm(struct_dim),
            nn.Linear(struct_dim, struct_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(struct_dim + esm_hidden, esm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(esm_hidden, esm_hidden),
            nn.Sigmoid(),
        )
        fuse_dim = struct_dim + esm_hidden
        self.node_fuse = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, node_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.seq_emb = nn.Embedding(seqbin_vocab, seqbin_emb)
        self.edge_rbf_dim = edge_rbf_dim
        edge_raw_dim = edge_rbf_dim + seqbin_emb + 1 + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_raw_dim, node_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_hidden, node_hidden),
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = nn.Sequential(
                nn.Linear(node_hidden, node_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(node_hidden, node_hidden),
            )
            self.convs.append(GINEConv(nn_edge, edge_dim=node_hidden))
        self.norms = nn.ModuleList([nn.LayerNorm(node_hidden) for _ in range(num_layers)])
        self.dropout = dropout
        self.readout = nn.Sequential(
            nn.Linear(node_hidden * 2, node_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, data: Data) -> torch.Tensor:
        h_base = self.base_mlp(data.x_base)
        h_b62 = self.b62_mlp(data.x_b62)
        h_struct = self.struct_mlp(torch.cat([h_base, h_b62], dim=1))
        h_esm = self.esm_mlp(data.x_esm)
        gate = self.gate_mlp(torch.cat([h_struct, h_esm], dim=1))
        h_esm = h_esm * gate
        x = self.node_fuse(torch.cat([h_struct, h_esm], dim=1))
        rbf_feat = rbf(data.edge_dist, D_min=0.0, D_max=20.0, D_count=self.edge_rbf_dim)
        seq_feat = self.seq_emb(data.edge_seqbin)
        is_seq = data.edge_is_seq.view(-1, 1)
        inv_dist = data.edge_inv_dist.view(-1, 1)
        edge_raw = torch.cat([rbf_feat, seq_feat, is_seq, inv_dist], dim=1)
        edge_attr = self.edge_mlp(edge_raw)
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, data.edge_index, edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_in
        h_mean = global_mean_pool(x, data.batch)
        h_max = global_max_pool(x, data.batch)
        return self.readout(torch.cat([h_mean, h_max], dim=1))


class GraphClassifier(nn.Module):
    def __init__(self, encoder: GINEEncoder, dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, 1)

    def forward(self, data: Data):
        h = self.encoder(data)
        logit = self.head(h).squeeze(1)
        return logit, h


@torch.no_grad()
def eval_loader(model: GraphClassifier, loader: DataLoader, device: torch.device, use_amp: bool):
    model.eval()
    ys, probs, pdb_ids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logit, _ = model(batch)
            prob = torch.sigmoid(logit).detach().float().cpu().numpy()
        y = batch.y.view(-1).detach().float().cpu().numpy()
        ys.append(y)
        probs.append(prob)
        batch_ids = getattr(batch, "pdb_id", None)
        if batch_ids is None:
            pdb_ids.extend([None] * len(y))
        elif isinstance(batch_ids, (list, tuple)):
            pdb_ids.extend(list(batch_ids))
        else:
            pdb_ids.extend([batch_ids])

    y_true = np.concatenate(ys).astype(int)
    prob = np.concatenate(probs)
    finite = np.isfinite(prob)
    if not finite.all():
        y_true = y_true[finite]
        prob = prob[finite]
        pdb_ids = [pdb_ids[i] for i, keep in enumerate(finite) if keep]

    thr, mcc = best_threshold_by_mcc(y_true, prob)
    pred = (prob >= thr).astype(int)
    prec, rec, f1 = precision_recall_f1(y_true, pred)

    if prob.size > 0 and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, prob))
        pr_auc = float(average_precision_score(y_true, prob))
        fpr, tpr, roc_thr = roc_curve(y_true, prob)
        pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true, prob)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")
        fpr, tpr, roc_thr = np.array([]), np.array([]), np.array([])
        pr_prec, pr_rec, pr_thr = np.array([]), np.array([]), np.array([])

    return {
        "thr": float(thr),
        "mcc": float(mcc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "prob_mean": float(np.mean(prob)) if prob.size else float("nan"),
        "prob_std": float(np.std(prob)) if prob.size else float("nan"),
        "y_true": y_true.tolist(),
        "prob": prob.tolist(),
        "pred": pred.tolist(),
        "pdb_id": list(pdb_ids),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thr.tolist(),
        "pr_precision": pr_prec.tolist(),
        "pr_recall": pr_rec.tolist(),
        "pr_thresholds": pr_thr.tolist(),
    }


def infer_dims_from_dataset(ds: GraphDataset):
    item = ds[0]
    return {
        "base_dim": int(item.x_base.shape[1]),
        "b62_dim": int(item.x_b62.shape[1]),
        "esm_dim": int(item.x_esm.shape[1]),
    }


def save_prediction_artifacts(metrics: dict, out_dir: str, prefix: str):
    pred_path = os.path.join(out_dir, f"{prefix}_predictions.csv")
    pd.DataFrame({
        "pdb_id": metrics.get("pdb_id", []),
        "y_true": metrics.get("y_true", []),
        "prob": metrics.get("prob", []),
        "pred": metrics.get("pred", []),
    }).to_csv(pred_path, index=False)

    roc_path = os.path.join(out_dir, f"{prefix}_roc_curve.csv")
    pd.DataFrame({
        "fpr": metrics.get("fpr", []),
        "tpr": metrics.get("tpr", []),
        "threshold": metrics.get("roc_thresholds", []),
    }).to_csv(roc_path, index=False)

    pr_path = os.path.join(out_dir, f"{prefix}_pr_curve.csv")
    pd.DataFrame({
        "precision": metrics.get("pr_precision", []),
        "recall": metrics.get("pr_recall", []),
    }).to_csv(pr_path, index=False)



def main():
    ap = argparse.ArgumentParser("Train upgraded graph encoder.")
    ap.add_argument("--train_node_dir", required=True)
    ap.add_argument("--train_edge_dir", required=True)
    ap.add_argument("--train_label_csv", required=True)
    ap.add_argument("--val_node_dir", required=True)
    ap.add_argument("--val_edge_dir", required=True)
    ap.add_argument("--val_label_csv", required=True)
    ap.add_argument("--test_node_dir", required=True)
    ap.add_argument("--test_edge_dir", required=True)
    ap.add_argument("--test_label_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_hidden", type=int, default=64)
    ap.add_argument("--b62_hidden", type=int, default=64)
    ap.add_argument("--esm_hidden", type=int, default=256)
    ap.add_argument("--node_hidden", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=5)
    ap.add_argument("--seqbin_vocab", type=int, default=9)
    ap.add_argument("--seqbin_emb", type=int, default=16)
    ap.add_argument("--edge_rbf_dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    if device.type == "cuda" and (not args.no_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tr_ds = GraphDataset(args.train_node_dir, args.train_edge_dir, args.train_label_csv, strict_finite=True)
    va_ds = GraphDataset(args.val_node_dir, args.val_edge_dir, args.val_label_csv, strict_finite=True)
    te_ds = GraphDataset(args.test_node_dir, args.test_edge_dir, args.test_label_csv, strict_finite=True)

    dims = infer_dims_from_dataset(tr_ds)
    print(f"[INFO] inferred node feature dimensions: {dims}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    enc = GINEEncoder(
        base_dim=dims["base_dim"],
        b62_dim=dims["b62_dim"],
        esm_dim=dims["esm_dim"],
        base_hidden=args.base_hidden,
        b62_hidden=args.b62_hidden,
        esm_hidden=args.esm_hidden,
        node_hidden=args.node_hidden,
        edge_rbf_dim=args.edge_rbf_dim,
        seqbin_vocab=args.seqbin_vocab,
        seqbin_emb=args.seqbin_emb,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model = GraphClassifier(enc, dim=args.node_hidden).to(device)

    y_tr = np.array([y for (_, y, _, _) in tr_ds.rows], dtype=int)
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_mcc = -1e9
    best_path_encoder = os.path.join(args.out_dir, "graph_encoder_best.pt")
    hist = []
    bad = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logit, _ = model(batch)
                y = batch.y.view(-1).to(device)
                loss = criterion(logit, y)
            if not torch.isfinite(loss):
                pids = getattr(batch, "pdb_id", None)
                print("[NAN] loss is not finite. Example pids:", pids[:5] if pids is not None else "N/A")
                raise RuntimeError("Non-finite loss encountered.")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(losses)) if losses else float("nan")

        va_metrics = eval_loader(model, va_loader, device, use_amp)
        print(
            f"[E{epoch:03d}] train_loss={tr_loss:.4f} "
            f"VAL_MCC={va_metrics['mcc']:.4f} F1={va_metrics['f1']:.4f} "
            f"ROC_AUC={va_metrics['roc_auc']:.4f} PR_AUC={va_metrics['pr_auc']:.4f} "
            f"thr={va_metrics['thr']:.3f} prob_mean={va_metrics['prob_mean']:.3f}±{va_metrics['prob_std']:.3f}"
        )

        hist.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_thr": va_metrics["thr"],
            "val_mcc": va_metrics["mcc"],
            "val_precision": va_metrics["precision"],
            "val_recall": va_metrics["recall"],
            "val_f1": va_metrics["f1"],
            "val_roc_auc": va_metrics["roc_auc"],
            "val_pr_auc": va_metrics["pr_auc"],
            "val_prob_mean": va_metrics["prob_mean"],
            "val_prob_std": va_metrics["prob_std"],
        })

        if va_metrics["mcc"] > best_mcc + 1e-6:
            best_mcc = float(va_metrics["mcc"])
            bad = 0
            torch.save({
                "encoder_state": model.encoder.state_dict(),
                "head_state": model.head.state_dict(),
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_mcc": best_mcc,
                "best_val_roc_auc": va_metrics["roc_auc"],
                "best_val_pr_auc": va_metrics["pr_auc"],
                "arch": {
                    "base_dim": dims["base_dim"],
                    "b62_dim": dims["b62_dim"],
                    "esm_dim": dims["esm_dim"],
                    "base_hidden": args.base_hidden,
                    "b62_hidden": args.b62_hidden,
                    "esm_hidden": args.esm_hidden,
                    "node_hidden": args.node_hidden,
                    "edge_rbf_dim": args.edge_rbf_dim,
                    "seqbin_vocab": args.seqbin_vocab,
                    "seqbin_emb": args.seqbin_emb,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
                "args": vars(args),
            }, best_path_encoder)
            save_prediction_artifacts(va_metrics, args.out_dir, "best_val")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EARLY STOP] no improvement for {args.patience} epochs.")
                break

    pd.DataFrame(hist).to_csv(os.path.join(args.out_dir, "train_history.csv"), index=False)

    # Load best model and run final TEST evaluation
    ckpt = torch.load(best_path_encoder, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    else:
        model.encoder.load_state_dict(ckpt["encoder_state"], strict=True)
        if "head_state" in ckpt:
            model.head.load_state_dict(ckpt["head_state"], strict=True)
        else:
            raise ValueError("Best checkpoint does not contain head_state/model_state; cannot run test probability evaluation.")

    test_metrics = eval_loader(model, te_loader, device, use_amp)
    save_prediction_artifacts(test_metrics, args.out_dir, "best_test")

    best_row = max(hist, key=lambda x: x["val_mcc"]) if len(hist) > 0 else {}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({
            "best_val_mcc": best_mcc,
            "best_val_roc_auc": best_row.get("val_roc_auc", None),
            "best_val_pr_auc": best_row.get("val_pr_auc", None),
            "best_test_mcc": test_metrics["mcc"],
            "best_test_f1": test_metrics["f1"],
            "best_test_precision": test_metrics["precision"],
            "best_test_recall": test_metrics["recall"],
            "best_test_roc_auc": test_metrics["roc_auc"],
            "best_test_pr_auc": test_metrics["pr_auc"],
            "best_encoder_ckpt": best_path_encoder,
        }, f, indent=2)

    print("\n[DONE] Graph encoder training finished.")
    print(f"Best VAL_MCC={best_mcc:.4f}")
    print(
        f"[BEST TEST] MCC={test_metrics['mcc']:.4f} F1={test_metrics['f1']:.4f} "
        f"Precision={test_metrics['precision']:.4f} Recall={test_metrics['recall']:.4f} "
        f"ROC_AUC={test_metrics['roc_auc']:.4f} PR_AUC={test_metrics['pr_auc']:.4f} "
        f"thr={test_metrics['thr']:.4f}"
    )
    print(f"Encoder ckpt: {best_path_encoder}")


if __name__ == "__main__":
    main()
