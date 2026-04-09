#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end inference from raw PDB files to predictions from:
1) all single models
2) all meta models
3) final selected model

Compatible with the formal training script output structure:
run_xxx/
├── single_models/
├── meta_models/
├── meta_compare/
├── final_model/
├── artifacts/
├── tables/
└── ...

Pipeline:
edges -> nodes -> global features -> graph/global embeddings -> fused embeddings -> all trained model inference
"""

import os
import json
import argparse
import subprocess
import glob
import warnings
from typing import List, Dict, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def run_cmd(cmd: List[str], cwd: Optional[str] = None):
    print("\n[CMD]", " ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=cwd)
    if result.stdout.strip():
        print("[STDOUT]", result.stdout.strip())
    if result.stderr.strip():
        print("[STDERR]", result.stderr.strip())


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def assert_exists(path: str, desc: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_protein_id(x: str) -> str:
    x = str(x).strip()
    for suffix in [".ef.pdb", ".pdb", ".ef"]:
        if x.endswith(suffix):
            x = x[:-len(suffix)]
            break
    return x


def list_pdb_ids(pdb_dir: str) -> List[str]:
    pdbs = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdbs:
        raise FileNotFoundError(f"No .pdb files found in: {pdb_dir}")
    return [normalize_protein_id(os.path.basename(p)) for p in pdbs]


def write_dummy_label_csv(pdb_ids: List[str], out_csv: str):
    df = pd.DataFrame({"PDB_ID": pdb_ids, "y": [0] * len(pdb_ids)})
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Dummy label CSV saved: {out_csv} (n={len(pdb_ids)})")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_proba_any(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        return np.clip(prob.astype(float), 0.0, 1.0)
    if hasattr(model, "decision_function"):
        prob = _sigmoid(model.decision_function(X))
        return np.clip(prob.astype(float), 0.0, 1.0)
    pred = model.predict(X).astype(float)
    return np.clip(pred, 0.0, 1.0)


def assert_unique_ids(df: pd.DataFrame, df_name: str):
    if "PDB_ID" not in df.columns:
        raise ValueError(f"{df_name} must contain PDB_ID column")
    dup = df["PDB_ID"].astype(str).duplicated(keep=False)
    if dup.any():
        examples = df.loc[dup, "PDB_ID"].astype(str).tolist()[:10]
        raise ValueError(f"Duplicate PDB_ID found in {df_name}: {examples}")


# ----------------------------------------------------------------------
# RBF function
# ----------------------------------------------------------------------
def rbf(dist: torch.Tensor, D_min: float = 0.0, D_max: float = 20.0, D_count: int = 32) -> torch.Tensor:
    centers = torch.linspace(D_min, D_max, D_count, device=dist.device, dtype=dist.dtype)
    width = (D_max - D_min) / (D_count - 1) if D_count > 1 else 1.0
    diff = dist.unsqueeze(-1) - centers
    return torch.exp(-((diff / width) ** 2))


# ----------------------------------------------------------------------
# Graph Encoder
# ----------------------------------------------------------------------
class GINEEncoder(nn.Module):
    def __init__(
        self,
        base_dim: int = 12,
        b62_dim: int = 20,
        esm_dim: int = 1280,
        base_hidden: int = 64,
        b62_hidden: int = 64,
        esm_hidden: int = 256,
        node_hidden: int = 256,
        edge_rbf_dim: int = 32,
        seqbin_vocab: int = 9,
        seqbin_emb: int = 16,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_rbf_dim = edge_rbf_dim

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

    def forward(self, data):
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
        inv_dist = data.edge_inv_dist.view(-1, 1) if hasattr(data, "edge_inv_dist") else torch.ones_like(is_seq)

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


# ----------------------------------------------------------------------
# Global Encoder
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Fusion model
# IMPORTANT: must match training-time architecture exactly
# ----------------------------------------------------------------------
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
        a = torch.sigmoid(self.gate(torch.cat([hg2, hl2], dim=-1)))
        hf = a * hg2 + (1 - a) * hl2
        return hf, a


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
        a = torch.sigmoid(self.gate(torch.cat([hg2, hl2], dim=-1)))
        hf = a * hg2 + (1 - a) * hl2
        return hf, a


class FusionModel(nn.Module):
    def __init__(self, dim: int, gate_type: str = "vector", use_ln: bool = True, dropout: float = 0.1):
        super().__init__()
        self.gate_type = gate_type
        if gate_type == "scalar":
            self.fuse = ScalarGateFuse(dim, use_ln, dropout)
        elif gate_type == "vector":
            self.fuse = VectorGateFuse(dim, use_ln, dropout)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
        self.classifier = nn.Linear(dim, 1)

    def forward(self, hg: torch.Tensor, hl: torch.Tensor):
        hf, a = self.fuse(hg, hl)
        logit = self.classifier(hf).squeeze(-1)
        return logit, hf, a


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
NODE_SUFFIX = [".node_features.npz", ".node_features.full.npz", ".ef.node_features.full.npz"]
EDGE_SUFFIX = [".edge_features.train.npz", ".ef.edge_features.train.npz"]


def resolve_npz(base_dir: str, pid: str, suffixes: List[str]) -> str:
    for suf in suffixes:
        p = os.path.join(base_dir, f"{pid}{suf}")
        if os.path.exists(p):
            print(f"[MATCH] Found {p}")
            return p
    raise FileNotFoundError(f"No file for {pid} in {base_dir} with suffixes {suffixes}")


class GraphOnlyDataset(Dataset):
    def __init__(self, node_dir: str, edge_dir: str, label_csv: str):
        df = pd.read_csv(label_csv)
        assert_unique_ids(df, "graph label csv")
        rows = []
        for _, r in df.iterrows():
            pid = normalize_protein_id(r["PDB_ID"])
            n = resolve_npz(node_dir, pid, NODE_SUFFIX)
            e = resolve_npz(edge_dir, pid, EDGE_SUFFIX)
            rows.append((pid, n, e))
        if not rows:
            raise RuntimeError("No matched node/edge files found.")
        print(f"[INFO] GraphOnlyDataset loaded {len(rows)} samples")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        pid, npath, epath = self.rows[idx]
        n = np.load(npath, allow_pickle=True)
        if "x_base" in n.files:
            x_base = n["x_base"].astype(np.float32)
            x_b62 = n["x_b62"].astype(np.float32)
            x_esm = n["x_esm"].astype(np.float32)
        else:
            x = n["x"].astype(np.float32)
            x_base = x[:, :12].astype(np.float32) if x.shape[1] >= 12 else x[:, : min(12, x.shape[1])]
            x_b62 = x[:, 12:32].astype(np.float32) if x.shape[1] >= 32 else np.zeros((x.shape[0], 20), dtype=np.float32)
            x_esm = x[:, 32:].astype(np.float32) if x.shape[1] > 32 else np.zeros((x.shape[0], 50), dtype=np.float32)

        e = np.load(epath, allow_pickle=True)
        edge_dist = e["edge_dist"].astype(np.float32)
        edge_inv_dist = (
            e["edge_inv_dist"].astype(np.float32)
            if "edge_inv_dist" in e.files
            else (1.0 / (edge_dist + 1e-3)).astype(np.float32)
        )

        data = Data(
            x_base=torch.from_numpy(x_base),
            x_b62=torch.from_numpy(x_b62),
            x_esm=torch.from_numpy(x_esm),
            edge_index=torch.from_numpy(e["edge_index"].astype(np.int64)),
            edge_dist=torch.from_numpy(edge_dist),
            edge_inv_dist=torch.from_numpy(edge_inv_dist),
            edge_seqbin=torch.from_numpy(e["edge_seqbin"].astype(np.int64)),
            edge_is_seq=torch.from_numpy(e["edge_is_seq"].astype(np.float32)),
            pdb_id=pid,
        )
        data.num_nodes = x_base.shape[0]
        return data


def infer_graph_arch_from_ckpt_or_dataset(ckpt: Dict[str, Any], ds: GraphOnlyDataset) -> Dict[str, Any]:
    arch = dict(ckpt.get("arch", {}) or ckpt.get("encoder_cfg", {}) or {})
    sample = ds[0]
    arch.setdefault("base_dim", int(sample.x_base.shape[1]))
    arch.setdefault("b62_dim", int(sample.x_b62.shape[1]))
    arch.setdefault("esm_dim", int(sample.x_esm.shape[1]))
    arch.setdefault("base_hidden", 64)
    arch.setdefault("b62_hidden", 64)
    arch.setdefault("esm_hidden", 256)
    arch.setdefault("node_hidden", 256)
    arch.setdefault("edge_rbf_dim", 32)
    arch.setdefault("seqbin_vocab", 9)
    arch.setdefault("seqbin_emb", 16)
    arch.setdefault("num_layers", 5)
    arch.setdefault("dropout", 0.1)
    return arch


# ----------------------------------------------------------------------
# Embedding extraction
# ----------------------------------------------------------------------
@torch.no_grad()
def compute_h_graph(node_dir: str, edge_dir: str, dummy_label_csv: str, graph_ckpt: str, device: torch.device, batch_size: int):
    ds = GraphOnlyDataset(node_dir, edge_dir, dummy_label_csv)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    ckpt = torch.load(graph_ckpt, map_location="cpu")
    state = ckpt["encoder_state"] if "encoder_state" in ckpt else ckpt
    arch = infer_graph_arch_from_ckpt_or_dataset(ckpt, ds)

    enc = GINEEncoder(**arch).to(device)
    enc.load_state_dict(state, strict=True)
    enc.eval()

    pids, Hs = [], []
    for batch in loader:
        batch = batch.to(device)
        h = enc(batch).detach().cpu().numpy()
        Hs.append(h)
        pids.extend(batch.pdb_id)

    H = np.vstack(Hs)
    cols = [f"h_graph_{j}" for j in range(H.shape[1])]
    df = pd.concat(
        [
            pd.DataFrame({"PDB_ID": pids}),
            pd.DataFrame(H, columns=cols),
        ],
        axis=1,
    )
    assert_unique_ids(df, "graph embeddings")
    return df


@torch.no_grad()
def compute_h_global(global_merged_csv: str, global_ckpt: str, device: torch.device):
    df = pd.read_csv(global_merged_csv)
    assert_unique_ids(df, "global merged features")

    ckpt = torch.load(global_ckpt, map_location="cpu")
    feat_cols = ckpt["feat_cols"]
    mean = ckpt["scaler_mean"].astype(np.float32)
    scale = ckpt["scaler_scale"].astype(np.float32)
    g_hidden = int(ckpt["args"]["g_hidden"])
    dropout = float(ckpt["args"]["dropout"])

    print("[GLOBAL] Filling missing values with 0")
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    X = df[feat_cols].to_numpy(dtype=np.float32)
    Xs = (X - mean) / (scale + 1e-12)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    enc = GlobalEncoder(g_in=Xs.shape[1], g_hidden=g_hidden, dropout=dropout).to(device)
    enc.load_state_dict(ckpt["encoder_state"], strict=True)
    enc.eval()

    H = enc(torch.from_numpy(Xs).to(device)).cpu().numpy()
    cols = [f"h_global_{j}" for j in range(H.shape[1])]
    df_out = pd.concat(
        [
            df[["PDB_ID"]].reset_index(drop=True),
            pd.DataFrame(H, columns=cols),
        ],
        axis=1,
    )
    assert_unique_ids(df_out, "global embeddings")
    return df_out


@torch.no_grad()
def compute_h_fused(hg_df: pd.DataFrame, hl_df: pd.DataFrame, fuse_ckpt: str, device: torch.device):
    assert_unique_ids(hg_df, "graph embeddings")
    assert_unique_ids(hl_df, "global embeddings")
    df = hg_df.merge(hl_df, on="PDB_ID", how="inner")
    if len(df) == 0:
        raise ValueError("No common PDB_ID between graph and global embeddings")

    gcols = sorted([c for c in df.columns if c.startswith("h_graph_")], key=lambda x: int(x.split("_")[-1]))
    lcols = sorted([c for c in df.columns if c.startswith("h_global_")], key=lambda x: int(x.split("_")[-1]))

    hg = torch.from_numpy(df[gcols].to_numpy(np.float32)).to(device)
    hl = torch.from_numpy(df[lcols].to_numpy(np.float32)).to(device)

    ckpt = torch.load(fuse_ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    gate_type = cfg.get("gate_type", "vector")
    use_ln = bool(cfg.get("use_ln", True))
    dropout = float(cfg.get("dropout", 0.1))
    dim = int(ckpt["dim"])

    model = FusionModel(dim, gate_type, use_ln, dropout).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    _, hf, a = model(hg, hl)
    hf = hf.detach().cpu().numpy()
    a = a.detach().cpu().numpy()

    cols = [f"h_fused_{j}" for j in range(hf.shape[1])]
    df_out = pd.concat(
        [
            df[["PDB_ID"]].reset_index(drop=True),
            pd.DataFrame(hf, columns=cols),
            pd.DataFrame({
                "gate_mean": a.mean(axis=1),
                "gate_std": a.std(axis=1),
            }),
        ],
        axis=1,
    )
    assert_unique_ids(df_out, "fused embeddings")
    return df_out


# ----------------------------------------------------------------------
# MLP bundle helpers
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_mlp_bundle(bundle_path: str, device: torch.device):
    bundle = torch.load(bundle_path, map_location="cpu")
    cfg = bundle["mlp_config"]
    model = MLP(int(bundle["feature_dim"]), cfg["hidden"], float(cfg["dropout"]))
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.to(device).eval()

    scaler = StandardScaler()
    scaler.mean_ = bundle["scaler_mean"]
    scaler.scale_ = bundle["scaler_scale"]
    if "scaler_var" in bundle:
        scaler.var_ = bundle["scaler_var"]
    scaler.n_features_in_ = int(bundle["feature_dim"])

    threshold = float(bundle.get("threshold", 0.5))
    input_feature_names = bundle.get("input_feature_names", None)
    return model, scaler, threshold, input_feature_names, bundle


@torch.no_grad()
def mlp_predict(model_scaler, X: np.ndarray, device: torch.device) -> np.ndarray:
    model, scaler = model_scaler
    Xs = scaler.transform(X).astype(np.float32)
    logits = model(torch.from_numpy(Xs).to(device)).cpu().numpy()
    return _sigmoid(logits)


# ----------------------------------------------------------------------
# Formal training-result inference
# ----------------------------------------------------------------------
def get_hfused_feature_columns(hf_df: pd.DataFrame) -> List[str]:
    feat_cols = [c for c in hf_df.columns if c.startswith("h_fused_")]
    feat_cols = sorted(feat_cols, key=lambda x: int(x.split("_")[-1]))
    if len(feat_cols) == 0:
        raise ValueError("No h_fused_ feature columns found in fused embedding dataframe.")
    return feat_cols


def build_feature_matrix_by_names(df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:20]}")
    X = df[feature_names].astype(np.float32).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in feature matrix")
    return X


def read_threshold_map(train_run_dir: str) -> Dict[str, float]:
    step1_table = os.path.join(train_run_dir, "tables", "results_step1_ml.csv")
    if not os.path.exists(step1_table):
        return {}
    df_step1 = pd.read_csv(step1_table)
    if "Model" not in df_step1.columns or "Threshold" not in df_step1.columns:
        return {}
    return dict(zip(df_step1["Model"].astype(str), df_step1["Threshold"].astype(float)))


def resolve_base_model_path(model_dir: str, model_name: str) -> Tuple[str, str]:
    joblib_path = os.path.join(model_dir, f"{model_name}.joblib")
    pt_path = os.path.join(model_dir, f"{model_name}.pt")
    if os.path.exists(joblib_path):
        return joblib_path, "joblib"
    if os.path.exists(pt_path):
        return pt_path, "mlp_bundle"
    raise FileNotFoundError(f"Base model {model_name} not found in {model_dir} as .joblib or .pt")


def predict_named_base_model(
    hf_df: pd.DataFrame,
    model_dir: str,
    model_name: str,
    device: torch.device,
    threshold_map: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    feat_cols = get_hfused_feature_columns(hf_df)
    X_default = hf_df[feat_cols].astype(np.float32).to_numpy()
    model_path, model_kind = resolve_base_model_path(model_dir, model_name)

    if model_kind == "joblib":
        model = joblib.load(model_path)
        prob = predict_proba_any(model, X_default)
        threshold = None
        if threshold_map is not None and model_name in threshold_map:
            threshold = float(threshold_map[model_name])
        pred = None if threshold is None else (prob >= threshold).astype(int)
        info = {
            "type": "single_ml",
            "threshold": threshold,
            "source_path": model_path,
        }
        return prob.astype(np.float32), pred, info

    model, scaler, threshold, input_feature_names, _ = load_mlp_bundle(model_path, device=device)
    X_model = build_feature_matrix_by_names(hf_df, input_feature_names if input_feature_names else feat_cols)
    prob = mlp_predict((model, scaler), X_model, device=device).astype(np.float32)
    pred = (prob >= float(threshold)).astype(int)
    info = {
        "type": "single_mlp",
        "threshold": float(threshold),
        "source_path": model_path,
    }
    return prob, pred, info


def infer_single_models(hf_df: pd.DataFrame, train_run_dir: str, device: torch.device) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    single_dir = os.path.join(train_run_dir, "single_models")
    assert_exists(single_dir, "single_models directory")
    assert_unique_ids(hf_df, "fused embedding dataframe")

    result = pd.DataFrame({"PDB_ID": hf_df["PDB_ID"].values})
    info_dict: Dict[str, Dict[str, Any]] = {}
    threshold_map = read_threshold_map(train_run_dir)

    joblib_names = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(single_dir, "*.joblib")))]
    pt_names = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(single_dir, "*.pt")))]
    model_names = sorted(set(joblib_names + pt_names))

    for model_name in model_names:
        prob, pred, info = predict_named_base_model(hf_df, single_dir, model_name, device, threshold_map)
        result[f"prob_{model_name}"] = prob
        if pred is not None:
            result[f"pred_{model_name}"] = pred
        info_dict[model_name] = info
        print(f"[SINGLE] done: {model_name}")

    return result, info_dict


def compute_selected_base_prob_matrix(
    hf_df: pd.DataFrame,
    model_dir: str,
    selected_base_models: List[str],
    device: torch.device,
) -> np.ndarray:
    probs = []
    for name in selected_base_models:
        prob, _, _ = predict_named_base_model(hf_df, model_dir, name, device, threshold_map=None)
        probs.append(prob.astype(np.float32))
    return np.stack(probs, axis=1).astype(np.float32)


def infer_meta_models(
    hf_df: pd.DataFrame,
    train_run_dir: str,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    meta_models_dir = os.path.join(train_run_dir, "meta_models")
    meta_compare_csv = os.path.join(train_run_dir, "meta_compare", "results_meta_compare.csv")
    selected_base_models_txt = os.path.join(train_run_dir, "artifacts", "selected_base_models.txt")
    single_dir = os.path.join(train_run_dir, "single_models")

    assert_exists(meta_models_dir, "meta_models directory")
    assert_exists(meta_compare_csv, "results_meta_compare.csv")
    assert_exists(selected_base_models_txt, "selected_base_models.txt")
    assert_exists(single_dir, "single_models directory")

    with open(selected_base_models_txt, "r", encoding="utf-8") as f:
        selected_base_models = [x.strip() for x in f if x.strip()]

    P = compute_selected_base_prob_matrix(hf_df, single_dir, selected_base_models, device)
    feat_cols = get_hfused_feature_columns(hf_df)
    X_raw = hf_df[feat_cols].astype(np.float32).to_numpy()

    df_meta_compare = pd.read_csv(meta_compare_csv)
    result = pd.DataFrame({"PDB_ID": hf_df["PDB_ID"].values})
    info_dict: Dict[str, Dict[str, Any]] = {}

    for _, row in df_meta_compare.iterrows():
        model_name = str(row["Model"])
        exp_dir = os.path.join(meta_models_dir, model_name)
        info_json = os.path.join(exp_dir, "model_info.json")
        assert_exists(info_json, f"model_info for meta model {model_name}")
        info = load_json(info_json)

        meta_type = str(info["type"])
        input_feature_names = info.get("input_feature_names", None)
        threshold = float(info.get("threshold", row["Threshold"] if "Threshold" in row else 0.5))

        if input_feature_names is None:
            raise ValueError(f"input_feature_names missing in {info_json}")

        df_meta_input = hf_df.copy()
        for j, name in enumerate(selected_base_models):
            df_meta_input[f"OOF_{name}"] = P[:, j]

        X_meta = build_feature_matrix_by_names(df_meta_input, input_feature_names)

        if meta_type == "simple_average":
            prob = P.mean(axis=1)
        elif meta_type == "stacking":
            model_path = os.path.join(exp_dir, "stacker.joblib")
            assert_exists(model_path, f"stacker model for {model_name}")
            stacker = joblib.load(model_path)
            prob = stacker.predict_proba(X_meta)[:, 1]
        elif meta_type == "mlp2":
            model_path = os.path.join(exp_dir, "mlp2_model.pt")
            assert_exists(model_path, f"mlp2 model for {model_name}")
            mlp_model, scaler, thr_bundle, feature_names_bundle, _ = load_mlp_bundle(model_path, device=device)
            if feature_names_bundle is not None:
                X_meta = build_feature_matrix_by_names(df_meta_input, feature_names_bundle)
            prob = mlp_predict((mlp_model, scaler), X_meta, device=device)
            threshold = float(thr_bundle)
        else:
            raise ValueError(f"Unknown meta model type in {info_json}: {meta_type}")

        result[f"prob_{model_name}"] = prob
        result[f"pred_{model_name}"] = (prob >= threshold).astype(int)
        info_dict[model_name] = {
            "type": meta_type,
            "threshold": threshold,
            "source_dir": exp_dir,
            "selected_base_models": selected_base_models,
        }
        print(f"[META] done: {model_name}")

    meta_context = {
        "selected_base_models": selected_base_models,
        "selected_base_prob_matrix_shape": list(P.shape),
        "raw_feature_dim": int(X_raw.shape[1]),
    }
    return result, info_dict, meta_context


def infer_final_model(
    hf_df: pd.DataFrame,
    train_run_dir: str,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    final_dir = os.path.join(train_run_dir, "final_model")
    manifest_path = os.path.join(final_dir, "final_model_manifest.json")
    base_models_dir = os.path.join(final_dir, "base_models")
    assert_exists(final_dir, "final_model directory")
    assert_exists(manifest_path, "final_model_manifest.json")
    assert_exists(base_models_dir, "final_model/base_models directory")

    manifest = load_json(manifest_path)
    selected_model = manifest["selected_model"]
    final_type = manifest["type"]
    threshold = float(manifest["threshold"])
    top_models = manifest["top_models"]
    add_raw_features = bool(manifest["add_raw_features"])
    final_model_filename = manifest["final_model_filename"]

    feat_cols = manifest["base_feature_names"]
    X = build_feature_matrix_by_names(hf_df, feat_cols)

    probs = []
    for name in top_models:
        prob, _, _ = predict_named_base_model(hf_df, base_models_dir, name, device, threshold_map=None)
        probs.append(prob.astype(np.float32))
    P = np.stack(probs, axis=1).astype(np.float32)

    if final_type == "average":
        final_prob = P.mean(axis=1)
    elif final_type == "stacking":
        stacker_path = os.path.join(final_dir, final_model_filename)
        assert_exists(stacker_path, "final stacker model")
        stacker = joblib.load(stacker_path)
        X_meta = np.concatenate([X, P], axis=1).astype(np.float32) if add_raw_features else P.astype(np.float32)
        final_prob = stacker.predict_proba(X_meta)[:, 1]
    elif final_type == "mlp2":
        mlp_path = os.path.join(final_dir, final_model_filename)
        assert_exists(mlp_path, "final mlp2 model")
        model, scaler, thr_bundle, input_feature_names, _ = load_mlp_bundle(mlp_path, device=device)
        df_meta_input = hf_df.copy()
        for j, name in enumerate(top_models):
            df_meta_input[f"OOF_{name}"] = P[:, j]
        if input_feature_names is not None:
            X_meta = build_feature_matrix_by_names(df_meta_input, input_feature_names)
        else:
            X_meta = np.concatenate([X, P], axis=1).astype(np.float32) if add_raw_features else P.astype(np.float32)
        final_prob = mlp_predict((model, scaler), X_meta, device=device)
        threshold = float(thr_bundle)
    else:
        raise ValueError(f"Unknown final model type: {final_type}")

    final_pred = (final_prob >= threshold).astype(int)
    out = pd.DataFrame({
        "PDB_ID": hf_df["PDB_ID"].values,
        "prob_final_model": final_prob,
        "pred_final_model": final_pred,
        "selected_model_name": selected_model,
        "threshold_used": threshold,
    })
    for j, name in enumerate(top_models):
        out[f"prob_final_base_{name}"] = P[:, j]

    info = {
        "selected_model": selected_model,
        "type": final_type,
        "threshold": threshold,
        "top_models": top_models,
        "add_raw_features": add_raw_features,
        "final_model_filename": final_model_filename,
    }
    return out, info


def infer_all_models(
    hf_df: pd.DataFrame,
    train_run_dir: str,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    single_df, single_info = infer_single_models(hf_df, train_run_dir, device)
    meta_df, meta_info, meta_context = infer_meta_models(hf_df, train_run_dir, device)
    final_df, final_info = infer_final_model(hf_df, train_run_dir, device)

    out = single_df.merge(meta_df, on="PDB_ID", how="outer").merge(final_df, on="PDB_ID", how="outer")

    selected_model = final_info["selected_model"]
    prob_col = f"prob_{selected_model}"
    pred_col = f"pred_{selected_model}"
    out["prob_selected_model"] = out[prob_col] if prob_col in out.columns else out["prob_final_model"]
    out["pred_selected_model"] = out[pred_col] if pred_col in out.columns else out["pred_final_model"]

    summary = {
        "single_models": single_info,
        "meta_models": meta_info,
        "meta_context": meta_context,
        "final_model": final_info,
    }
    return out, summary


# ----------------------------------------------------------------------
# Main pipeline functions
# ----------------------------------------------------------------------
def run_global_features(
    pdb_dir: str,
    out_dir: str,
    n_jobs: int,
    global_scripts_dir: str,
    script_names: List[str],
) -> str:
    ensure_dir(out_dir)

    scripts = []
    for i, name in enumerate(script_names, 1):
        sp = os.path.join(global_scripts_dir, name)
        assert_exists(sp, f"global script {i} ({name})")
        scripts.append((str(i), sp))

    reduced_paths = []
    for tag, sp in scripts:
        out_full = os.path.join(out_dir, f"{tag}_full.csv")
        out_red = os.path.join(out_dir, f"{tag}_reduced.csv")
        run_cmd([
            "python", sp,
            "--pdb_dir", pdb_dir,
            "--out_csv", out_full,
            "--out_csv_reduced", out_red,
            "--n_jobs", str(n_jobs),
        ])
        reduced_paths.append(out_red)

    dfs = []
    for p in reduced_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            print(f"[WARN] Reduced CSV not found: {p}")

    if not dfs:
        raise RuntimeError("No global reduced CSVs generated.")

    merged = dfs[0]
    assert_unique_ids(merged, f"global reduced csv: {reduced_paths[0]}")
    for p, d in zip(reduced_paths[1:], dfs[1:]):
        assert_unique_ids(d, f"global reduced csv: {p}")
        merged = merged.merge(d, on="PDB_ID", how="outer")

    merged_path = os.path.join(out_dir, "global_features_merged.csv")
    merged.to_csv(merged_path, index=False)
    print(f"[GLOBAL] Merged global features saved: {merged_path} (n={len(merged)})")
    return merged_path


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="End-to-end inference from PDB to all trained model predictions")

    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--graph_ckpt", required=True)
    ap.add_argument("--global_ckpt", required=True)
    ap.add_argument("--fuse_ckpt", required=True)

    ap.add_argument(
        "--train_run_dir",
        required=True,
        help="Path to the formal training run directory, e.g. results_formal/run_20260317_123456",
    )

    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--build_edges_py", default="/media/yugengliu/Computer21/SurfSolNet/2_graph_feature/1-1-build_edges.py")
    ap.add_argument("--build_node_features_full_py", default="/media/yugengliu/Computer21/SurfSolNet/2_graph_feature/1-2-build_node_features_full.py")
    ap.add_argument("--global_scripts_dir", default="/media/yugengliu/Computer21/SurfSolNet/1_global_feature")
    ap.add_argument("--global_script_names", nargs=5, default=[
        "1-sequence_global.py",
        "2-structure_confidence_global.py",
        "3-surface_physchem_aggregation_risk_global.py",
        "4-compactness_shape_global.py",
        "5-interaction_network_global.py",
    ])

    ap.add_argument("--edge-nproc", type=int, default=16)
    ap.add_argument("--node-nproc", type=int, default=16)
    ap.add_argument("--global-n-jobs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2500)
    ap.add_argument("--graph-batch", type=int, default=16)

    ap.add_argument(
        "--skip-to",
        type=int,
        default=1,
        choices=range(1, 7),
        help="Start from which stage: 1=full pipeline, 3=from graph emb, ..., 6=final prediction only",
    )

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ensure_dir(args.work_dir)
    assert_exists(args.train_run_dir, "train_run_dir")

    print(f"[START] {args.pdb_dir} -> {args.out_csv}")
    print(f"[INFO] Device: {device} | Skip-to stage: {args.skip_to}")
    print(f"[INFO] Training run dir: {args.train_run_dir}")

    dummy_label = os.path.join(args.work_dir, "dummy_labels.csv")
    edge_dir = os.path.join(args.work_dir, "edges")
    node_dir = os.path.join(args.work_dir, "nodes")
    global_dir = os.path.join(args.work_dir, "global")
    global_merged = os.path.join(global_dir, "global_features_merged.csv")
    graph_emb_path = os.path.join(args.work_dir, "graph_embeddings.csv")
    global_emb_path = os.path.join(args.work_dir, "global_embeddings.csv")
    fused_emb_path = os.path.join(args.work_dir, "fused_embeddings.csv")
    summary_json = os.path.join(args.work_dir, "inference_summary.json")

    if args.skip_to <= 2:
        print("[1-2] Preparing dummy / edges / nodes / global features...")
        pdb_ids = list_pdb_ids(args.pdb_dir)
        write_dummy_label_csv(pdb_ids, dummy_label)

        ensure_dir(edge_dir)
        run_cmd([
            "python", args.build_edges_py,
            "--pdb_dir", args.pdb_dir,
            "--out_dir", edge_dir,
            "--index_csv", os.path.join(args.work_dir, "edge_index.csv"),
            "--nproc", str(args.edge_nproc),
        ])

        ensure_dir(node_dir)
        node_dev = "cuda" if "cuda" in str(device) else "cpu"
        run_cmd([
            "python", args.build_node_features_full_py,
            "--pdb_dir", args.pdb_dir,
            "--out_dir", node_dir,
            "--index_csv", os.path.join(args.work_dir, "node_index.csv"),
            "--nproc", str(args.node_nproc),
            "--device", node_dev,
            "--fp16",
            "--max_len", str(args.max_len),
        ])

        run_global_features(args.pdb_dir, global_dir, args.global_n_jobs, args.global_scripts_dir, args.global_script_names)
    else:
        print("[Skip 1-2] Assuming features already prepared")
        if args.skip_to <= 3:
            assert_exists(dummy_label, "dummy_labels.csv")
            assert_exists(node_dir, "node_dir")
            assert_exists(edge_dir, "edge_dir")
        if args.skip_to <= 4:
            assert_exists(global_merged, "global_features_merged.csv")

    if args.skip_to <= 3:
        if os.path.exists(graph_emb_path):
            print(f"[3] Loading existing graph embeddings: {graph_emb_path}")
            hg_df = pd.read_csv(graph_emb_path)
        else:
            print("[3] Computing graph embeddings...")
            hg_df = compute_h_graph(node_dir, edge_dir, dummy_label, args.graph_ckpt, device, args.graph_batch)
            hg_df.to_csv(graph_emb_path, index=False)
            print(f"[3] Saved: {graph_emb_path}")
    else:
        print(f"[Skip 3] Loading {graph_emb_path}")
        hg_df = pd.read_csv(graph_emb_path)
    assert_unique_ids(hg_df, "graph embedding dataframe")

    if args.skip_to <= 4:
        if os.path.exists(global_emb_path):
            print(f"[4] Loading existing global embeddings: {global_emb_path}")
            hl_df = pd.read_csv(global_emb_path)
        else:
            print("[4] Computing global embeddings...")
            hl_df = compute_h_global(global_merged, args.global_ckpt, device)
            hl_df.to_csv(global_emb_path, index=False)
            print(f"[4] Saved: {global_emb_path}")
    else:
        print(f"[Skip 4] Loading {global_emb_path}")
        hl_df = pd.read_csv(global_emb_path)
    assert_unique_ids(hl_df, "global embedding dataframe")

    if args.skip_to <= 5:
        if os.path.exists(fused_emb_path):
            print(f"[5] Loading existing fused embeddings: {fused_emb_path}")
            hf_df = pd.read_csv(fused_emb_path)
        else:
            print("[5] Performing fusion...")
            hf_df = compute_h_fused(hg_df, hl_df, args.fuse_ckpt, device)
            hf_df.to_csv(fused_emb_path, index=False)
            print(f"[5] Saved: {fused_emb_path}")
    else:
        print(f"[Skip 5] Loading {fused_emb_path}")
        hf_df = pd.read_csv(fused_emb_path)
    assert_unique_ids(hf_df, "fused embedding dataframe")

    print("[6] Running trained-model inference...")
    out_df, summary = infer_all_models(hf_df, args.train_run_dir, device)
    out_df.to_csv(args.out_csv, index=False)
    save_json(summary, summary_json)

    print(f"[DONE] Predictions saved to: {args.out_csv}")
    print(f"[DONE] Summary saved to: {summary_json}")
    print(f"[DONE] n={len(out_df)}")


if __name__ == "__main__":
    main()