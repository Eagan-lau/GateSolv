#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool


def normalize_protein_id(x: str) -> str:
    x = str(x).strip()
    if x.endswith(".ef.pdb"):
        x = x[:-7]
    elif x.endswith(".pdb"):
        x = x[:-4]
    elif x.endswith(".ef"):
        x = x[:-3]
    return x


def rbf(dist, D_min=0.0, D_max=20.0, D_count=32, device="cpu"):
    centers = torch.linspace(D_min, D_max, D_count, device=device)
    widths = (D_max - D_min) / D_count
    gamma = 1.0 / (widths ** 2 + 1e-8)
    dist = dist.view(-1, 1)
    return torch.exp(-gamma * (dist - centers) ** 2)


NODE_SUFFIX = [".node_features.full.npz", ".ef.node_features.full.npz"]
EDGE_SUFFIX = [".edge_features.train.npz", ".ef.edge_features.train.npz"]


def resolve(base_dir, pid, suffixes):
    for suf in suffixes:
        p = os.path.join(base_dir, f"{pid}{suf}")
        if os.path.exists(p):
            return p
    return None


class GraphOnlyDataset(Dataset):
    def __init__(self, node_dir, edge_dir, label_csv):
        df = pd.read_csv(label_csv)
        rows = []

        for _, r in df.iterrows():
            pid = normalize_protein_id(r["PDB_ID"])
            y = float(r["y"])
            n = resolve(node_dir, pid, NODE_SUFFIX)
            e = resolve(edge_dir, pid, EDGE_SUFFIX)
            if n and e:
                rows.append((pid, y, n, e))

        if not rows:
            raise RuntimeError("No samples matched.")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        pid, y, npath, epath = self.rows[idx]

        n = np.load(npath, allow_pickle=True)
        if "x_base" in n.files:
            x_base = n["x_base"].astype(np.float32)
            x_b62 = n["x_b62"].astype(np.float32)
            x_esm = n["x_esm"].astype(np.float32)
        else:
            x = n["x"].astype(np.float32)
            x_base = x[:, :7].astype(np.float32)
            x_b62 = x[:, 7:27].astype(np.float32)
            x_esm = x[:, 27:].astype(np.float32)

        e = np.load(epath, allow_pickle=True)
        edge_dist = e["edge_dist"].astype(np.float32)
        edge_inv_dist = e["edge_inv_dist"].astype(np.float32) if "edge_inv_dist" in e.files else (1.0 / (edge_dist + 1e-3)).astype(np.float32)

        return Data(
            x_base=torch.from_numpy(x_base),
            x_b62=torch.from_numpy(x_b62),
            x_esm=torch.from_numpy(x_esm),
            edge_index=torch.from_numpy(e["edge_index"].astype(np.int64)),
            edge_dist=torch.from_numpy(edge_dist),
            edge_inv_dist=torch.from_numpy(edge_inv_dist),
            edge_seqbin=torch.from_numpy(e["edge_seqbin"].astype(np.int64)),
            edge_is_seq=torch.from_numpy(e["edge_is_seq"].astype(np.float32)),
            y=torch.tensor([y], dtype=torch.float32),
            pdb_id=pid
        )


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
        dropout=0.1
    ):
        super().__init__()

        self.base_mlp = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, base_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_hidden, base_hidden),
            nn.GELU()
        )

        self.b62_mlp = nn.Sequential(
            nn.LayerNorm(b62_dim),
            nn.Linear(b62_dim, b62_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(b62_hidden, b62_hidden),
            nn.GELU()
        )

        self.esm_mlp = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, esm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(esm_hidden, esm_hidden),
            nn.GELU()
        )

        struct_dim = base_hidden + b62_hidden
        self.struct_mlp = nn.Sequential(
            nn.LayerNorm(struct_dim),
            nn.Linear(struct_dim, struct_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(struct_dim + esm_hidden, esm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(esm_hidden, esm_hidden),
            nn.Sigmoid()
        )

        fuse_dim = struct_dim + esm_hidden
        self.node_fuse = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, node_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.seq_emb = nn.Embedding(seqbin_vocab, seqbin_emb)
        self.edge_rbf_dim = edge_rbf_dim

        edge_raw_dim = edge_rbf_dim + seqbin_emb + 1 + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_raw_dim, node_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_hidden, node_hidden)
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            nn_edge = nn.Sequential(
                nn.Linear(node_hidden, node_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(node_hidden, node_hidden)
            )
            self.convs.append(GINEConv(nn_edge, edge_dim=node_hidden))

        self.norms = nn.ModuleList([nn.LayerNorm(node_hidden) for _ in range(num_layers)])
        self.dropout = dropout
        self.readout = nn.Sequential(
            nn.Linear(node_hidden * 2, node_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        h_base = self.base_mlp(data.x_base)
        h_b62 = self.b62_mlp(data.x_b62)
        h_struct = self.struct_mlp(torch.cat([h_base, h_b62], dim=1))

        h_esm = self.esm_mlp(data.x_esm)
        gate = self.gate_mlp(torch.cat([h_struct, h_esm], dim=1))
        h_esm = h_esm * gate

        x = self.node_fuse(torch.cat([h_struct, h_esm], dim=1))

        device = x.device
        rbf_feat = rbf(data.edge_dist.to(device), D_min=0.0, D_max=20.0, D_count=self.edge_rbf_dim, device=device)
        seq_feat = self.seq_emb(data.edge_seqbin.to(device))
        is_seq = data.edge_is_seq.to(device).view(-1, 1)
        inv_dist = data.edge_inv_dist.to(device).view(-1, 1)
        edge_raw = torch.cat([rbf_feat, seq_feat, is_seq, inv_dist], dim=1)
        edge_attr = self.edge_mlp(edge_raw)

        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, data.edge_index.to(device), edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_in

        h_mean = global_mean_pool(x, data.batch)
        h_max = global_max_pool(x, data.batch)
        return self.readout(torch.cat([h_mean, h_max], dim=1))


@torch.no_grad()
def export_h_graph(encoder, loader, device, out_csv):
    encoder.eval()
    rows = []

    for batch in loader:
        batch = batch.to(device)
        h = encoder(batch).cpu().numpy()
        ys = batch.y.view(-1).cpu().numpy()
        pids = batch.pdb_id

        for i, pid in enumerate(pids):
            r = {"PDB_ID": pid, "y": float(ys[i])}
            for j in range(h.shape[1]):
                r[f"h_graph_{j}"] = float(h[i, j])
            rows.append(r)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] saved h_graph: {out_csv} (n={len(rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_dir", required=True)
    ap.add_argument("--edge_dir", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--ckpt", required=True, help="graph_encoder_best.pt")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["encoder_state"] if isinstance(ckpt, dict) and "encoder_state" in ckpt else ckpt
    arch = ckpt.get("arch", {}) if isinstance(ckpt, dict) else {}

    ds = GraphOnlyDataset(args.node_dir, args.edge_dir, args.label_csv)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    enc = GINEEncoder(
        base_dim=arch.get("base_dim", 12),
        b62_dim=arch.get("b62_dim", 20),
        esm_dim=arch.get("esm_dim", 1280),
        base_hidden=arch.get("base_hidden", 64),
        b62_hidden=arch.get("b62_hidden", 64),
        esm_hidden=arch.get("esm_hidden", 256),
        node_hidden=arch.get("node_hidden", 256),
        edge_rbf_dim=arch.get("edge_rbf_dim", 32),
        seqbin_vocab=arch.get("seqbin_vocab", 9),
        seqbin_emb=arch.get("seqbin_emb", 16),
        num_layers=arch.get("num_layers", 5),
        dropout=arch.get("dropout", 0.1),
    ).to(device)

    enc.load_state_dict(state, strict=True)
    export_h_graph(enc, loader, device, args.out_csv)


if __name__ == "__main__":
    main()