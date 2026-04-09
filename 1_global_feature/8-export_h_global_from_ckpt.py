#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Export learned h_global embeddings using a trained global encoder checkpoint."
    )
    parser.add_argument("--global_csv", required=True)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to global_encoder_best.pt")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    gdf = pd.read_csv(args.global_csv)
    ldf = pd.read_csv(args.label_csv)
    df = ldf.merge(gdf, on="PDB_ID", how="inner")
    if len(df) == 0:
        raise RuntimeError("No rows after merge on PDB_ID.")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    feat_cols = ckpt["feat_cols"]
    df = df.dropna(subset=feat_cols).copy()

    for col in feat_cols:
        if col not in df.columns:
            raise ValueError(f"Missing feature column in merged dataframe: {col}")

    X = df[feat_cols].astype(np.float32).to_numpy()
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    Xs = (X - mean) / (scale + 1e-12)

    if np.any(~np.isfinite(Xs)):
        raise ValueError("Found NaN or Inf after scaling. Check the input global feature CSV.")

    enc = GlobalEncoder(
        g_in=Xs.shape[1],
        g_hidden=int(ckpt["args"]["g_hidden"]),
        dropout=float(ckpt["args"]["dropout"]),
    )
    enc.load_state_dict(ckpt["encoder_state"], strict=True)
    enc = enc.to(device).eval()

    H = enc(torch.from_numpy(Xs).to(device)).cpu().numpy()

    rows = []
    has_label = "y" in df.columns
    for i, pid in enumerate(df["PDB_ID"].astype(str).tolist()):
        row = {"PDB_ID": pid}
        if has_label:
            row["y"] = float(df["y"].iloc[i])
        for j in range(H.shape[1]):
            row[f"h_global_{j}"] = float(H[i, j])
        rows.append(row)

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[OK] saved h_global embeddings: {args.out_csv} (n={len(rows)})")


if __name__ == "__main__":
    main()
