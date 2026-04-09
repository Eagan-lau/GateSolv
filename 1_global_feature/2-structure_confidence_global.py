#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser


def normalize_protein_id(x: str) -> str:
    x = str(x).strip()
    if x.endswith(".ef.pdb"):
        x = x[:-7]
    elif x.endswith(".pdb"):
        x = x[:-4]
    return x


def get_chain_plddt(chain):
    plddts = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        if "CA" not in residue:
            continue
        bfactor = residue["CA"].get_bfactor()
        if np.isfinite(bfactor):
            plddts.append(float(bfactor))
    return np.asarray(plddts, dtype=float)


def longest_low_conf_segment_length(plddt_norm: np.ndarray, thr_01: float) -> int:
    best = 0
    current = 0
    for value in plddt_norm:
        if value < thr_01:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def compute_plddt_global(plddt: np.ndarray, thr_list=(50.0, 70.0)):
    if len(plddt) == 0:
        return None

    if np.nanmax(plddt) > 1.0001:
        plddt_norm = plddt / 100.0
        input_scale = "0_100"
    else:
        plddt_norm = plddt.copy()
        input_scale = "0_1"

    if np.nanmax(plddt_norm) > 1.05 or np.nanmin(plddt_norm) < -0.05:
        raise ValueError(
            f"Abnormal pLDDT range after normalization: "
            f"[{np.nanmin(plddt_norm):.3f}, {np.nanmax(plddt_norm):.3f}]"
        )

    std_val = float(np.std(plddt_norm, ddof=1)) if len(plddt_norm) > 1 else 0.0

    feats = {
        "n_res_plddt": int(plddt.size),
        "plddt_mean": float(np.mean(plddt_norm)),
        "plddt_median": float(np.median(plddt_norm)),
        "plddt_std": std_val,
        "plddt_min": float(np.min(plddt_norm)),
        "plddt_max": float(np.max(plddt_norm)),
        "plddt_input_scale": input_scale,
    }

    for raw_thr in thr_list:
        thr_01 = raw_thr / 100.0 if raw_thr > 1.0 else raw_thr
        suffix = str(raw_thr).replace(".", "p")
        mask = plddt_norm < thr_01
        feats[f"plddt_frac_below_{suffix}"] = float(mask.mean())
        feats[f"plddt_longest_below_{suffix}"] = int(longest_low_conf_segment_length(plddt_norm, thr_01))
    return feats


def worker(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    plddt = get_chain_plddt(chain)
    feats = compute_plddt_global(plddt)
    if feats is None:
        return None
    base = os.path.basename(pdb_path)
    feats["pdb_file"] = base
    feats["protein_id"] = normalize_protein_id(base)
    feats["chain_id"] = chain.id
    return feats


def main(pdb_dir: str, out_csv: str, out_csv_reduced: str, n_jobs: int):
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in: {pdb_dir}")

    rows = []
    failed = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_path = {executor.submit(worker, p): p for p in pdb_files}
        for future in as_completed(future_to_path):
            pdb_path = future_to_path[future]
            try:
                feats = future.result()
            except Exception as exc:
                failed += 1
                print(f"[WARN] failed {os.path.basename(pdb_path)}: {exc}")
                continue
            if feats is None:
                failed += 1
                print(f"[WARN] skipped {os.path.basename(pdb_path)}: no valid pLDDT values")
                continue
            rows.append(feats)

    if not rows:
        raise RuntimeError("No valid PDBs processed.")

    df = pd.DataFrame(rows).sort_values("pdb_file").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved full: {out_csv} (n={len(df)}, failed_or_skipped={failed})")

    reduced_cols = [
        "protein_id",
        "plddt_mean",
        "plddt_std",
        "plddt_min",
        "plddt_frac_below_50p0",
        "plddt_frac_below_70p0",
        "plddt_longest_below_50p0",
        "plddt_longest_below_70p0",
    ]
    reduced_cols = [c for c in reduced_cols if c in df.columns]
    reduced = df[reduced_cols].rename(columns={"protein_id": "PDB_ID"})
    reduced.to_csv(out_csv_reduced, index=False)
    print(f"[OK] saved reduced: {out_csv_reduced} (n={len(reduced)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw global pLDDT features from single-chain PDB files."
    )
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_csv_reduced", required=True)
    parser.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    main(args.pdb_dir, args.out_csv, args.out_csv_reduced, args.n_jobs)
