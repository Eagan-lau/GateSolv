#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as IP
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
HYDRO_SET = set("ILVFMWYAC")


def normalize_protein_id(x: str) -> str:
    x = str(x).strip()
    if x.endswith(".ef.pdb"):
        x = x[:-7]
    elif x.endswith(".pdb"):
        x = x[:-4]
    return x


def shannon_entropy(window: str) -> float:
    counts = Counter(window)
    n = len(window)
    if n == 0:
        return 0.0
    entropy = 0.0
    for v in counts.values():
        p = v / n
        entropy -= p * math.log(p, 2)
    return entropy


def low_complexity_ratio(seq: str, w: int = 15, ent_thresh: float = 2.2) -> float:
    seq = seq.upper()
    length = len(seq)
    if length < w:
        return 0.0
    cover = np.zeros(length, dtype=np.int8)
    for i in range(0, length - w + 1):
        window = seq[i:i + w]
        if any(ch not in KD for ch in window):
            continue
        if shannon_entropy(window) < ent_thresh:
            cover[i:i + w] = 1
    return float(cover.mean())


def repeat_ratio_kmer(seq: str, k: int = 3) -> float:
    seq = seq.upper()
    length = len(seq)
    if length < k:
        return 0.0
    positions = defaultdict(list)
    for i in range(length - k + 1):
        kmer = seq[i:i + k]
        if any(ch not in KD for ch in kmer):
            continue
        positions[kmer].append(i)
    cover = np.zeros(length, dtype=np.int8)
    for starts in positions.values():
        if len(starts) > 1:
            for s in starts:
                cover[s:s + k] = 1
    return float(cover.mean())


def max_hydro_window_mean(seq: str, w: int = 9, default: float = 0.0) -> float:
    seq = seq.upper()
    values = [KD.get(aa, None) for aa in seq]
    best = None
    for i in range(0, len(values) - w + 1):
        window = values[i:i + w]
        if any(v is None for v in window):
            continue
        mean_val = sum(window) / w
        best = mean_val if best is None else max(best, mean_val)
    return float(best) if best is not None else default


def max_hydrophobic_run(seq: str):
    seq = seq.upper()
    best_len = 0
    best_mean = 0.0
    current_values = []
    current_len = 0
    for aa in seq:
        if aa in HYDRO_SET:
            current_len += 1
            current_values.append(KD[aa])
        else:
            if current_len > best_len:
                best_len = current_len
                best_mean = sum(current_values) / current_len if current_len else 0.0
            current_len = 0
            current_values = []
    if current_len > best_len:
        best_len = current_len
        best_mean = sum(current_values) / current_len if current_len else 0.0
    return int(best_len), float(best_mean)


def aa_composition(seq: str):
    seq = seq.upper()
    counts = Counter(seq)
    valid_len = sum(counts[a] for a in AA_ORDER)
    if valid_len == 0:
        return [0.0] * 20, 1.0, 0
    vec = [counts[a] / valid_len for a in AA_ORDER]
    nonstd_ratio = (len(seq) - valid_len) / len(seq) if len(seq) else 0.0
    return vec, float(nonstd_ratio), int(valid_len)


def compute_sequence_features(
    seq: str,
    ph_points=(5.5, 6.5, 7.5, 8.5, 9.5),
    window_entropy_w: int = 15,
    window_entropy_thresh: float = 2.2,
    min_len_for_window: int = 8,
):
    seq = seq.upper()
    seq_std = "".join([aa for aa in seq if aa in AA_ORDER])
    if len(seq_std) == 0:
        return None

    length = len(seq_std)
    comp20, nonstd_ratio, valid_len = aa_composition(seq)

    pa = ProteinAnalysis(seq_std)
    mw = pa.molecular_weight()
    pi = pa.isoelectric_point()
    gravy_val = pa.gravy()

    ip = IP(seq_std)
    charges = {f"net_charge_pH_{p:.1f}": float(ip.charge_at_pH(p)) for p in ph_points}

    q_75 = charges.get("net_charge_pH_7.5")
    q_65 = charges.get("net_charge_pH_6.5")
    q_85 = charges.get("net_charge_pH_8.5")

    charge_density_75 = float(q_75) / length if q_75 is not None and length > 0 else 0.0
    abs_charge_density_75 = abs(float(q_75)) / length if q_75 is not None and length > 0 else 0.0
    delta_charge_85_65 = (float(q_85) - float(q_65)) if (q_85 is not None and q_65 is not None) else 0.0

    if length < min_len_for_window:
        max_win9 = 0.0
        max_win15 = 0.0
        max_run_len = 0
        max_run_mean = 0.0
    else:
        max_win9 = max_hydro_window_mean(seq_std, w=9, default=0.0)
        max_win15 = max_hydro_window_mean(seq_std, w=15, default=0.0)
        max_run_len, max_run_mean = max_hydrophobic_run(seq_std)

    lcr = low_complexity_ratio(seq_std, w=window_entropy_w, ent_thresh=window_entropy_thresh)
    rep3 = repeat_ratio_kmer(seq_std, k=3)

    feats = {
        "L": int(length),
        "MW": float(mw),
        "pI": float(pi),
        "GRAVY": float(gravy_val),
        "max_hydro_window_mean_w9": float(max_win9),
        "max_hydro_window_mean_w15": float(max_win15),
        "max_hydrophobic_run_len": int(max_run_len),
        "max_hydrophobic_run_mean": float(max_run_mean),
        "low_complexity_ratio_w15_H2.2": float(lcr),
        "repeat_ratio_k3": float(rep3),
        "nonstd_ratio_in_raw_seq": float(nonstd_ratio),
        "valid_len_used": int(valid_len),
        "seq_len_raw_extracted": int(len(seq)),
        "charge_density_pH_7.5": float(charge_density_75),
        "abs_charge_density_pH_7.5": float(abs_charge_density_75),
        "delta_charge_pH_8.5_6.5": float(delta_charge_85_65),
    }
    feats.update(charges)
    for aa, value in zip(AA_ORDER, comp20):
        feats[f"aa_frac_{aa}"] = float(value)
    return feats


def extract_single_chain_sequence_from_pdb(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    peptides = PPBuilder().build_peptides(chain)
    if not peptides:
        return chain.id, ""
    seq = "".join(str(pp.get_sequence()) for pp in peptides)
    return chain.id, seq


def worker(pdb_path: str):
    chain_id, seq = extract_single_chain_sequence_from_pdb(pdb_path)
    if not seq:
        return None
    feats = compute_sequence_features(seq)
    if feats is None:
        return None
    base = os.path.basename(pdb_path)
    feats["pdb_file"] = base
    feats["protein_id"] = normalize_protein_id(base)
    feats["chain_id"] = chain_id
    feats["representative"] = "single_chain"
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
                print(f"[WARN] skipped {os.path.basename(pdb_path)}: no valid sequence extracted")
                continue
            rows.append(feats)

    if not rows:
        raise RuntimeError("No valid PDBs processed.")

    df = pd.DataFrame(rows).sort_values("pdb_file").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved full: {out_csv} (n={len(df)}, failed_or_skipped={failed})")

    reduced_cols = [
        "protein_id",
        "L",
        "MW",
        "pI",
        "GRAVY",
        "max_hydro_window_mean_w9",
        "max_hydro_window_mean_w15",
        "max_hydrophobic_run_len",
        "max_hydrophobic_run_mean",
        "low_complexity_ratio_w15_H2.2",
        "repeat_ratio_k3",
        "nonstd_ratio_in_raw_seq",
        "valid_len_used",
        "seq_len_raw_extracted",
        "charge_density_pH_7.5",
        "abs_charge_density_pH_7.5",
        "delta_charge_pH_8.5_6.5",
        "net_charge_pH_5.5",
        "net_charge_pH_6.5",
        "net_charge_pH_7.5",
        "net_charge_pH_8.5",
        "net_charge_pH_9.5",
    ]
    reduced_cols = [c for c in reduced_cols if c in df.columns]
    reduced = df[reduced_cols].rename(columns={"protein_id": "PDB_ID"}).copy()

    for aa in AA_ORDER:
        col = f"aa_frac_{aa}"
        if col in df.columns:
            reduced[col] = df[col].astype(float)

    reduced.to_csv(out_csv_reduced, index=False)
    print(f"[OK] saved reduced: {out_csv_reduced} (n={len(reduced)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw global sequence features for single-chain PDB files."
    )
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--out_csv", required=True, help="Output full CSV path")
    parser.add_argument("--out_csv_reduced", required=True, help="Output reduced CSV path")
    parser.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    main(args.pdb_dir, args.out_csv, args.out_csv_reduced, args.n_jobs)
