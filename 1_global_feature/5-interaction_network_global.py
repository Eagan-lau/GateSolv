#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Parameters (tune if needed)
# ----------------------------
DISULFIDE_SG_DIST = 2.2      # Å
SALT_BRIDGE_DIST = 4.0       # Å
HBOND_DA_DIST = 3.5          # Å
MIN_SEQ_SEP = 4              # minimum sequence separation for non-local contacts

# Atom sets
NEG_O_ATOMS = {"OD1", "OD2", "OE1", "OE2"}
NEG_RES = {"ASP", "GLU"}
POS_RES = {"LYS", "ARG"}
LYS_ATOMS = {"NZ"}
ARG_ATOMS = {"NH1", "NH2"}

AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"
}

def normalize_protein_id(x: str) -> str:
    x = str(x).strip()
    if x.endswith(".ef.pdb"):
        x = x[:-7]
    elif x.endswith(".pdb"):
        x = x[:-4]
    return x

def aa3_to_aa1(resname: str):
    return AA3_TO_AA1.get(resname.upper(), None)

def get_single_chain_standard_residues(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    residues = []
    for res in chain.get_residues():
        if res.id[0] != " " or aa3_to_aa1(res.resname) is None:
            continue
        residues.append(res)
    return chain.id, residues

def atom_coord(res, atom_name):
    return np.array(res[atom_name].coord, dtype=float) if atom_name in res else None

def disulfide_bonds(residues):
    """Count Cys–Cys disulfides (no seq-sep filter needed)"""
    cys = [(i, np.array(res["SG"].coord, dtype=float))
           for i, res in enumerate(residues)
           if res.resname.upper() == "CYS" and "SG" in res]
    
    pair_set = set()
    for a in range(len(cys)):
        i, ci = cys[a]
        for b in range(a + 1, len(cys)):
            j, cj = cys[b]
            if np.linalg.norm(ci - cj) <= DISULFIDE_SG_DIST:
                pair_set.add((i, j))
    return len(pair_set), pair_set

def salt_bridges(residues):
    """Salt bridges with seq-sep filter"""
    neg_atoms = []
    pos_atoms = []
    for i, res in enumerate(residues):
        rn = res.resname.upper()
        if rn in NEG_RES:
            for an in NEG_O_ATOMS:
                c = atom_coord(res, an)
                if c is not None:
                    neg_atoms.append((i, c))
        elif rn == "LYS":
            c = atom_coord(res, "NZ")
            if c is not None:
                pos_atoms.append((i, c))
        elif rn == "ARG":
            for an in ARG_ATOMS:
                c = atom_coord(res, an)
                if c is not None:
                    pos_atoms.append((i, c))

    pair_set = set()
    for i, ci in neg_atoms:
        for j, cj in pos_atoms:
            if abs(i - j) < 2 or np.linalg.norm(ci - cj) > SALT_BRIDGE_DIST:
                continue
            pair_set.add((min(i, j), max(i, j)))
    return len(pair_set), pair_set

def hydrogen_bonds_distance_only(residues):
    """Improved: ONLY non-local (|i-j| >= MIN_SEQ_SEP) distance-only H-bonds"""
    donors = []
    acceptors = []
    for i, res in enumerate(residues):
        # Backbone
        if "N" in res:
            donors.append((i, np.array(res["N"].coord, dtype=float)))
        if "O" in res:
            acceptors.append((i, np.array(res["O"].coord, dtype=float)))
        # Sidechain
        for an in ["OD1","OD2","OE1","OE2","ND1","NE2","OG","OG1","OH","ND2","NE","NZ"]:
            if an in res:
                coord = np.array(res[an].coord, dtype=float)
                if an in ["N","ND1","ND2","NE","NE2","NZ","OG","OG1","OH"]:   # donors
                    donors.append((i, coord))
                else:                                                          # acceptors
                    acceptors.append((i, coord))

    pair_set = set()
    for i, ci in donors:
        for j, cj in acceptors:
            if i == j or abs(i - j) < MIN_SEQ_SEP:
                continue
            if np.linalg.norm(ci - cj) <= HBOND_DA_DIST:
                pair_set.add((min(i, j), max(i, j)))
    return len(pair_set), pair_set

def compute_interaction_global_features(pdb_path: str):
    chain_id, residues = get_single_chain_standard_residues(pdb_path)
    L = len(residues)
    if L == 0:
        return None

    dis_count, _ = disulfide_bonds(residues)
    salt_count, _ = salt_bridges(residues)
    hb_count, _ = hydrogen_bonds_distance_only(residues)

    n_cys = sum(1 for r in residues if r.resname.upper() == "CYS")

    base = os.path.basename(pdb_path)
    return {
        "pdb_file": base,
        "protein_id": normalize_protein_id(base),
        "chain_id": chain_id,
        "L": int(L),
        # disulfide
        "disulfide_bonds_count": int(dis_count),
        "n_cys": int(n_cys),
        "disulfide_bonds_density_per_L": float(dis_count / L),
        "disulfide_paired_cys_fraction": float(min(1.0, 2 * dis_count / max(1, n_cys))),  # FIXED
        # ionic
        "ionic_bonds_count": int(salt_count),
        "ionic_bonds_density_per_L": float(salt_count / L),
        # hydrogen bonds (now non-local)
        "hbond_pairs_count": int(hb_count),
        "hbond_density_per_L": float(hb_count / L),
        # reproducibility
        "DISULFIDE_SG_DIST_A": float(DISULFIDE_SG_DIST),
        "SALT_BRIDGE_DIST_A": float(SALT_BRIDGE_DIST),
        "HBOND_DA_DIST_A": float(HBOND_DA_DIST),
        "min_seq_sep_nonlocal": int(MIN_SEQ_SEP),
        "salt_bridge_cations_included": "LYS,ARG",
        "hbond_mode": "distance_only_nonlocal",
    }

# Worker
def _worker(pdb_path: str):
    feats = compute_interaction_global_features(pdb_path)
    if feats is None:
        print(f"[WARN] skipped {os.path.basename(pdb_path)} (no residues)")
        return None
    return feats

# Main
def main(pdb_dir: str, out_csv: str, out_csv_reduced: str, n_jobs: int):
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in: {pdb_dir}")

    rows = []
    failed = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        future_to_path = {ex.submit(_worker, p): p for p in pdb_files}
        for fut in as_completed(future_to_path):
            try:
                feats = fut.result()
                if feats:
                    rows.append(feats)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"[WARN] failed {os.path.basename(future_to_path[fut])}: {e}")

    if not rows:
        raise RuntimeError("No valid PDBs processed.")

    df = pd.DataFrame(rows).sort_values("pdb_file").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved full: {out_csv} (n={len(df)}, failed={failed})")

    # Reduced CSV（4， ML）
    reduced = pd.DataFrame({
        "PDB_ID": df["protein_id"].astype(str),
        "disulfide_bonds_density_per_L": df["disulfide_bonds_density_per_L"].astype(float),
        "disulfide_paired_cys_fraction": df["disulfide_paired_cys_fraction"].astype(float),   # 
        "ionic_bonds_density_per_L": df["ionic_bonds_density_per_L"].astype(float),
        "hbond_density_per_L": df["hbond_density_per_L"].astype(float),                     # 
    })
    reduced.to_csv(out_csv_reduced, index=False)
    print(f"[OK] saved reduced: {out_csv_reduced} (n={len(reduced)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract raw interaction-network features for single-chain PDB files.")
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_csv_reduced", required=True)
    ap.add_argument("--n_jobs", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()
    main(args.pdb_dir, args.out_csv, args.out_csv_reduced, args.n_jobs)