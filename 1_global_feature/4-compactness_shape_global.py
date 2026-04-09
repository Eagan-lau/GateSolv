#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import freesasa
from Bio.PDB import PDBParser
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Config
# ----------------------------
CONTACT_CUTOFF_A = 8.0          # Cbeta-Cbeta contact threshold; glycine uses Calpha
EXCLUDE_SEQ_NEIGHBOR = 2        # exclude contacts with |i-j| <= this
MIN_CHAIN_LENGTH = 20           # chains shorter than this use fallback behavior for some features
USE_CA_ONLY_FOR_RG = True

AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "MSE":"M"
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


def residue_key(res):
    hetflag, resseq, icode = res.id
    return (int(resseq), "" if icode == " " else str(icode).strip())


def get_rep_atom_coord(res):
    """CB (Gly->CA)"""
    if res.resname.upper() == "GLY":
        return np.array(res["CA"].coord, dtype=float) if "CA" in res else None
    if "CB" in res:
        return np.array(res["CB"].coord, dtype=float)
    if "CA" in res:
        return np.array(res["CA"].coord, dtype=float)
    return None


def get_ca_coords(residues):
    coords = []
    idx_map = []
    for i, res in enumerate(residues):
        if "CA" in res:
            coords.append(np.array(res["CA"].coord, dtype=float))
            idx_map.append(i)
    if not coords:
        return np.zeros((0, 3), dtype=float), []
    return np.vstack(coords), idx_map


def get_single_chain_residues(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    residues = [res for res in chain.get_residues() if res.id[0] == " " and aa3_to_aa1(res.resname)]
    return chain.id, residues


def radius_of_gyration(coords: np.ndarray):
    if coords.shape[0] < 3:
        return float("nan")
    com = coords.mean(axis=0)
    rg2 = np.mean(np.sum((coords - com) ** 2, axis=1))
    return float(np.sqrt(rg2))


def inertia_tensor(coords: np.ndarray):
    if coords.shape[0] < 3:
        return np.full((3, 3), np.nan)
    x = coords - coords.mean(axis=0)
    S = (x.T @ x) / coords.shape[0]
    return S


def shape_anisotropy_asphericity(coords: np.ndarray):
    S = inertia_tensor(coords)
    if np.any(np.isnan(S)):
        return float("nan"), float("nan")
    eigvals = np.linalg.eigvalsh(S)  # ascending
    l1, l2, l3 = float(eigvals[2]), float(eigvals[1]), float(eigvals[0])  # descending
    I1 = l1 + l2 + l3
    if I1 <= 1e-10:
        return float("nan"), float("nan")
    I2 = l1 * l2 + l2 * l3 + l3 * l1
    asphericity = l1 - 0.5 * (l2 + l3)
    kappa2 = 1.0 - 3.0 * (I2 / (I1 ** 2))
    kappa2 = max(0.0, min(1.0, float(kappa2)))
    return float(asphericity), float(kappa2)


def total_contacts_and_density(residues, cutoff_A=8.0, exclude_seq_neighbor=2):
    L = len(residues)
    if L == 0:
        return 0, float("nan")

    valid_coords = []
    valid_indices = []
    for i, res in enumerate(residues):
        c = get_rep_atom_coord(res)
        if c is not None:
            valid_coords.append(c)
            valid_indices.append(i)

    if len(valid_coords) < 2:
        return 0, 0.0

    coords = np.array(valid_coords)
    total = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            seq_dist = abs(valid_indices[i] - valid_indices[j])
            if seq_dist <= exclude_seq_neighbor:
                continue
            if np.linalg.norm(coords[i] - coords[j]) < cutoff_A:
                total += 1

    density = total / L
    return int(total), float(density)


def total_sasa_nm2(pdb_path: str, target_chain: str):
    """ SASA (nm²)"""
    try:
        fs_struct = freesasa.Structure(pdb_path)
        fs_res = freesasa.calc(fs_struct)
        total = 0.0
        for i in range(fs_struct.nAtoms()):
            if fs_struct.chainLabel(i) == target_chain:
                total += float(fs_res.atomArea(i))
        return float(total / 100.0)
    except Exception as e:
        print(f"[WARN] freesasa failed for {os.path.basename(pdb_path)}: {e}")
        return float("nan")


def compute_features_for_pdb(pdb_path: str):
    chain_id, residues = get_single_chain_residues(pdb_path)
    if not residues:
        return None

    L = len(residues)
    base = os.path.basename(pdb_path)

    # 
    if L < MIN_CHAIN_LENGTH:
        return {
            "pdb_file": base,
            "protein_id": normalize_protein_id(base),
            "chain_id": chain_id,
            "L_chain_residues": int(L),
            "Rg_A": float("nan"),
            "Rg_norm_L1_3": float("nan"),
            "total_contacts": 0,
            "contact_density": float("nan"),
            "total_SASA_nm2": float("nan"),
            "SASA_over_L_nm2": float("nan"),
            "asphericity": float("nan"),
            "anisotropy_kappa2": float("nan"),
            "note": "chain too short"
        }

    # coords for Rg and shape (use CA)
    ca_coords, _ = get_ca_coords(residues)
    rg = radius_of_gyration(ca_coords)
    rg_norm = rg / (L ** (1.0 / 3.0)) if L > 0 and not np.isnan(rg) else float("nan")

    # SASA
    sasa = total_sasa_nm2(pdb_path, chain_id)
    sasa_over_L = sasa / L if L > 0 and not np.isnan(sasa) else float("nan")

    # contacts
    total_contacts, contact_density = total_contacts_and_density(
        residues,
        cutoff_A=CONTACT_CUTOFF_A,
        exclude_seq_neighbor=EXCLUDE_SEQ_NEIGHBOR
    )

    # shape
    asphericity, anisotropy_k2 = shape_anisotropy_asphericity(ca_coords)

    return {
        "pdb_file": base,
        "protein_id": normalize_protein_id(base),
        "chain_id": chain_id,
        "L_chain_residues": int(L),
        "Rg_A": float(rg),
        "Rg_norm_L1_3": float(rg_norm),
        "total_contacts": int(total_contacts),
        "contact_density": float(contact_density),
        "total_SASA_nm2": float(sasa),
        "SASA_over_L_nm2": float(sasa_over_L),
        "asphericity": float(asphericity),
        "anisotropy_kappa2": float(anisotropy_k2),
        "contact_cutoff_A": float(CONTACT_CUTOFF_A),
        "exclude_seq_neighbor": int(EXCLUDE_SEQ_NEIGHBOR),
        "rg_atoms": "CA_only" if USE_CA_ONLY_FOR_RG else "representative",
    }


def _worker(pdb_path: str):
    try:
        return compute_features_for_pdb(pdb_path)
    except Exception as e:
        print(f"[ERROR] {os.path.basename(pdb_path)}: {e}")
        return None



def main(pdb_dir: str, out_csv: str, out_csv_reduced: str, n_jobs: int):
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in: {pdb_dir}")

    rows = []
    failed = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        future_to_path = {ex.submit(_worker, p): p for p in pdb_files}
        for fut in as_completed(future_to_path):
            pdb_path = future_to_path[fut]
            feats = fut.result()
            if feats is None:
                failed += 1
                continue
            rows.append(feats)

    if not rows:
        raise RuntimeError("No valid PDBs processed.")

    df = pd.DataFrame(rows).sort_values("pdb_file").reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    print(f"[OK] saved full: {out_csv} (n={len(df)}, failed={failed})")

    # Reduced CSV
    reduced_cols = [
        "protein_id",
        "Rg_norm_L1_3",
        "contact_density",
        "SASA_over_L_nm2",
        "anisotropy_kappa2",
        "asphericity",
    ]
    reduced_cols = [c for c in reduced_cols if c in df.columns]
    reduced = df[reduced_cols].rename(columns={"protein_id": "PDB_ID"})

    reduced.to_csv(out_csv_reduced, index=False)
    print(f"[OK] saved reduced: {out_csv_reduced} (n={len(reduced)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Extract raw compactness and shape features for single-chain PDB files."
    )
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_csv_reduced", required=True)
    ap.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    main(
        args.pdb_dir,
        args.out_csv,
        args.out_csv_reduced,
        args.n_jobs,
    )