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
RSA_THR = 0.20
PATCH_DIST = 6.0  # angstrom
HYDRO_SET = set("AVILMFWY")
POS_SET = set("KRH")
NEG_SET = set("DE")

MAX_ASA = {
    'A': 121.0, 'R': 265.0, 'N': 187.0, 'D': 187.0, 'C': 148.0,
    'Q': 214.0, 'E': 214.0, 'G': 97.0, 'H': 216.0, 'I': 195.0,
    'L': 191.0, 'K': 230.0, 'M': 203.0, 'F': 228.0, 'P': 154.0,
    'S': 143.0, 'T': 163.0, 'W': 264.0, 'Y': 255.0, 'V': 165.0
}

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M"
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
    """Use CB coordinates, or CA for glycine."""
    if res.resname.upper() == "GLY":
        return np.array(res["CA"].coord, dtype=float) if "CA" in res else None
    if "CB" in res:
        return np.array(res["CB"].coord, dtype=float)
    if "CA" in res:
        return np.array(res["CA"].coord, dtype=float)
    return None


def atom_is_polar(atom_name: str) -> bool:
    an = atom_name.strip().lstrip("0123456789")
    if not an:
        return False
    return an[0].upper() in ("N", "O")


def build_patch_components(nodes, coords, dist_thr):
    if not nodes:
        return []
    adj = {n: [] for n in nodes}
    for i in range(len(nodes)):
        ni = nodes[i]
        ci = coords.get(ni)
        if ci is None:
            continue
        for j in range(i + 1, len(nodes)):
            nj = nodes[j]
            cj = coords.get(nj)
            if cj is None:
                continue
            if np.linalg.norm(ci - cj) < dist_thr:
                adj[ni].append(nj)
                adj[nj].append(ni)
    comps = []
    visited = set()
    for n in nodes:
        if n in visited:
            continue
        stack = [n]
        visited.add(n)
        comp = [n]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
                    comp.append(v)
        comps.append(comp)
    return comps


def aggregate_atom_areas(fs_struct, fs_result, target_chain):
    per_res_sasa = {}
    polar = 0.0
    apolar = 0.0
    total_chain_area = 0.0

    has_icode = hasattr(fs_struct, "residueInsertionCode")

    for i in range(fs_struct.nAtoms()):
        if fs_struct.chainLabel(i) != target_chain:
            continue
        area = float(fs_result.atomArea(i))
        total_chain_area += area

        resseq = int(fs_struct.residueNumber(i))
        icode = ""
        if has_icode:
            ic = fs_struct.residueInsertionCode(i)
            icode = "" if ic is None else str(ic).strip()
        key = (resseq, icode)

        per_res_sasa[key] = per_res_sasa.get(key, 0.0) + area

        atom_name = fs_struct.atomName(i)
        if atom_is_polar(atom_name):
            polar += area
        else:
            apolar += area

    return per_res_sasa, polar, apolar, total_chain_area, target_chain


def compute_surface_features_single_chain(pdb_path: str):
    # Bio.PDB 
    parser = PDBParser(QUIET=True)
    base = os.path.basename(pdb_path)
    structure = parser.get_structure(base, pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    chain_id = chain.id
    residues = [res for res in chain.get_residues() if res.id[0] == " " and aa3_to_aa1(res.resname)]
    if not residues:
        return None

    pdb_res_map = {residue_key(r): r for r in residues}
    L = len(residues)

    # FreeSASA 
    fs_struct = freesasa.Structure(pdb_path)
    fs_result = freesasa.calc(fs_struct)

    # 
    per_res_sasa, polar_A2, apolar_A2, total_sasa_A2, used_chain = \
        aggregate_atom_areas(fs_struct, fs_result, chain_id)

    chain_id = used_chain

    if total_sasa_A2 < 1e-6:
        return None  # ，

    polar_ratio = polar_A2 / total_sasa_A2
    apolar_ratio = apolar_A2 / total_sasa_A2

    # RSA & 
    surface_keys = []
    for key, area in per_res_sasa.items():
        if key not in pdb_res_map:
            continue
        aa1 = aa3_to_aa1(pdb_res_map[key].resname)
        if aa1 not in MAX_ASA:
            continue
        rsa = area / MAX_ASA[aa1]
        if rsa >= RSA_THR:
            surface_keys.append(key)

    surface_count = len(surface_keys)
    surface_ratio = surface_count / L if L > 0 else 0.0

    # 
    surf_hydro_keys = [k for k in surface_keys
                       if aa3_to_aa1(pdb_res_map[k].resname) in HYDRO_SET]
    surf_hydro_count = len(surf_hydro_keys)
    surf_hydro_ratio = surf_hydro_count / surface_count if surface_count > 0 else 0.0

    surf_hydro_area_A2 = sum(per_res_sasa.get(k, 0.0) for k in surf_hydro_keys)
    surf_hydro_area_nm2 = surf_hydro_area_A2 / 100.0
    surf_hydro_area_ratio = surf_hydro_area_A2 / total_sasa_A2 if total_sasa_A2 > 0 else 0.0

    # 
    surf_pos = sum(1 for k in surface_keys
                   if aa3_to_aa1(pdb_res_map[k].resname) in POS_SET)
    surf_neg = sum(1 for k in surface_keys
                   if aa3_to_aa1(pdb_res_map[k].resname) in NEG_SET)

    surf_charge_count = surf_pos + surf_neg
    surf_charge_ratio = surf_charge_count / surface_count if surface_count > 0 else 0.0
    surf_pos_ratio = surf_pos / surface_count if surface_count > 0 else 0.0
    surf_neg_ratio = surf_neg / surface_count if surface_count > 0 else 0.0

    surface_net_charge_proxy = surf_pos_ratio - surf_neg_ratio

    # 
    coords = {k: get_rep_atom_coord(pdb_res_map[k]) for k in surf_hydro_keys}
    comps = build_patch_components(surf_hydro_keys, coords, PATCH_DIST)
    patch_areas_A2 = [sum(per_res_sasa.get(k, 0.0) for k in comp) for comp in comps]
    patch_areas_nm2 = [a / 100.0 for a in patch_areas_A2]

    patch_count = len(patch_areas_nm2)
    patch_mean_nm2 = np.mean(patch_areas_nm2) if patch_count > 0 else 0.0
    patch_max_nm2 = np.max(patch_areas_nm2) if patch_count > 0 else 0.0

    feats = {
        "pdb_file": base,
        "protein_id": normalize_protein_id(base),
        "chain_id": chain_id,
        "L_chain_residues": int(L),
        # SASA
        "total_SASA_A2": float(total_sasa_A2),
        "total_SASA_nm2": float(total_sasa_A2 / 100.0),
        "polar_SASA_A2": float(polar_A2),
        "apolar_SASA_A2": float(apolar_A2),
        "polar_SASA_ratio": float(polar_ratio),
        "apolar_SASA_ratio": float(apolar_ratio),
        # surface
        "surface_residue_count": int(surface_count),
        "surface_residue_ratio": float(surface_ratio),
        # surface hydrophobic
        "surface_hydrophobic_res_count": int(surf_hydro_count),
        "surface_hydrophobic_res_ratio": float(surf_hydro_ratio),
        "surface_hydrophobic_area_nm2": float(surf_hydro_area_nm2),
        "surface_hydrophobic_area_ratio": float(surf_hydro_area_ratio),
        # hydrophobic patches
        "hydrophobic_patch_count": int(patch_count),
        "hydrophobic_patch_mean_area_nm2": float(patch_mean_nm2),
        "hydrophobic_patch_max_area_nm2": float(patch_max_nm2),
        # surface charge
        "surface_pos_res_count": int(surf_pos),
        "surface_neg_res_count": int(surf_neg),
        "surface_charged_res_count": int(surf_charge_count),
        "surface_charged_res_ratio": float(surf_charge_ratio),
        "surface_pos_res_ratio": float(surf_pos_ratio),
        "surface_neg_res_ratio": float(surf_neg_ratio),
        "surface_net_charge_proxy": float(surface_net_charge_proxy),
        # meta / parameters
        "RSA_thr": float(RSA_THR),
        "patch_dist_A": float(PATCH_DIST),
        "freesasa_classifier": "default (ProtOr radii)",
        "freesasa_probe_radius": 1.4,
        "polar_apolar_rule": "N/O polar; others apolar (atomName-based)",
        "residue_sasa_source": "atomArea aggregated (no residueAreas API)",
    }
    return feats


def _worker(pdb_path: str):
    try:
        return compute_surface_features_single_chain(pdb_path)
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

    # reduced CSV
    reduced_cols = [
        "protein_id",
        "polar_SASA_ratio",
        "apolar_SASA_ratio",
        "surface_residue_ratio",
        "surface_hydrophobic_res_ratio",
        "surface_hydrophobic_area_ratio",
        "hydrophobic_patch_count",
        "hydrophobic_patch_max_area_nm2",
        "surface_charged_res_ratio",
        "surface_net_charge_proxy",
    ]
    reduced_cols = [c for c in reduced_cols if c in df.columns]
    reduced = df[reduced_cols].rename(columns={"protein_id": "PDB_ID"})

    reduced.to_csv(out_csv_reduced, index=False)
    print(f"[OK] saved reduced: {out_csv_reduced} (n={len(reduced)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Extract raw surface physicochemical and aggregation-risk features for single-chain PDB files."
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