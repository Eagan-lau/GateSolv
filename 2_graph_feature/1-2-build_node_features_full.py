#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import freesasa
from Bio.PDB import PDBParser, DSSP
from Bio.Align import substitution_matrices
import torch
import esm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import logging


logging.basicConfig(filename='failed_pdbs.log', level=logging.WARNING,
                    format='%(asctime)s - %(message)s')

CONTACT_CUTOFF_A = 8.0
EXCLUDE_SEQ_NEIGHBOR = 2
EPS = 1e-6
MAX_ASA = {
    'A': 121.0, 'R': 265.0, 'N': 187.0, 'D': 187.0, 'C': 148.0,
    'Q': 214.0, 'E': 214.0, 'G': 97.0, 'H': 216.0, 'I': 195.0,
    'L': 191.0, 'K': 230.0, 'M': 203.0, 'F': 228.0, 'P': 154.0,
    'S': 143.0, 'T': 163.0, 'W': 264.0, 'Y': 255.0, 'V': 165.0
}
AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"
}
SS_HELIX = {"H", "G", "I"}
SS_SHEET = {"E", "B"}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}
KD = {
    'A': 1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C': 2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
    'H':-3.2,'I': 4.5,'L': 3.8,'K':-3.9,'M': 1.9,'F': 2.8,'P':-1.6,'S':-0.8,
    'T':-0.7,'W':-0.9,'Y':-1.3,'V': 4.2
}
POLAR_SET = set("RNDQEKHSTYC")
AROMATIC_SET = set("FWYH")
POS_SET = set("KRH")
NEG_SET = set("DE")

DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"
DEFAULT_ESM_LAYERS = [30, 31, 32, 33]
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_FP16 = torch.cuda.is_available()
DEFAULT_MAX_LEN = 2500

def normalize_protein_id(x: str) -> str:
    x = str(x).strip()
    for suffix in [".ef.pdb", ".pdb", ".ef"]:
        if x.endswith(suffix):
            x = x[:-len(suffix)]
            break
    return x


def aa3_to_aa1(resname: str):
    return AA3_TO_AA1.get(resname.upper(), None)


def residue_key(res):
    hetflag, resseq, icode = res.id
    return (int(resseq), "" if icode == " " else str(icode).strip())


def get_single_chain_residues(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    residues = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        aa1 = aa3_to_aa1(res.resname)
        if aa1 is None:
            continue
        residues.append(res)
    return model, chain, residues


def get_plddt_per_residue(residues):
    vals = []
    for res in residues:
        if "CA" in res:
            plddt_value = float(res["CA"].get_bfactor())
            if plddt_value > 1:  
                plddt_value /= 100.0
            vals.append(plddt_value)
        else:
            vals.append(0.0) 
    return np.asarray(vals, dtype=np.float32)


def get_rep_coord_for_contact(res):
    if res.resname.upper() == "GLY":
        if "CA" in res:
            return np.asarray(res["CA"].coord, dtype=np.float32)
        return None
    if "CB" in res:
        return np.asarray(res["CB"].coord, dtype=np.float32)
    if "CA" in res:
        return np.asarray(res["CA"].coord, dtype=np.float32)
    return None


def contact_number_per_residue(residues, cutoff_A=8.0, exclude_seq_neighbor=2):
    L = len(residues)
    coords = [get_rep_coord_for_contact(res) for res in residues]
    coords_arr = np.array([c if c is not None else [np.nan]*3 for c in coords], dtype=np.float32)
    cn = np.zeros(L, dtype=np.int32)
    for i in range(L):
        ci = coords_arr[i]
        if np.any(np.isnan(ci)):
            continue
        for j in range(i + 1, L):
            if abs(i - j) <= exclude_seq_neighbor:
                continue
            cj = coords_arr[j]
            if np.any(np.isnan(cj)):
                continue
            if np.linalg.norm(ci - cj) < cutoff_A:
                cn[i] += 1
                cn[j] += 1
    return cn


def sasa_rsa_sidechain_per_residue(pdb_path: str, chain_id: str, residues):
    fs_struct = freesasa.Structure(pdb_path)
    fs_res = freesasa.calc(fs_struct)
    total_map = {}
    sc_map = {}
    has_icode = hasattr(fs_struct, "residueInsertionCode")
    total_chain_area = 0.0
    for i in range(fs_struct.nAtoms()):
        if fs_struct.chainLabel(i) != chain_id:
            continue
        area = float(fs_res.atomArea(i))
        total_chain_area += area
        resseq = int(fs_struct.residueNumber(i))
        icode = ""
        if has_icode:
            ic = fs_struct.residueInsertionCode(i)
            icode = "" if ic is None else str(ic).strip()
        key = (resseq, icode)
        total_map[key] = total_map.get(key, 0.0) + area
        atom_name = fs_struct.atomName(i).strip()
        if atom_name not in BACKBONE_ATOMS:
            sc_map[key] = sc_map.get(key, 0.0) + area
    L = len(residues)
    res_sasa = np.full(L, np.nan, dtype=np.float32)
    sc_sasa = np.full(L, np.nan, dtype=np.float32)
    rsa = np.full(L, np.nan, dtype=np.float32)
    for idx, res in enumerate(residues):
        key = residue_key(res)
        area_total = total_map.get(key, np.nan)
        area_sc = sc_map.get(key, 0.0)
        res_sasa[idx] = area_total
        sc_sasa[idx] = area_sc
        aa1 = aa3_to_aa1(res.resname)
        if aa1 in MAX_ASA and np.isfinite(area_total):
            rsa[idx] = area_total / MAX_ASA[aa1]
    return res_sasa, sc_sasa, rsa, total_chain_area


def ss_onehot_from_dssp(model, pdb_path: str, chain_id: str, residues):
    L = len(residues)
    onehot = np.zeros((L, 3), dtype=np.int8)
    onehot[:, 2] = 1  
    
    try:
        
        dssp = DSSP(model, pdb_path)
    except Exception as e:
        
        try:
            dssp = DSSP(model, pdb_path, chain=chain_id)
        except Exception as e2:
            logging.warning(f"DSSP failed for {os.path.basename(pdb_path)}: {e} | {e2}")
            return onehot
    
    
    ss_map = {}
    for k in dssp.keys():
        ch, resid = k[0], k[1]
        if ch != chain_id:
            continue
        hetflag, resseq, icode = resid
        if hetflag != " ":
            continue
        key = (int(resseq), "" if icode == " " else str(icode).strip())
        ss_map[key] = dssp[k][2]
    
    for i, res in enumerate(residues):
        code = ss_map.get(residue_key(res), "-")
        if code in SS_HELIX:
            onehot[i, 0] = 1
        elif code in SS_SHEET:
            onehot[i, 1] = 1
        else:
            onehot[i, 2] = 1
    return onehot


def residue_physchem_features(seq: str, rsa: np.ndarray):
    L = len(seq)
    hydropathy = np.zeros(L, dtype=np.float32)
    charge = np.zeros(L, dtype=np.float32)
    polarity = np.zeros(L, dtype=np.float32)
    aromatic = np.zeros(L, dtype=np.float32)
    surface_exposed = np.zeros(L, dtype=np.float32)
    for i, aa in enumerate(seq):
        hydropathy[i] = float(KD.get(aa, 0.0))
        if aa in POS_SET:
            charge[i] = 1.0
        elif aa in NEG_SET:
            charge[i] = -1.0
        polarity[i] = 1.0 if aa in POLAR_SET else 0.0
        aromatic[i] = 1.0 if aa in AROMATIC_SET else 0.0
        surface_exposed[i] = 1.0 if (np.isfinite(rsa[i]) and rsa[i] >= 0.20) else 0.0
    return hydropathy, charge, polarity, aromatic, surface_exposed


def blosum62_features(seq: str):
    matrix = substitution_matrices.load("BLOSUM62")
    L = len(seq)
    x = np.zeros((L, 20), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            idx = AA_TO_IDX[aa]
            x[i, idx] = 1.0  
    return x


def esm2_per_residue(seq: str, model, alphabet, layers, device: str) -> np.ndarray:
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=layers, return_contacts=False)
    reps = []
    for layer in layers:
        rep = out["representations"][layer][0]
        rep = rep[1:1+len(seq), :].contiguous()
        reps.append(rep.float())
    rep = torch.stack(reps, dim=0).mean(dim=0)
    return rep.cpu().numpy().astype(np.float32)


def process_one_base(args):
    pdb_path, out_dir = args
    base = os.path.basename(pdb_path)
    protein_id = normalize_protein_id(base)
    out_npz = os.path.join(out_dir, f"{protein_id}.node_features.base.npz")
    try:
        model, chain, residues = get_single_chain_residues(pdb_path)
        L = len(residues)
        if L == 0:
            return None
        aa = np.asarray([aa3_to_aa1(r.resname) for r in residues], dtype=object)
        seq = "".join(aa)
        plddt = get_plddt_per_residue(residues)
        plddt = np.nan_to_num(plddt, nan=0.0)  # nan 填充 0
        res_sasa_A2, sc_sasa_A2, rsa = sasa_rsa_sidechain_per_residue(pdb_path, chain, residues)[:3]
        rsa = np.nan_to_num(rsa, nan=0.0)
        ss_onehot = ss_onehot_from_dssp(model, pdb_path, chain.id, residues)
        contact_num = contact_number_per_residue(residues, CONTACT_CUTOFF_A, EXCLUDE_SEQ_NEIGHBOR)
        sc_sasa_ratio = np.full(L, 0.0, dtype=np.float32)
        valid = np.isfinite(res_sasa_A2) & (res_sasa_A2 > EPS)
        sc_sasa_ratio[valid] = sc_sasa_A2[valid] / res_sasa_A2[valid]
        hydropathy, charge, polarity, aromatic, surface_exposed = residue_physchem_features(seq, rsa)
        x_base = np.column_stack([
            plddt, rsa, contact_num, sc_sasa_ratio,
            ss_onehot, hydropathy, charge, polarity, aromatic, surface_exposed
        ]).astype(np.float32)
        np.savez_compressed(
            out_npz,
            aa=aa,
            x_base=x_base,
            base_feature_names=np.array([
                "plddt", "rsa", "contact_number", "sc_sasa_ratio",
                "ss_helix", "ss_sheet", "ss_coil",
                "hydropathy", "charge", "polarity", "aromatic", "surface_exposed"
            ], dtype=object)
        )
        return {
            "protein_id": protein_id,
            "name": protein_id,
            "pdb_file": base,
            "base_npz": out_npz,
            "L": int(L),
            "chain_id": chain.id
        }
    except Exception as e:
        logging.warning(f"failed(base) {base}: {e}")
        print(f"[WARN] failed(base) {base}: {e}")
        return None


def main(pdb_dir: str,
         out_dir: str,
         index_csv: str,
         nproc: int = 16,
         esm_model_name: str = DEFAULT_ESM_MODEL,
         esm_layers: list = DEFAULT_ESM_LAYERS,
         device: str = DEFAULT_DEVICE,
         fp16: bool = DEFAULT_FP16,
         max_len: int = DEFAULT_MAX_LEN):
    os.makedirs(out_dir, exist_ok=True)
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in: {pdb_dir}")

    
    tasks = [(p, out_dir) for p in pdb_files]
    rows = []
    with Pool(processes=nproc) as pool:
        for rec in pool.imap_unordered(process_one_base, tasks, chunksize=5):
            if rec is not None:
                rows.append(rec)

    if not rows:
        raise RuntimeError("No base node features generated.")

    df_base = pd.DataFrame(rows).sort_values("pdb_file").reset_index(drop=True)

    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model, alphabet = esm.pretrained.__dict__[esm_model_name]()
    model.eval()
    model = model.to(device)
    if fp16 and device.startswith("cuda"):
        model = model.half()

    out_rows = []
    all_esm_feats = []
    for _, r in df_base.iterrows():
        pdb_file = r["pdb_file"]
        protein_id = r["protein_id"]
        base_npz = r["base_npz"]
        out_npz = os.path.join(out_dir, f"{protein_id}.node_features.full.npz")
        try:
            npz = np.load(base_npz, allow_pickle=True)
            aa = npz["aa"].astype(object)
            seq = "".join(aa)
            L = len(seq)
            if L == 0:
                continue
            if max_len is not None and L > max_len:
                print(f"[WARN] skip {pdb_file}: L={L} > max_len={max_len}")
                continue
            x_base = npz["x_base"].astype(np.float32)
            x_b62 = blosum62_features(seq)
            x_esm = esm2_per_residue(seq, model, alphabet, layers=esm_layers, device=device)
            if x_esm.shape[0] != L:
                raise RuntimeError(f"ESM length mismatch: seq L={L}, esm={x_esm.shape[0]}")
            all_esm_feats.append(x_esm)
            x = np.concatenate([x_base, x_b62, x_esm], axis=1).astype(np.float32)
            np.savez_compressed(
                out_npz,
                aa=aa,
                x_base=x_base,
                x_b62=x_b62.astype(np.float32),
                x_esm=x_esm,
                x=x,
                base_dim=np.int32(x_base.shape[1]),
                b62_dim=np.int32(x_b62.shape[1]),
                esm_dim=np.int32(x_esm.shape[1]),
                x_dim=np.int32(x.shape[1]),
                esm_model=np.asarray([esm_model_name], dtype=object),
                esm_layers=np.asarray(esm_layers, dtype=object),
            )
            out_rows.append({
                "protein_id": protein_id,
                "name": protein_id,
                "pdb_file": pdb_file,
                "npz_path": out_npz,
                "L": int(L),
                "chain_id": r["chain_id"],
                "base_dim": int(x_base.shape[1]),
                "b62_dim": int(x_b62.shape[1]),
                "esm_dim": int(x_esm.shape[1]),
                "x_dim": int(x.shape[1]),
            })
        except Exception as e:
            logging.warning(f"failed(full) {pdb_file}: {e}")
            print(f"[WARN] failed(full) {pdb_file}: {e}")
            continue

    if not out_rows:
        raise RuntimeError("No full node features generated successfully.")

    
    if all_esm_feats:
        all_esm = np.concatenate(all_esm_feats, axis=0)
        pca = PCA(n_components=50)
        all_esm_reduced = pca.fit_transform(all_esm)
        
        print("[INFO] PCA applied to all ESM features globally.")

    df = pd.DataFrame(out_rows).sort_values("pdb_file").reset_index(drop=True)
    df.to_csv(index_csv, index=False)
    print(f"[OK] full node npz saved in: {out_dir}")
    print(f"[OK] index saved: {index_csv} (n={len(df)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build node features: base + BLOSUM62 + ESM2(last-4 mean).")
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--esm_model_name", default=DEFAULT_ESM_MODEL)
    ap.add_argument("--esm_layers", default=DEFAULT_ESM_LAYERS,
                    help="Comma-separated layer indices, e.g. 30,31,32,33")
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    args = ap.parse_args()

    
    if isinstance(args.esm_layers, list):
        esm_layers = [int(x) for x in args.esm_layers]
    else:
        esm_layers = [int(x.strip()) for x in str(args.esm_layers).split(",") if x.strip()]

    main(
        pdb_dir=args.pdb_dir,
        out_dir=args.out_dir,
        index_csv=args.index_csv,
        nproc=args.nproc,
        esm_model_name=args.esm_model_name,
        esm_layers=esm_layers,  
        device=args.device,
        fp16=args.fp16,
        max_len=args.max_len
    )