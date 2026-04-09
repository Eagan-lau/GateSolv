#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from Bio.PDB import PDBParser
import warnings
import logging


logging.basicConfig(filename='failed_pdbs.log', level=logging.WARNING, 
                    format='%(asctime)s - %(message)s')


try:
    from scipy.spatial import cKDTree as KDTree
    _KNN_BACKEND = "scipy"
except ImportError:
    KDTree = None
    _KNN_BACKEND = "sklearn"

try:
    from sklearn.neighbors import NearestNeighbors
    _HAS_SK = True
except ImportError:
    _HAS_SK = False

KNN_K = 24
SEQ_BINS = [1, 2, 3, 4, 8, 16, 32, 64]
MIN_LENGTH = 10
DIST_EPS = 1e-6

def normalize_protein_id(x: str) -> str:
    x = str(x).strip()
    for suffix in [".ef.pdb", ".pdb", ".ef"]:
        if x.endswith(suffix):
            x = x[:-len(suffix)]
            break
    return x


def seq_bin_id(d: int, bins=SEQ_BINS) -> int:
    for bi, b in enumerate(bins):
        if d <= b:
            return bi
    return len(bins)


def get_single_chain_ca_coords(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    model = next(structure.get_models())
    chain = next(model.get_chains())
    coords = []
    res_index = []
    for i, res in enumerate(chain.get_residues()):
        if res.id[0] != " ":
            continue
        if "CA" not in res:
            continue
        ca_coord = np.asarray(res["CA"].coord, dtype=np.float32)
        if not np.all(np.isfinite(ca_coord)):
            continue
        coords.append(ca_coord)
        
        res_index.append(i)  
    if len(coords) < MIN_LENGTH:
        return None
    coords = np.asarray(coords, dtype=np.float32)
    res_index = np.asarray(res_index, dtype=np.int32)
    return coords, res_index, chain.id


def knn_edges(coords: np.ndarray, k: int):
    L = coords.shape[0]
    if L < 2:
        return np.zeros((2, 0), dtype=np.int32)
    
    kk = min(k + 1, L)
    
    if KDTree is not None:
        tree = KDTree(coords)
        dists, idx = tree.query(coords, k=kk)
    else:
        if not _HAS_SK:
            raise RuntimeError("Neither scipy.spatial.cKDTree nor sklearn.neighbors available.")
        nn = NearestNeighbors(n_neighbors=kk, algorithm="auto")
        nn.fit(coords)
        dists, idx = nn.kneighbors(return_distance=True)
    
    if kk > 1:
        idx = idx[:, 1:]
    else:
        idx = np.empty((L, 0), dtype=np.int32)
    
    src = np.repeat(np.arange(L, dtype=np.int32), idx.shape[1])
    dst = idx.ravel().astype(np.int32)
    
    return np.vstack([src, dst])


def seq_edges(L: int):
    if L < 2:
        return np.zeros((2, 0), dtype=np.int32)
    src = np.arange(L - 1, dtype=np.int32)
    dst = src + 1
    edge_src = np.concatenate([src, dst])
    edge_dst = np.concatenate([dst, src])
    return np.vstack([edge_src, edge_dst])


def merge_and_dedup_edges(edge_index_list, L: int):
    if not edge_index_list:
        return np.zeros((2, 0), dtype=np.int32)
    
    ei = np.concatenate(edge_index_list, axis=1)
    edges_set = set(zip(ei[0], ei[1]))
    if not edges_set:
        return np.zeros((2, 0), dtype=np.int32)
    
    edge_index = np.array(list(edges_set), dtype=np.int32).T
    return edge_index


def build_edge_attr(edge_index: np.ndarray, coords: np.ndarray):
    if edge_index.shape[1] == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int8),
            np.array([], dtype=np.int8)
        )
    
    src = edge_index[0].astype(np.int32)
    dst = edge_index[1].astype(np.int32)
    
    vec = coords[src] - coords[dst]
    dist = np.sqrt(np.sum(vec * vec, axis=1)).astype(np.float32)
    dist = np.maximum(dist, DIST_EPS)
    
    inv_dist = (1.0 / dist).astype(np.float32)
    
    dseq = np.abs(src - dst).astype(np.int32)
    seqbin = np.array([seq_bin_id(int(d)) for d in dseq], dtype=np.int8)
    edge_is_seq = (dseq == 1).astype(np.int8)
    
    return dist, inv_dist, seqbin, edge_is_seq


def process_one(args):
    pdb_path, out_dir, knn_k = args
    base = os.path.basename(pdb_path)
    protein_id = normalize_protein_id(base)
    out_npz = os.path.join(out_dir, f"{protein_id}.edge_features.train.npz")
    
    try:
        out = get_single_chain_ca_coords(pdb_path)
        if out is None:
            logging.warning(f"Skipped {base}: short or invalid chain")
            return {
                "name": protein_id,
                "protein_id": protein_id,
                "pdb_file": base,
                "chain_id": "N/A",
                "L": 0,
                "E": 0,
                "npz_path": "skipped",
            }
        
        coords, res_index, chain_id = out
        L = coords.shape[0]
        
        ei_seq = seq_edges(L)
        ei_knn = knn_edges(coords, knn_k)
        edge_index = merge_and_dedup_edges([ei_seq, ei_knn], L=L)
        
        edge_dist, edge_inv_dist, edge_seqbin, edge_is_seq = build_edge_attr(edge_index, coords)
        
        np.savez_compressed(
            out_npz,
            L=np.int32(L),
            res_index=res_index.astype(np.int32),
            edge_index=edge_index.astype(np.int32),
            edge_dist=edge_dist.astype(np.float32),
            edge_inv_dist=edge_inv_dist.astype(np.float32),
            edge_seqbin=edge_seqbin.astype(np.int8),
            edge_is_seq=edge_is_seq.astype(np.int8),
        )
        
        return {
            "name": protein_id,
            "protein_id": protein_id,
            "pdb_file": base,
            "chain_id": chain_id,
            "L": int(L),
            "E": int(edge_index.shape[1]),
            "npz_path": out_npz,
        }
    
    except Exception as e:
        logging.warning(f"Failed {base}: {e}")
        print(f"[WARN] failed {base}: {e}")
        return None


def main(pdb_dir: str, out_dir: str, index_csv: str, knn_k: int = KNN_K, nproc: int = None):
    os.makedirs(out_dir, exist_ok=True)
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in {pdb_dir}")
    
    if nproc is None:
        nproc = max(1, cpu_count() - 1)
    
    print(f"[INFO] PDBs: {len(pdb_files)}")
    print(f"[INFO] KNN backend: {_KNN_BACKEND}")
    print(f"[INFO] nproc={nproc}, knn_k={knn_k}")
    print(f"[INFO] SEQ_BINS={SEQ_BINS}")
    
    tasks = [(p, out_dir, knn_k) for p in pdb_files]
    
    rows = []
    
    with Pool(processes=nproc) as pool:
        for rec in pool.imap_unordered(process_one, tasks, chunksize=50):
            if rec is not None:
                rows.append(rec)
    
    if not rows:
        raise RuntimeError("No valid proteins processed.")
    
    df = pd.DataFrame(rows).sort_values("pdb_file").reset_index(drop=True)
    df.to_csv(index_csv, index=False)
    print(f"[OK] index saved: {index_csv} (n={len(df)})")
    print(f"[OK] per-protein edge npz saved in: {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build per-protein edge features for graph encoder.")
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--knn_k", type=int, default=KNN_K)
    ap.add_argument("--nproc", type=int, default=None)
    args = ap.parse_args()
    
    main(
        pdb_dir=args.pdb_dir,
        out_dir=args.out_dir,
        index_csv=args.index_csv,
        knn_k=args.knn_k,
        nproc=args.nproc
    )