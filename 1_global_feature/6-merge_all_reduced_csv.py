import os
import glob
import pandas as pd


def normalize_protein_id(x: str) -> str:
    """
    Normalize protein ID in a robust, pipeline-wide consistent way.

    Handles:
      - xxx.ef.pdb -> xxx
      - xxx.pdb    -> xxx
      - xxx.ef     -> xxx   (compatibility for old intermediate CSVs)
      - xxx        -> xxx
    """
    x = str(x).strip()
    if x.endswith(".ef.pdb"):
        x = x[:-7]
    elif x.endswith(".pdb"):
        x = x[:-4]
    elif x.endswith(".ef"):
        x = x[:-3]
    return x


def find_reduced_csvs_by_split(root_dir: str):
    """
    Find reduced CSVs for each split:
      - *train_reduced.csv
      - *valid_reduced.csv
      - *test_reduced.csv
    """
    split_patterns = {
        "train": os.path.join(root_dir, "**", "*train_reduced.csv"),
        "valid": os.path.join(root_dir, "**", "*valid_reduced.csv"),
        "test": os.path.join(root_dir, "**", "*test_reduced.csv"),
    }

    split_paths = {}
    for split, pattern in split_patterns.items():
        split_paths[split] = sorted(glob.glob(pattern, recursive=True))

    return split_paths


def normalize_id_col(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    if df.shape[1] < 1:
        raise ValueError(f"Empty CSV (no columns): {csv_path}")

    # First column is protein ID by your convention
    first_col = df.columns[0]
    if first_col != "PDB_ID":
        df = df.rename(columns={first_col: "PDB_ID"})

    # Ensure consistent protein ID normalization
    df["PDB_ID"] = df["PDB_ID"].astype(str).map(normalize_protein_id)
    return df


def safe_suffix_from_path(csv_path: str) -> str:
    """
    Create a short suffix from file name + parent folder to disambiguate duplicate columns.
    """
    base = os.path.splitext(os.path.basename(csv_path))[0]
    parent = os.path.basename(os.path.dirname(csv_path))
    suf = f"{parent}__{base}"
    suf = suf.replace(" ", "_").replace("-", "_")
    return suf


def merge_one_split_csvs(paths, out_csv: str, split_name: str):
    if not paths:
        print(f"[WARN] No '*{split_name}_reduced.csv' found. Skip.")
        return

    merged = None
    used_cols = {"PDB_ID"}

    for p in paths:
        df = pd.read_csv(p)
        df = normalize_id_col(df, p)

        # Drop invalid / empty IDs
        df = df[df["PDB_ID"].notna() & (df["PDB_ID"].astype(str).str.len() > 0)].copy()

        # Optional safety: remove duplicated PDB_ID rows inside a single CSV
        # Keep first occurrence to avoid merge explosion
        if df["PDB_ID"].duplicated().any():
            dup_n = int(df["PDB_ID"].duplicated().sum())
            print(f"[WARN][{split_name}] duplicated PDB_ID in {p}: {dup_n} duplicated rows, keeping first")
            df = df.drop_duplicates(subset=["PDB_ID"], keep="first").copy()

        # If duplicate feature names already exist in merged, suffix them
        suf = safe_suffix_from_path(p)
        rename_map = {}
        for c in df.columns:
            if c == "PDB_ID":
                continue
            if c in used_cols:
                rename_map[c] = f"{c}__{suf}"

        if rename_map:
            df = df.rename(columns=rename_map)

        used_cols.update([c for c in df.columns if c != "PDB_ID"])

        # Merge
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="PDB_ID", how="outer")

        print(f"[OK][{split_name}] merged: {p}  (rows={len(df)}, cols={df.shape[1]})")

    merged = merged.sort_values("PDB_ID").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    merged.to_csv(out_csv, index=False)

    print(f"\n[DONE][{split_name}] saved merged global features: {out_csv}")
    print(f"[INFO][{split_name}] total files merged: {len(paths)}")
    print(f"[INFO][{split_name}] final shape: {merged.shape}\n")


def merge_reduced_csvs_by_split(root_dir: str, out_dir: str):
    split_paths = find_reduced_csvs_by_split(root_dir)

    outputs = {
        "train": os.path.join(out_dir, "global_features_train_merged.csv"),
        "valid": os.path.join(out_dir, "global_features_valid_merged.csv"),
        "test": os.path.join(out_dir, "global_features_test_merged.csv"),
    }

    for split in ["train", "valid", "test"]:
        merge_one_split_csvs(
            paths=split_paths[split],
            out_csv=outputs[split],
            split_name=split
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Merge *train_reduced.csv, *valid_reduced.csv, and *test_reduced.csv separately."
    )
    ap.add_argument(
        "--root_dir",
        default=".",
        help="Root directory to search recursively (default: current dir)"
    )
    ap.add_argument(
        "--out_dir",
        default=".",
        help="Output directory for merged CSVs"
    )
    args = ap.parse_args()

    merge_reduced_csvs_by_split(args.root_dir, args.out_dir)