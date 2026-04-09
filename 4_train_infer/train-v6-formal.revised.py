#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Formal training script for protein solubility classification from fused features.

Academic design
---------------
1. Step 1:
   - benchmark traditional ML models on train/val/test
   - benchmark one raw-feature MLP baseline (MLP1)

2. Step 2:
   - select Top-k traditional ML models under family-constrained rules
   - generate train OOF probabilities for anti-leakage meta-learning
   - compare compact meta learners on probability inputs:
       average / logistic stacking / MLP2
   - optional hybrid meta experiments with raw features + probabilities

3. Final:
   - choose the final meta model by VAL_MCC
   - report test performance only once
   - save a complete inference package:
       base models + final meta model + feature order + manifest + example inference script

Notes
-----
- Validation is used for model selection and threshold selection.
- Test is used only for final reporting and never for model ranking or family selection.
- Default mainline does NOT apply post-hoc calibration, to keep the methodological line clean.
"""

import os
import json
import sys
import shutil
import argparse
import warnings
import random
import platform
import importlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Tuple, Callable, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, auc, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")


# =========================================================
# Global configs
# =========================================================
MODEL_FAMILY_MAP = {
    "LR": "linear",
    "SGDLog": "linear",
    "RandomForest": "rf_et",
    "ExtraTrees": "rf_et",
    "XGBoost": "boosting",
    "LightGBM": "boosting",
    "CatBoost": "boosting",
}

FAMILY_PRIORITY = ["linear", "rf_et", "boosting"]


# =========================================================
# Utility
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


def save_txt_lines(lines: List[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(str(x) + "\n")


def get_package_version(pkg_name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(pkg_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def get_environment_info() -> Dict[str, Any]:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": get_package_version("numpy"),
        "pandas": get_package_version("pandas"),
        "sklearn": get_package_version("sklearn"),
        "matplotlib": get_package_version("matplotlib"),
        "joblib": get_package_version("joblib"),
        "torch": get_package_version("torch"),
        "xgboost": get_package_version("xgboost"),
        "lightgbm": get_package_version("lightgbm"),
        "catboost": get_package_version("catboost"),
    }
    return info


# =========================================================
# Optional booster imports
# =========================================================
def try_import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None


def try_import_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None


def try_import_cat():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except Exception:
        return None


# =========================================================
# Data
# =========================================================
def load_split(path: str, id_col: str = "PDB_ID", y_col: str = "y", feat_prefix: str = "h_fused_"):
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"Missing id column '{id_col}' in {path}")
    if y_col not in df.columns:
        raise ValueError(f"Missing label column '{y_col}' in {path}")

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    if df[id_col].isna().any():
        raise ValueError(f"Found missing values in id column '{id_col}' in {path}")

    dup_mask = df[id_col].duplicated(keep=False)
    if dup_mask.any():
        dup_ids = df.loc[dup_mask, id_col].astype(str).unique().tolist()[:10]
        raise ValueError(
            f"Found duplicated IDs in {path}. First duplicated IDs: {dup_ids}"
        )

    feat_cols = [c for c in df.columns if c.startswith(feat_prefix)]
    if len(feat_cols) == 0:
        raise ValueError(f"No feature columns starting with '{feat_prefix}' in {path}")

    X = df[feat_cols].astype(np.float32).to_numpy()
    y = df[y_col].astype(int).to_numpy()

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Label column '{y_col}' must be binary 0/1. Found: {np.unique(y)}")
    if np.any(~np.isfinite(X)):
        raise ValueError(f"Found NaN/Inf in features of {path}")

    return df, X, y, feat_cols


def check_feature_alignment(train_cols: List[str], other_cols: List[str], split_name: str):
    if train_cols != other_cols:
        missing = [c for c in train_cols if c not in other_cols][:10]
        extra = [c for c in other_cols if c not in train_cols][:10]
        raise ValueError(
            f"Feature columns mismatch for {split_name}. "
            f"Missing(first10)={missing}, Extra(first10)={extra}"
        )


def summarize_split(name: str, df: pd.DataFrame, feat_cols: List[str], y: np.ndarray) -> Dict[str, Any]:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return {
        "split": name,
        "n": int(len(y)),
        "pos": pos,
        "neg": neg,
        "pos_ratio": float(y.mean()),
        "n_features": int(len(feat_cols)),
        "id_unique": int(df["PDB_ID"].nunique()),
    }


# =========================================================
# Metrics
# =========================================================
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def find_best_threshold(y_true: np.ndarray, prob: np.ndarray, metric: str = "mcc") -> Tuple[float, float]:
    y_true = y_true.astype(int)
    prob = np.clip(prob.astype(float), 0.0, 1.0)

    best_t, best_v = 0.5, -1e18
    for t in np.linspace(0.01, 0.99, 99):
        pred = (prob >= t).astype(int)
        if metric == "mcc":
            v = matthews_corrcoef(y_true, pred)
        elif metric == "f1":
            v = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError("metric must be 'mcc' or 'f1'")
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return best_t, best_v


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = y_true.astype(int)
    prob = np.clip(prob.astype(float), 0.0, 1.0)
    pred = (prob >= threshold).astype(int)

    out = {
        "MCC": float(matthews_corrcoef(y_true, pred)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Accuracy": float(np.mean(pred == y_true)),
    }

    try:
        out["ROC_AUC"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["ROC_AUC"] = float("nan")

    try:
        out["PR_AUC"] = float(average_precision_score(y_true, prob))
    except Exception:
        out["PR_AUC"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return out


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Standard ECE:
    compare mean predicted probability vs observed positive fraction in each bin.
    """
    y_true = y_true.astype(int)
    prob = np.clip(prob.astype(float), 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (prob >= lo) & (prob < hi)
        else:
            mask = (prob >= lo) & (prob <= hi)

        n_k = int(mask.sum())
        if n_k == 0:
            continue

        p_k = prob[mask]
        y_k = y_true[mask]

        conf_k = float(np.mean(p_k))
        acc_k = float(np.mean(y_k))
        ece += (n_k / N) * abs(acc_k - conf_k)

    return float(ece)


def calibration_curve_table(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    y_true = y_true.astype(int)
    prob = np.clip(prob.astype(float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (prob >= lo) & (prob < hi)
        else:
            mask = (prob >= lo) & (prob <= hi)

        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bin": i + 1,
                "lo": lo,
                "hi": hi,
                "n": 0,
                "mean_prob": np.nan,
                "empirical_pos": np.nan,
            })
        else:
            rows.append({
                "bin": i + 1,
                "lo": lo,
                "hi": hi,
                "n": n,
                "mean_prob": float(prob[mask].mean()),
                "empirical_pos": float(y_true[mask].mean()),
            })
    return pd.DataFrame(rows)


# =========================================================
# Plot helpers
# =========================================================
def plot_bar_comparison(df: pd.DataFrame, metric: str, out_path: str, title: str, topn: int = 12):
    d = df.sort_values(metric, ascending=False).head(topn).copy()
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(d)), 5.5))

    x = np.arange(len(d))
    ax.bar(x, d[metric].astype(float).values)
    ax.set_xticks(x)
    ax.set_xticklabels(d["Model"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, prob: np.ndarray, prefix: str, out_dir: str, split_name: str):
    fpr, tpr, _ = roc_curve(y_true, prob)
    precision, recall, _ = precision_recall_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, prob)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve ({split_name})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_{split_name}_ROC.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, label=f"AUPRC={pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve ({split_name})")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_{split_name}_PR.pdf"))
    plt.close(fig)


def plot_reliability(y_true: np.ndarray, prob: np.ndarray, out_path: str, title: str, n_bins: int = 15):
    df = calibration_curve_table(y_true, prob, n_bins=n_bins)
    valid = df["n"] > 0

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(df.loc[valid, "mean_prob"], df.loc[valid, "empirical_pos"], marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return df


def plot_prob_correlation_heatmap(prob_df: pd.DataFrame, out_path: str, title: str):
    corr = prob_df.corr(method="pearson")
    arr = corr.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(0.75 * len(corr.columns) + 3, 0.75 * len(corr.columns) + 2.5))
    im = ax.imshow(arr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return corr


# =========================================================
# Save helpers
# =========================================================
def save_prediction_csv(df_split: pd.DataFrame, y: np.ndarray, prob: np.ndarray, thr: float, path: str):
    out = df_split[["PDB_ID"]].copy()
    out["y"] = y.astype(int)
    out["prob"] = prob.astype(float)
    out["pred"] = (prob >= thr).astype(int)
    out.to_csv(path, index=False)


def save_mlp_bundle(
    out_path: str,
    model_scaler,
    cfg: Dict[str, Any],
    feature_dim: int,
    input_feature_names: List[str],
    threshold: float,
    extra_info: Optional[Dict[str, Any]] = None,
):
    model, scaler = model_scaler
    bundle = {
        "state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "mlp_config": cfg,
        "feature_dim": int(feature_dim),
        "input_feature_names": input_feature_names,
        "threshold": float(threshold),
        "extra_info": extra_info or {},
    }
    torch.save(bundle, out_path)


# =========================================================
# MLP
# =========================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


@dataclass
class MLPConfig:
    hidden: List[int]
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int = 280
    patience: int = 30


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: MLPConfig,
    device: str,
    pos_weight: float,
    seed: int = 42,
):
    seed_everything(seed)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(y_train.astype(np.float32))
    Xva_t = torch.from_numpy(Xva)
    yva_np = y_val.astype(int)

    model = MLP(in_dim=Xtr.shape[1], hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    n = Xtr_t.shape[0]
    idx = np.arange(n)

    best_state = None
    best_mcc = -1e18
    best_thr = 0.5
    best_epoch = 0
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        np.random.shuffle(idx)

        for s in range(0, n, cfg.batch_size):
            bi = idx[s:s + cfg.batch_size]
            xb = Xtr_t[bi].to(device)
            yb = ytr_t[bi].to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits_va = model(Xva_t.to(device)).detach().cpu().numpy()
            prob_va = _sigmoid(logits_va)

        thr, mcc = find_best_threshold(yva_np, prob_va, metric="mcc")

        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = thr
            best_epoch = epoch
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "scaler_mean": scaler.mean_.copy(),
                "scaler_scale": scaler.scale_.copy(),
            }
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("No best model found during MLP training.")

    model.load_state_dict(best_state["model_state"])
    scaler.mean_ = best_state["scaler_mean"]
    scaler.scale_ = best_state["scaler_scale"]
    model.eval()

    with torch.no_grad():
        logits_va = model(torch.from_numpy(scaler.transform(X_val).astype(np.float32)).to(device)).cpu().numpy()
    prob_val = _sigmoid(logits_va)

    info = {
        "hidden": cfg.hidden,
        "dropout": float(cfg.dropout),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_size),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_thr),
        "best_val_mcc": float(best_mcc),
    }
    return info, prob_val, (model, scaler)


def mlp_predict(model_scaler, X: np.ndarray, device: str) -> np.ndarray:
    model, scaler = model_scaler
    Xs = scaler.transform(X).astype(np.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xs).to(device)).cpu().numpy()
    return _sigmoid(logits)


def parse_hidden_list(hidden_str_list: List[str]) -> List[List[int]]:
    out = []
    for s in hidden_str_list:
        out.append([int(x) for x in s.split(",") if x.strip()])
    return out


# =========================================================
# Model factories
# =========================================================
def build_model_factories(seed: int, pos_ratio: float) -> Tuple[Dict[str, Callable[[], Any]], Dict[str, str]]:
    class_weight = "balanced"
    spw = (1.0 - pos_ratio) / max(pos_ratio, 1e-9)

    factories: Dict[str, Callable[[], Any]] = {}
    availability: Dict[str, str] = {}

    factories["LR"] = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=seed
        )),
    ])
    availability["LR"] = "enabled"

    factories["SGDLog"] = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=5000,
            class_weight=class_weight,
            random_state=seed
        )),
    ])
    availability["SGDLog"] = "enabled"

    factories["RandomForest"] = lambda: RandomForestClassifier(
        n_estimators=800,
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weight,
        max_features="sqrt"
    )
    availability["RandomForest"] = "enabled"

    factories["ExtraTrees"] = lambda: ExtraTreesClassifier(
        n_estimators=1200,
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weight,
        max_features="sqrt"
    )
    availability["ExtraTrees"] = "enabled"

    xgb = try_import_xgb()
    if xgb is not None:
        factories["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=20000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            scale_pos_weight=spw,
        )
        availability["XGBoost"] = "enabled"
    else:
        availability["XGBoost"] = "missing"

    lgb = try_import_lgb()
    if lgb is not None:
        factories["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=20000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary",
            n_jobs=-1,
            random_state=seed,
            scale_pos_weight=spw,
        )
        availability["LightGBM"] = "enabled"
    else:
        availability["LightGBM"] = "missing"

    CatBoostClassifier = try_import_cat()
    if CatBoostClassifier is not None:
        factories["CatBoost"] = lambda: CatBoostClassifier(
            iterations=20000,
            learning_rate=0.03,
            depth=8,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=seed,
            verbose=False,
            class_weights=[1.0, float(spw)],
        )
        availability["CatBoost"] = "enabled"
    else:
        availability["CatBoost"] = "missing"

    return factories, availability


def model_family(model_name: str) -> str:
    return MODEL_FAMILY_MAP.get(model_name, "other")


def predict_proba_any(model, X) -> np.ndarray:
    """
    Only use models that can produce meaningful probabilities.
    """
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        return np.clip(prob.astype(float), 0.0, 1.0)

    if hasattr(model, "decision_function"):
        # kept only as a fallback; current benchmark models should all have predict_proba
        return np.clip(_sigmoid(model.decision_function(X)), 0.0, 1.0)

    raise ValueError("Model does not support predict_proba or decision_function.")


# =========================================================
# Fitting helpers
# =========================================================
def fit_with_early_stopping_on_val(model, Xtr, ytr, Xva, yva, name: str):
    """
    Use provided validation set for early stopping.
    Appropriate for Step-1 benchmark where val is the designated model-selection split.
    """
    lname = name.lower()

    if "xgboost" in lname:
        try:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, early_stopping_rounds=300)
        except TypeError:
            try:
                import xgboost as xgb
                cb = xgb.callback.EarlyStopping(rounds=300, save_best=True)
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, callbacks=[cb])
            except Exception:
                model.fit(Xtr, ytr)
        return

    if "lightgbm" in lname:
        import lightgbm as lgb
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
        )
        return

    if "catboost" in lname:
        model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        return

    model.fit(Xtr, ytr)


def fit_with_internal_early_stopping(model, Xtr, ytr, name: str, seed: int):
    """
    For OOF generation:
    use only the training fold and split an inner early-stopping subset from it,
    so the outer OOF validation fold is not used for early stopping.
    """
    lname = name.lower()

    if ("xgboost" not in lname) and ("lightgbm" not in lname) and ("catboost" not in lname):
        model.fit(Xtr, ytr)
        return

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    sub_tr_idx, sub_es_idx = next(splitter.split(Xtr, ytr))
    X_sub_tr, y_sub_tr = Xtr[sub_tr_idx], ytr[sub_tr_idx]
    X_sub_es, y_sub_es = Xtr[sub_es_idx], ytr[sub_es_idx]

    if "xgboost" in lname:
        try:
            model.fit(X_sub_tr, y_sub_tr, eval_set=[(X_sub_es, y_sub_es)], verbose=False, early_stopping_rounds=300)
        except TypeError:
            try:
                import xgboost as xgb
                cb = xgb.callback.EarlyStopping(rounds=300, save_best=True)
                model.fit(X_sub_tr, y_sub_tr, eval_set=[(X_sub_es, y_sub_es)], verbose=False, callbacks=[cb])
            except Exception:
                model.fit(X_sub_tr, y_sub_tr)
        return

    if "lightgbm" in lname:
        import lightgbm as lgb
        model.fit(
            X_sub_tr, y_sub_tr,
            eval_set=[(X_sub_es, y_sub_es)],
            eval_metric="logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
        )
        return

    if "catboost" in lname:
        model.fit(X_sub_tr, y_sub_tr, eval_set=(X_sub_es, y_sub_es), use_best_model=True)
        return


def make_oof_probs(
    factories: Dict[str, Callable[[], Any]],
    model_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(model_names)), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva = X[va_idx]

        for j, name in enumerate(model_names):
            m = factories[name]()
            fit_with_internal_early_stopping(m, Xtr, ytr, name=name, seed=seed + fold)
            oof[va_idx, j] = predict_proba_any(m, Xva).astype(np.float32)

        print(f"[OOF] fold {fold}/{n_splits} done")

    return oof


# =========================================================
# Family-constrained model selection
# =========================================================
def rank_models_for_selection(df_ml: pd.DataFrame) -> pd.DataFrame:
    out = df_ml.copy()
    out["Family"] = out["Model"].map(model_family)
    sort_cols = ["Family", "VAL_MCC", "VAL_PR_AUC", "VAL_ROC_AUC"]
    ascending = [True, False, False, False]
    out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    out["FamilyRank"] = out.groupby("Family").cumcount() + 1
    return out


def select_models_by_family(
    df_ml: pd.DataFrame,
    val_prob_df: pd.DataFrame,
    total_topk: int = 4,
    boosting_corr_threshold: float = 0.985,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if total_topk < 3:
        raise ValueError("--topk must be >= 3 under the current family-constrained selection strategy.")

    ranked = rank_models_for_selection(df_ml)
    within_family = ranked.copy()

    selected: List[str] = []
    logic: Dict[str, Any] = {
        "selection_strategy": "family_constrained",
        "ranking_priority": ["VAL_MCC", "VAL_PR_AUC", "VAL_ROC_AUC"],
        "family_rules": {
            "linear": "Select the best one by family ranking",
            "rf_et": "Select the best one by family ranking",
            "boosting": "Select one mandatory model; select a second only if allowed by topk and validation-probability correlation is below threshold",
        },
        "requested_total_topk": int(total_topk),
        "boosting_corr_threshold": float(boosting_corr_threshold),
        "decisions": [],
        "families_available": {fam: ranked.loc[ranked["Family"] == fam, "Model"].tolist() for fam in FAMILY_PRIORITY},
    }

    for fam in ["linear", "rf_et"]:
        fam_df = ranked[ranked["Family"] == fam].reset_index(drop=True)
        if fam_df.empty:
            logic["decisions"].append({
                "family": fam,
                "status": "missing",
                "reason": f"No available model in family '{fam}'"
            })
            continue

        row = fam_df.iloc[0]
        selected.append(row["Model"])
        logic["decisions"].append({
            "family": fam,
            "status": "selected",
            "selected_model": row["Model"],
            "reason": "Highest family rank by VAL_MCC -> VAL_PR_AUC -> VAL_ROC_AUC",
            "metrics": {
                "VAL_MCC": float(row["VAL_MCC"]),
                "VAL_PR_AUC": float(row["VAL_PR_AUC"]),
                "VAL_ROC_AUC": float(row["VAL_ROC_AUC"]),
            },
        })

    boosting_df = ranked[ranked["Family"] == "boosting"].reset_index(drop=True)
    max_boosting_to_keep = max(1, min(2, int(total_topk) - len(selected))) if len(boosting_df) > 0 else 0

    if boosting_df.empty:
        logic["decisions"].append({
            "family": "boosting",
            "status": "missing",
            "reason": "No available boosting model"
        })
    else:
        first = boosting_df.iloc[0]
        selected.append(first["Model"])
        logic["decisions"].append({
            "family": "boosting",
            "slot": 1,
            "status": "selected",
            "selected_model": first["Model"],
            "reason": "Mandatory best boosting model by family ranking",
            "metrics": {
                "VAL_MCC": float(first["VAL_MCC"]),
                "VAL_PR_AUC": float(first["VAL_PR_AUC"]),
                "VAL_ROC_AUC": float(first["VAL_ROC_AUC"]),
            },
        })

        if max_boosting_to_keep >= 2:
            second_selected = None
            for _, row in boosting_df.iloc[1:].iterrows():
                corr = float(val_prob_df[[first["Model"], row["Model"]]].corr().iloc[0, 1])
                if np.isnan(corr):
                    corr = 1.0
                if abs(corr) <= boosting_corr_threshold:
                    second_selected = (row, corr)
                    break

            if second_selected is not None:
                row, corr = second_selected
                selected.append(row["Model"])
                logic["decisions"].append({
                    "family": "boosting",
                    "slot": 2,
                    "status": "selected",
                    "selected_model": row["Model"],
                    "reason": "Best remaining boosting model passing correlation constraint",
                    "corr_with_first_boosting": float(corr),
                    "metrics": {
                        "VAL_MCC": float(row["VAL_MCC"]),
                        "VAL_PR_AUC": float(row["VAL_PR_AUC"]),
                        "VAL_ROC_AUC": float(row["VAL_ROC_AUC"]),
                    },
                })
            else:
                fallback = []
                for _, row in boosting_df.iloc[1:].iterrows():
                    corr = float(val_prob_df[[first["Model"], row["Model"]]].corr().iloc[0, 1])
                    if np.isnan(corr):
                        corr = 1.0
                    fallback.append({
                        "candidate": row["Model"],
                        "corr_with_first_boosting": float(corr),
                        "VAL_MCC": float(row["VAL_MCC"]),
                    })
                logic["decisions"].append({
                    "family": "boosting",
                    "slot": 2,
                    "status": "skipped",
                    "reason": "No second boosting model satisfied correlation threshold",
                    "evaluated_candidates": fallback,
                })

    selected = list(dict.fromkeys(selected))
    selected_family_table = within_family[within_family["Model"].isin(selected)].copy()
    selected_family_table["SelectionOrder"] = selected_family_table["Model"].map({m: i + 1 for i, m in enumerate(selected)})
    selected_family_table = selected_family_table.sort_values("SelectionOrder").reset_index(drop=True)

    logic["selected_models"] = selected
    return selected, within_family, selected_family_table, logic


# =========================================================
# Meta learning
# =========================================================
def build_meta_inputs(
    add_raw_features: bool,
    Xtr: np.ndarray,
    Xva: np.ndarray,
    Xte: np.ndarray,
    oof_train: np.ndarray,
    P_val: np.ndarray,
    P_test: np.ndarray,
    top_names: List[str],
    feat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if add_raw_features:
        feature_names = feat_cols + [f"OOF_{n}" for n in top_names]
        return (
            np.concatenate([Xtr, oof_train], axis=1).astype(np.float32),
            np.concatenate([Xva, P_val], axis=1).astype(np.float32),
            np.concatenate([Xte, P_test], axis=1).astype(np.float32),
            feature_names,
        )

    feature_names = [f"OOF_{n}" for n in top_names]
    return (
        oof_train.astype(np.float32),
        P_val.astype(np.float32),
        P_test.astype(np.float32),
        feature_names,
    )


def train_stacker(Xtr: np.ndarray, ytr: np.ndarray, seed: int):
    stacker = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=seed
        )),
    ])
    stacker.fit(Xtr, ytr)
    return stacker


def run_meta_experiments(
    args,
    run_dirs: Dict[str, str],
    feat_cols: List[str],
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    top_names: List[str],
    oof_train: np.ndarray,
    P_val: np.ndarray,
    P_test: np.ndarray,
    pos_weight: float,
    device: str,
):
    compare_dir = run_dirs["meta_compare"]
    model_dir = run_dirs["meta_models"]
    pred_dir = run_dirs["predictions"]

    experiments = [
        {
            "name": "avg_probs",
            "meta_learner": "average",
            "add_raw_features": False,
            "description": "Simple average of Top-k ML probabilities"
        },
        {
            "name": "stacking_probs",
            "meta_learner": "stacking",
            "add_raw_features": False,
            "description": "Logistic stacking using only Top-k ML probabilities"
        },
        {
            "name": "mlp2_probs",
            "meta_learner": "mlp2",
            "add_raw_features": False,
            "description": "MLP2 using only Top-k ML probabilities"
        },
    ]

    if args.run_extended_compare:
        experiments += [
            {
                "name": "stacking_hybrid",
                "meta_learner": "stacking",
                "add_raw_features": True,
                "description": "Hybrid logistic stacking using raw fused features + Top-k probabilities"
            },
            {
                "name": "mlp2_hybrid",
                "meta_learner": "mlp2",
                "add_raw_features": True,
                "description": "Hybrid MLP2 using raw fused features + Top-k probabilities"
            },
        ]

    results = []
    best_artifact = None
    best_score = -1e18

    mlp2_hidden = parse_hidden_list(args.mlp2_hidden)

    for exp in experiments:
        name = exp["name"]
        exp_dir = os.path.join(model_dir, name)
        ensure_dir(exp_dir)

        Xtr_input, Xva_input, Xte_input, input_feature_names = build_meta_inputs(
            exp["add_raw_features"], Xtr, Xva, Xte, oof_train, P_val, P_test, top_names, feat_cols
        )

        print(f"\n[Meta] Running {name} | input_dim={Xtr_input.shape[1]}")

        if exp["meta_learner"] == "average":
            prob_val = P_val.mean(axis=1)
            prob_test = P_test.mean(axis=1)
            thr, _ = find_best_threshold(yva, prob_val, metric="mcc")

            save_json({
                "type": "simple_average",
                "top_models": top_names,
                "threshold": float(thr),
                "input_feature_names": input_feature_names,
            }, os.path.join(exp_dir, "model_info.json"))

            artifact = {
                "type": "average",
                "name": name,
                "model_path": os.path.join(exp_dir, "model_info.json"),
                "threshold": float(thr),
                "input_feature_names": input_feature_names,
                "add_raw_features": bool(exp["add_raw_features"]),
            }

        elif exp["meta_learner"] == "stacking":
            stacker = train_stacker(Xtr_input, ytr, seed=args.seed)
            prob_val = stacker.predict_proba(Xva_input)[:, 1]
            prob_test = stacker.predict_proba(Xte_input)[:, 1]
            thr, _ = find_best_threshold(yva, prob_val, metric="mcc")

            joblib.dump(stacker, os.path.join(exp_dir, "stacker.joblib"))
            save_json({
                "type": "stacking",
                "threshold": float(thr),
                "input_feature_names": input_feature_names,
            }, os.path.join(exp_dir, "model_info.json"))

            artifact = {
                "type": "stacking",
                "name": name,
                "model_path": os.path.join(exp_dir, "stacker.joblib"),
                "threshold": float(thr),
                "input_feature_names": input_feature_names,
                "add_raw_features": bool(exp["add_raw_features"]),
            }

        else:
            grid_rows = []
            best_local = None
            best_local_score = -1e18
            best_local_prob_val = None
            best_local_model = None

            for hidden in mlp2_hidden:
                for dp in args.mlp2_dropout:
                    for lr_ in args.mlp2_lr:
                        for wd in args.mlp2_wd:
                            for bs in args.mlp2_batch:
                                cfg = MLPConfig(
                                    hidden=hidden,
                                    dropout=float(dp),
                                    lr=float(lr_),
                                    weight_decay=float(wd),
                                    batch_size=int(bs),
                                    max_epochs=int(args.mlp2_max_epochs),
                                    patience=int(args.mlp2_patience),
                                )
                                info, prob_val_tmp, model_scaler = train_mlp(
                                    Xtr_input, ytr, Xva_input, yva, cfg,
                                    device=device, pos_weight=pos_weight, seed=args.seed
                                )
                                thr_tmp = float(info["best_threshold"])
                                m_val_tmp = compute_metrics(yva, prob_val_tmp, threshold=thr_tmp)

                                grid_row = {
                                    "experiment": name,
                                    **info,
                                    **{f"VAL_{k}": v for k, v in m_val_tmp.items()}
                                }
                                grid_rows.append(grid_row)

                                if m_val_tmp["MCC"] > best_local_score:
                                    best_local_score = m_val_tmp["MCC"]
                                    best_local = info
                                    best_local_prob_val = prob_val_tmp.copy()
                                    best_local_model = model_scaler

            df_grid = pd.DataFrame(grid_rows).sort_values("VAL_MCC", ascending=False)
            df_grid.to_csv(os.path.join(exp_dir, "mlp2_grid_results.csv"), index=False)

            prob_val = best_local_prob_val
            prob_test = mlp_predict(best_local_model, Xte_input, device=device)
            thr = float(best_local["best_threshold"])

            save_mlp_bundle(
                os.path.join(exp_dir, "mlp2_model.pt"),
                best_local_model,
                cfg={
                    "hidden": best_local["hidden"],
                    "dropout": best_local["dropout"],
                    "lr": best_local["lr"],
                    "weight_decay": best_local["weight_decay"],
                    "batch_size": best_local["batch_size"],
                    "max_epochs": int(args.mlp2_max_epochs),
                    "patience": int(args.mlp2_patience),
                },
                feature_dim=Xtr_input.shape[1],
                input_feature_names=input_feature_names,
                threshold=thr,
                extra_info={"meta_learner": "mlp2", "experiment": name},
            )

            save_json({
                "type": "mlp2",
                "threshold": float(thr),
                "best_info": best_local,
                "input_feature_names": input_feature_names,
            }, os.path.join(exp_dir, "model_info.json"))

            artifact = {
                "type": "mlp2",
                "name": name,
                "model_path": os.path.join(exp_dir, "mlp2_model.pt"),
                "threshold": float(thr),
                "input_feature_names": input_feature_names,
                "add_raw_features": bool(exp["add_raw_features"]),
            }

        m_val = compute_metrics(yva, prob_val, threshold=thr)
        m_te = compute_metrics(yte, prob_test, threshold=thr)
        ece_val = expected_calibration_error(yva, prob_val, n_bins=args.ece_bins)
        ece_test = expected_calibration_error(yte, prob_test, n_bins=args.ece_bins)

        save_prediction_csv(df_va, yva, prob_val, thr, os.path.join(pred_dir, f"val_{name}.csv"))
        save_prediction_csv(df_te, yte, prob_test, thr, os.path.join(pred_dir, f"test_{name}.csv"))
        np.save(os.path.join(pred_dir, f"val_{name}.npy"), prob_val)
        np.save(os.path.join(pred_dir, f"test_{name}.npy"), prob_test)

        save_json({
            "name": name,
            "description": exp["description"],
            "meta_learner": exp["meta_learner"],
            "add_raw_features": exp["add_raw_features"],
            "threshold": float(thr),
            "val_metrics": m_val,
            "test_metrics": m_te,
            "ece_val": float(ece_val),
            "ece_test": float(ece_test),
            "top_models": top_names,
            "input_feature_names": input_feature_names,
        }, os.path.join(exp_dir, "evaluation.json"))

        row = {
            "Model": name,
            "Description": exp["description"],
            "MetaLearner": exp["meta_learner"],
            "AddRawFeatures": bool(exp["add_raw_features"]),
            "Threshold": float(thr),
            "ECE_VAL": float(ece_val),
            "ECE_TEST": float(ece_test),
        }
        row.update({f"VAL_{k}": v for k, v in m_val.items()})
        row.update({f"TEST_{k}": v for k, v in m_te.items()})
        results.append(row)

        artifact["prob_val"] = prob_val
        artifact["prob_test"] = prob_test
        artifact["threshold"] = float(thr)
        artifact["row"] = row
        artifact["top_models"] = top_names
        artifact["input_dim"] = int(Xtr_input.shape[1])

        if row["VAL_MCC"] > best_score:
            best_score = row["VAL_MCC"]
            best_artifact = artifact

    df_compare = pd.DataFrame(results).sort_values(["VAL_MCC", "TEST_MCC"], ascending=False)
    df_compare.to_csv(os.path.join(compare_dir, "results_meta_compare.csv"), index=False)
    return df_compare, best_artifact


# =========================================================
# Final inference package
# =========================================================
def save_inference_example_script(final_dir: str):
    content = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_manifest(pkg_dir):
    with open(os.path.join(pkg_dir, "final_model_manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_base_models(pkg_dir, top_models):
    base_models = {}
    for name in top_models:
        path = os.path.join(pkg_dir, "base_models", f"{name}.joblib")
        base_models[name] = joblib.load(path)
    return base_models


def predict_proba_any(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return sigmoid(model.decision_function(X))
    raise ValueError("Model does not support probability prediction.")


def load_mlp_bundle(path, device="cpu"):
    bundle = torch.load(path, map_location=device)
    cfg = bundle["mlp_config"]
    feature_dim = bundle["feature_dim"]
    model = MLP(in_dim=feature_dim, hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    class DummyScaler:
        pass

    scaler = DummyScaler()
    scaler.mean_ = bundle["scaler_mean"]
    scaler.scale_ = bundle["scaler_scale"]

    return model, scaler, bundle


def mlp_predict(model_scaler, X, device="cpu"):
    model, scaler = model_scaler
    Xs = ((X - scaler.mean_) / scaler.scale_).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(Xs).to(device)).cpu().numpy()
    return sigmoid(logits)


def predict_new_samples(feature_csv, pkg_dir, out_csv, feat_prefix="h_fused_", device="cpu"):
    manifest = load_manifest(pkg_dir)

    df = pd.read_csv(feature_csv)
    feat_cols = manifest["base_feature_names"]
    X = df[feat_cols].astype(np.float32).to_numpy()

    top_models = manifest["top_models"]
    base_models = load_base_models(pkg_dir, top_models)

    base_prob_list = []
    for name in top_models:
        prob = predict_proba_any(base_models[name], X)
        base_prob_list.append(prob)

    P = np.stack(base_prob_list, axis=1).astype(np.float32)

    if manifest["type"] == "average":
        final_prob = P.mean(axis=1)

    elif manifest["type"] == "stacking":
        stacker = joblib.load(os.path.join(pkg_dir, manifest["final_model_filename"]))
        if manifest["add_raw_features"]:
            X_meta = np.concatenate([X, P], axis=1).astype(np.float32)
        else:
            X_meta = P.astype(np.float32)
        final_prob = stacker.predict_proba(X_meta)[:, 1]

    elif manifest["type"] == "mlp2":
        model, scaler, bundle = load_mlp_bundle(
            os.path.join(pkg_dir, manifest["final_model_filename"]),
            device=device
        )
        if manifest["add_raw_features"]:
            X_meta = np.concatenate([X, P], axis=1).astype(np.float32)
        else:
            X_meta = P.astype(np.float32)
        final_prob = mlp_predict((model, scaler), X_meta, device=device)

    else:
        raise ValueError(f"Unknown final model type: {manifest['type']}")

    threshold = float(manifest["threshold"])
    pred = (final_prob >= threshold).astype(int)

    out = df.copy()
    out["final_prob"] = final_prob
    out["final_pred"] = pred

    for i, name in enumerate(top_models):
        out[f"base_prob_{name}"] = P[:, i]

    out.to_csv(out_csv, index=False)
    print(f"[OK] Saved predictions to: {out_csv}")


if __name__ == "__main__":
    # Example:
    # python inference_example.py --feature_csv new_samples.csv --pkg_dir ./final_model --out_csv preds.csv
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--pkg_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--feat_prefix", default="h_fused_")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    predict_new_samples(
        feature_csv=args.feature_csv,
        pkg_dir=args.pkg_dir,
        out_csv=args.out_csv,
        feat_prefix=args.feat_prefix,
        device=args.device
    )
'''
    with open(os.path.join(final_dir, "inference_example.py"), "w", encoding="utf-8") as f:
        f.write(content)


def save_final_inference_package(
    run_dirs: Dict[str, str],
    feat_cols: List[str],
    top_names: List[str],
    best_artifact: Dict[str, Any],
):
    final_dir = run_dirs["final_model"]
    ensure_dir(final_dir)
    ensure_dir(os.path.join(final_dir, "base_models"))

    final_model_filename = os.path.basename(best_artifact["model_path"])

    manifest = {
        "selected_model": best_artifact["name"],
        "type": best_artifact["type"],
        "threshold": float(best_artifact["threshold"]),
        "top_models": top_names,
        "base_feature_names": feat_cols,
        "meta_input_feature_names": best_artifact["input_feature_names"],
        "add_raw_features": bool(best_artifact["add_raw_features"]),
        "topk": int(len(top_names)),
        "final_model_filename": final_model_filename,
        "notes": (
            "This package contains all files needed for downstream inference on new samples: "
            "selected base models, final meta model, feature order, and example inference script."
        ),
    }

    save_json(manifest, os.path.join(final_dir, "final_model_manifest.json"))
    save_txt_lines(feat_cols, os.path.join(final_dir, "base_feature_names.txt"))
    save_txt_lines(best_artifact["input_feature_names"], os.path.join(final_dir, "meta_input_feature_names.txt"))

    # copy final model
    src_model = best_artifact["model_path"]
    if os.path.exists(src_model):
        shutil.copy2(src_model, os.path.join(final_dir, os.path.basename(src_model)))
    else:
        raise FileNotFoundError(f"Final model file not found: {src_model}")

    # copy selected base models
    for name in top_names:
        src = os.path.join(run_dirs["single_models"], f"{name}.joblib")
        dst = os.path.join(final_dir, "base_models", f"{name}.joblib")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Selected base model not found: {src}")
        shutil.copy2(src, dst)

    save_inference_example_script(final_dir)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feat_prefix", default="h_fused_")
    ap.add_argument("--ece_bins", type=int, default=15)

    ap.add_argument(
        "--topk", type=int, default=4,
        help="Total number of traditional ML base models to keep under family-constrained selection. Recommended: 4."
    )
    ap.add_argument(
        "--boosting_corr_threshold", type=float, default=0.985,
        help="Maximum allowed absolute validation-probability correlation between the first and optional second boosting model."
    )
    ap.add_argument("--oof_folds", type=int, default=5)

    # MLP1: raw fused features
    ap.add_argument("--mlp1_hidden", nargs="+", default=["512", "512,256", "1024,256"])
    ap.add_argument("--mlp1_dropout", nargs="+", type=float, default=[0.1, 0.3])
    ap.add_argument("--mlp1_lr", nargs="+", type=float, default=[1e-3, 3e-4])
    ap.add_argument("--mlp1_wd", nargs="+", type=float, default=[1e-4])
    ap.add_argument("--mlp1_batch", nargs="+", type=int, default=[256])
    ap.add_argument("--mlp1_max_epochs", type=int, default=280)
    ap.add_argument("--mlp1_patience", type=int, default=30)

    # MLP2: meta features
    ap.add_argument("--mlp2_hidden", nargs="+", default=["64", "128", "128,64"])
    ap.add_argument("--mlp2_dropout", nargs="+", type=float, default=[0.1, 0.3])
    ap.add_argument("--mlp2_lr", nargs="+", type=float, default=[1e-3, 3e-4])
    ap.add_argument("--mlp2_wd", nargs="+", type=float, default=[1e-4])
    ap.add_argument("--mlp2_batch", nargs="+", type=int, default=[128, 256])
    ap.add_argument("--mlp2_max_epochs", type=int, default=220)
    ap.add_argument("--mlp2_patience", type=int, default=25)

    ap.add_argument(
        "--run_extended_compare",
        action="store_true",
        help="Also compare hybrid meta learners that concatenate raw fused features with Top-k probabilities."
    )

    args = ap.parse_args()

    if args.topk < 3:
        raise ValueError("--topk must be >= 3 in the current family-constrained design.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.out_dir, f"run_{timestamp}")

    run_dirs = {
        "root": run_root,
        "tables": os.path.join(run_root, "tables"),
        "figures": os.path.join(run_root, "figures"),
        "predictions": os.path.join(run_root, "predictions"),
        "single_models": os.path.join(run_root, "single_models"),
        "meta_models": os.path.join(run_root, "meta_models"),
        "meta_compare": os.path.join(run_root, "meta_compare"),
        "artifacts": os.path.join(run_root, "artifacts"),
        "final_model": os.path.join(run_root, "final_model"),
    }
    for p in run_dirs.values():
        ensure_dir(p)

    # device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    seed_everything(args.seed)

    # save environment info
    save_json(get_environment_info(), os.path.join(run_root, "environment_info.json"))

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------
    df_tr, Xtr, ytr, feat_cols = load_split(args.train_csv, feat_prefix=args.feat_prefix)
    df_va, Xva, yva, feat_cols_va = load_split(args.val_csv, feat_prefix=args.feat_prefix)
    df_te, Xte, yte, feat_cols_te = load_split(args.test_csv, feat_prefix=args.feat_prefix)

    check_feature_alignment(feat_cols, feat_cols_va, "val")
    check_feature_alignment(feat_cols, feat_cols_te, "test")

    save_txt_lines(feat_cols, os.path.join(run_dirs["artifacts"], "feature_names.txt"))

    dataset_summary = pd.DataFrame([
        summarize_split("train", df_tr, feat_cols, ytr),
        summarize_split("val", df_va, feat_cols, yva),
        summarize_split("test", df_te, feat_cols, yte),
    ])
    dataset_summary.to_csv(os.path.join(run_dirs["tables"], "dataset_summary.csv"), index=False)

    pos_ratio = float(ytr.mean())
    pos_weight = (1.0 - pos_ratio) / max(pos_ratio, 1e-9)

    print(f"[INFO] train size={len(ytr)} pos_ratio={pos_ratio:.4f} device={device}")

    factories, availability = build_model_factories(seed=args.seed, pos_ratio=pos_ratio)
    save_json(availability, os.path.join(run_dirs["artifacts"], "model_availability.json"))

    # -----------------------------------------------------
    # Step 1: traditional ML benchmark
    # -----------------------------------------------------
    print("\n[Step1] Traditional ML benchmark")
    rows = []
    val_prob_df = pd.DataFrame(index=np.arange(len(yva)))
    test_prob_df = pd.DataFrame(index=np.arange(len(yte)))

    for name in factories.keys():
        print(f"[Step1][ML] Training {name} ...")
        model = factories[name]()
        fit_with_early_stopping_on_val(model, Xtr, ytr, Xva, yva, name=name)

        prob_val = predict_proba_any(model, Xva).astype(np.float32)
        prob_test = predict_proba_any(model, Xte).astype(np.float32)

        thr, _ = find_best_threshold(yva, prob_val, metric="mcc")
        m_val = compute_metrics(yva, prob_val, threshold=thr)
        m_te = compute_metrics(yte, prob_test, threshold=thr)
        ece_val = expected_calibration_error(yva, prob_val, n_bins=args.ece_bins)
        ece_test = expected_calibration_error(yte, prob_test, n_bins=args.ece_bins)

        row = {
            "Model": name,
            "Threshold": float(thr),
            "ECE_VAL": float(ece_val),
            "ECE_TEST": float(ece_test),
        }
        row.update({f"VAL_{k}": v for k, v in m_val.items()})
        row.update({f"TEST_{k}": v for k, v in m_te.items()})
        rows.append(row)

        val_prob_df[name] = prob_val
        test_prob_df[name] = prob_test

        joblib.dump(model, os.path.join(run_dirs["single_models"], f"{name}.joblib"))
        np.save(os.path.join(run_dirs["predictions"], f"val_{name}.npy"), prob_val)
        np.save(os.path.join(run_dirs["predictions"], f"test_{name}.npy"), prob_test)
        save_prediction_csv(df_va, yva, prob_val, thr, os.path.join(run_dirs["predictions"], f"val_{name}.csv"))
        save_prediction_csv(df_te, yte, prob_test, thr, os.path.join(run_dirs["predictions"], f"test_{name}.csv"))

    df_ml = pd.DataFrame(rows).sort_values("VAL_MCC", ascending=False)
    df_ml.to_csv(os.path.join(run_dirs["tables"], "results_step1_ml.csv"), index=False)

    corr_val = plot_prob_correlation_heatmap(
        val_prob_df,
        os.path.join(run_dirs["figures"], "Fig_BaseModelCorrelation_Val.pdf"),
        "Correlation of base-model probabilities on validation set"
    )
    corr_val.to_csv(os.path.join(run_dirs["tables"], "base_model_probability_correlation_val.csv"))

    # -----------------------------------------------------
    # Step 1b: MLP1 on raw fused features
    # -----------------------------------------------------
    print("\n[Step1][MLP1] Hyperparameter search")
    grid_rows = []
    best_info, best_model, best_score, best_prob_val = None, None, -1e18, None

    for hidden in parse_hidden_list(args.mlp1_hidden):
        for dp in args.mlp1_dropout:
            for lr_ in args.mlp1_lr:
                for wd in args.mlp1_wd:
                    for bs in args.mlp1_batch:
                        cfg = MLPConfig(
                            hidden=hidden,
                            dropout=dp,
                            lr=lr_,
                            weight_decay=wd,
                            batch_size=bs,
                            max_epochs=args.mlp1_max_epochs,
                            patience=args.mlp1_patience
                        )
                        info, prob_val, model_scaler = train_mlp(
                            Xtr, ytr, Xva, yva, cfg,
                            device=device, pos_weight=pos_weight, seed=args.seed
                        )
                        thr = info["best_threshold"]
                        m_val = compute_metrics(yva, prob_val, threshold=thr)
                        score = m_val["MCC"]

                        rec = {"Model": "MLP1", **info, **{f"VAL_{k}": v for k, v in m_val.items()}}
                        grid_rows.append(rec)

                        print(f"[MLP1] hidden={hidden}, dp={dp}, lr={lr_} -> VAL_MCC={score:.4f}")

                        if score > best_score:
                            best_score = score
                            best_info = info.copy()
                            best_model = model_scaler
                            best_prob_val = prob_val.copy()

    df_mlp1_grid = pd.DataFrame(grid_rows).sort_values("VAL_MCC", ascending=False)
    df_mlp1_grid.to_csv(os.path.join(run_dirs["tables"], "mlp1_grid_results.csv"), index=False)

    prob_test_mlp1 = mlp_predict(best_model, Xte, device=device)
    thr_mlp1 = float(best_info["best_threshold"])

    m_val_mlp1 = compute_metrics(yva, best_prob_val, threshold=thr_mlp1)
    m_te_mlp1 = compute_metrics(yte, prob_test_mlp1, threshold=thr_mlp1)

    save_mlp_bundle(
        os.path.join(run_dirs["single_models"], "MLP1_best.pt"),
        best_model,
        cfg={
            "hidden": best_info["hidden"],
            "dropout": best_info["dropout"],
            "lr": best_info["lr"],
            "weight_decay": best_info["weight_decay"],
            "batch_size": best_info["batch_size"],
            "max_epochs": int(args.mlp1_max_epochs),
            "patience": int(args.mlp1_patience),
        },
        feature_dim=Xtr.shape[1],
        input_feature_names=feat_cols,
        threshold=thr_mlp1,
        extra_info={"stage": "MLP1_raw_features"},
    )

    save_json({
        "best_config": best_info,
        "VAL": m_val_mlp1,
        "TEST": m_te_mlp1,
    }, os.path.join(run_dirs["single_models"], "MLP1_best.json"))

    np.save(os.path.join(run_dirs["predictions"], "val_MLP1_best.npy"), best_prob_val)
    np.save(os.path.join(run_dirs["predictions"], "test_MLP1_best.npy"), prob_test_mlp1)
    save_prediction_csv(df_va, yva, best_prob_val, thr_mlp1, os.path.join(run_dirs["predictions"], "val_MLP1_best.csv"))
    save_prediction_csv(df_te, yte, prob_test_mlp1, thr_mlp1, os.path.join(run_dirs["predictions"], "test_MLP1_best.csv"))

    df_step1 = pd.concat([
        df_ml,
        pd.DataFrame([{
            "Model": "MLP1(best)",
            "Threshold": float(thr_mlp1),
            "ECE_VAL": expected_calibration_error(yva, best_prob_val, n_bins=args.ece_bins),
            "ECE_TEST": expected_calibration_error(yte, prob_test_mlp1, n_bins=args.ece_bins),
            **{f"VAL_{k}": v for k, v in m_val_mlp1.items()},
            **{f"TEST_{k}": v for k, v in m_te_mlp1.items()},
        }])
    ], ignore_index=True).sort_values("VAL_MCC", ascending=False)
    df_step1.to_csv(os.path.join(run_dirs["tables"], "results_step1_summary.csv"), index=False)

    # -----------------------------------------------------
    # Step 2: select traditional ML base models -> OOF meta-learning
    # -----------------------------------------------------
    selected_ml_models, within_family_table, selected_family_table, selection_logic = select_models_by_family(
        df_ml=df_ml,
        val_prob_df=val_prob_df,
        total_topk=args.topk,
        boosting_corr_threshold=args.boosting_corr_threshold,
    )

    if len(selected_ml_models) < 3:
        raise RuntimeError(f"Family-constrained selection produced too few traditional ML models: {selected_ml_models}")

    within_family_table.to_csv(os.path.join(run_dirs["tables"], "within_family_model_ranking.csv"), index=False)
    selected_family_table.to_csv(os.path.join(run_dirs["tables"], "selected_base_models_by_family.csv"), index=False)
    save_json(selection_logic, os.path.join(run_dirs["artifacts"], "selected_base_models_logic.json"))
    save_txt_lines(selected_ml_models, os.path.join(run_dirs["artifacts"], "selected_base_models.txt"))

    selected_corr_val = val_prob_df[selected_ml_models].corr(method="pearson")
    selected_corr_val.to_csv(os.path.join(run_dirs["tables"], "selected_base_model_correlation_val.csv"))

    plot_prob_correlation_heatmap(
        val_prob_df[selected_ml_models],
        os.path.join(run_dirs["figures"], "Fig_SelectedBaseModelCorrelation_Val.pdf"),
        "Correlation of selected base-model probabilities on validation set",
    )

    top_names = selected_ml_models
    print(f"\n[Step2] Selected traditional ML models ({len(top_names)}): {top_names}")

    oof_train = make_oof_probs(
        factories=factories,
        model_names=top_names,
        X=Xtr,
        y=ytr,
        n_splits=args.oof_folds,
        seed=args.seed
    )

    P_val, P_test = [], []
    for name in top_names:
        m = joblib.load(os.path.join(run_dirs["single_models"], f"{name}.joblib"))
        P_val.append(predict_proba_any(m, Xva))
        P_test.append(predict_proba_any(m, Xte))

    P_val = np.stack(P_val, axis=1).astype(np.float32)
    P_test = np.stack(P_test, axis=1).astype(np.float32)

    np.save(os.path.join(run_dirs["artifacts"], "oof_train_topk.npy"), oof_train)
    np.save(os.path.join(run_dirs["artifacts"], "val_topk_prob.npy"), P_val)
    np.save(os.path.join(run_dirs["artifacts"], "test_topk_prob.npy"), P_test)

    oof_df = pd.DataFrame({"PDB_ID": df_tr["PDB_ID"].values, "y": ytr.astype(int)})
    for i, x in enumerate(top_names):
        oof_df[f"OOF_{x}"] = oof_train[:, i]
    oof_df.to_csv(os.path.join(run_dirs["tables"], "oof_train_topk.csv"), index=False)

    val_topk_df = pd.DataFrame({"PDB_ID": df_va["PDB_ID"].values, "y": yva.astype(int)})
    for i, x in enumerate(top_names):
        val_topk_df[f"VAL_{x}"] = P_val[:, i]
    val_topk_df.to_csv(os.path.join(run_dirs["tables"], "val_topk_prob.csv"), index=False)

    test_topk_df = pd.DataFrame({"PDB_ID": df_te["PDB_ID"].values, "y": yte.astype(int)})
    for i, x in enumerate(top_names):
        test_topk_df[f"TEST_{x}"] = P_test[:, i]
    test_topk_df.to_csv(os.path.join(run_dirs["tables"], "test_topk_prob.csv"), index=False)

    # -----------------------------------------------------
    # Meta comparison
    # -----------------------------------------------------
    df_meta, best_artifact = run_meta_experiments(
        args=args,
        run_dirs=run_dirs,
        feat_cols=feat_cols,
        df_va=df_va,
        df_te=df_te,
        Xtr=Xtr,
        ytr=ytr,
        Xva=Xva,
        yva=yva,
        Xte=Xte,
        yte=yte,
        top_names=top_names,
        oof_train=oof_train,
        P_val=P_val,
        P_test=P_test,
        pos_weight=pos_weight,
        device=device,
    )

    # -----------------------------------------------------
    # Leaderboard & figures
    # -----------------------------------------------------
    leaderboard = pd.concat([
        df_step1[["Model", "VAL_MCC", "TEST_MCC", "VAL_PR_AUC", "TEST_PR_AUC", "VAL_ROC_AUC", "TEST_ROC_AUC"]],
        df_meta[["Model", "VAL_MCC", "TEST_MCC", "VAL_PR_AUC", "TEST_PR_AUC", "VAL_ROC_AUC", "TEST_ROC_AUC"]],
    ], ignore_index=True).sort_values(["VAL_MCC", "TEST_MCC"], ascending=False)

    leaderboard.to_csv(os.path.join(run_dirs["tables"], "leaderboard_all_models.csv"), index=False)

    plot_bar_comparison(
        leaderboard,
        "TEST_MCC",
        os.path.join(run_dirs["figures"], "Fig_ModelComparison_Test_MCC.pdf"),
        "Model comparison on test set (MCC)",
        topn=min(12, len(leaderboard))
    )

    plot_bar_comparison(
        leaderboard,
        "TEST_PR_AUC",
        os.path.join(run_dirs["figures"], "Fig_ModelComparison_Test_PRAUC.pdf"),
        "Model comparison on test set (PR-AUC)",
        topn=min(12, len(leaderboard))
    )

    # -----------------------------------------------------
    # Final selected model
    # -----------------------------------------------------
    final_name = best_artifact["name"]
    final_val_prob = best_artifact["prob_val"]
    final_test_prob = best_artifact["prob_test"]
    final_thr = float(best_artifact["threshold"])

    final_val_metrics = compute_metrics(yva, final_val_prob, final_thr)
    final_test_metrics = compute_metrics(yte, final_test_prob, final_thr)
    final_ece_val = expected_calibration_error(yva, final_val_prob, n_bins=args.ece_bins)
    final_ece_test = expected_calibration_error(yte, final_test_prob, n_bins=args.ece_bins)

    save_prediction_csv(
        df_va, yva, final_val_prob, final_thr,
        os.path.join(run_dirs["final_model"], "final_val_predictions.csv")
    )
    save_prediction_csv(
        df_te, yte, final_test_prob, final_thr,
        os.path.join(run_dirs["final_model"], "final_test_predictions.csv")
    )

    save_json({
        "selected_model": final_name,
        "threshold": final_thr,
        "VAL": final_val_metrics,
        "TEST": final_test_metrics,
        "ECE_VAL": final_ece_val,
        "ECE_TEST": final_ece_test,
        "top_models": top_names,
        "academic_note": (
            "Validation was used for model selection and threshold selection. "
            "Test was used only for final reporting."
        ),
    }, os.path.join(run_dirs["final_model"], "final_model_performance.json"))

    save_final_inference_package(
        run_dirs=run_dirs,
        feat_cols=feat_cols,
        top_names=top_names,
        best_artifact=best_artifact,
    )

    plot_roc_pr(yva, final_val_prob, final_name, run_dirs["figures"], "val")
    plot_roc_pr(yte, final_test_prob, final_name, run_dirs["figures"], "test")

    cal_val_df = plot_reliability(
        yva, final_val_prob,
        os.path.join(run_dirs["figures"], f"Fig_{final_name}_Val_Reliability.pdf"),
        f"Reliability diagram ({final_name}, val)",
        n_bins=args.ece_bins
    )
    cal_test_df = plot_reliability(
        yte, final_test_prob,
        os.path.join(run_dirs["figures"], f"Fig_{final_name}_Test_Reliability.pdf"),
        f"Reliability diagram ({final_name}, test)",
        n_bins=args.ece_bins
    )

    cal_val_df.to_csv(os.path.join(run_dirs["tables"], f"reliability_{final_name}_val.csv"), index=False)
    cal_test_df.to_csv(os.path.join(run_dirs["tables"], f"reliability_{final_name}_test.csv"), index=False)

    run_manifest = {
        "script": os.path.basename(__file__),
        "timestamp": timestamp,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "device": device,
        "seed": int(args.seed),
        "topk": int(len(top_names)),
        "oof_folds": int(args.oof_folds),
        "feat_prefix": args.feat_prefix,
        "ece_bins": int(args.ece_bins),
        "run_extended_compare": bool(args.run_extended_compare),
        "selected_final_model": final_name,
        "selected_base_models": top_names,
        "base_model_selection_strategy": "family_constrained",
        "boosting_corr_threshold": float(args.boosting_corr_threshold),
        "final_threshold": final_thr,
        "final_val_mcc": float(final_val_metrics["MCC"]),
        "final_test_mcc": float(final_test_metrics["MCC"]),
    }
    save_json(run_manifest, os.path.join(run_root, "run_manifest.json"))

    print("\n[OK] Formal training finished.")
    print(f"Run directory: {run_root}")
    print(f"Selected final model: {final_name}")
    print(f"Final VAL_MCC={final_val_metrics['MCC']:.4f} | TEST_MCC={final_test_metrics['MCC']:.4f}")
    print(f"Tables: {run_dirs['tables']}")
    print(f"Figures: {run_dirs['figures']}")
    print(f"Inference package: {run_dirs['final_model']}")


if __name__ == "__main__":
    main()