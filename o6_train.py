"""
[6/7] train.py
--------------
Model training for IEM Cologne 2026 CS2 match prediction.

Reads data/raw_matches.parquet written by collect.py, then:
  1. Engineers rolling, H2H, ELO, and ranking features  (features.py)
  2. Time-based train / val / test split
  3. Trains PyTorch ResNet MLP          (neural_net.py)
  4. Trains XGBoost booster
  5. Grid-searches ensemble blend weight
  6. Evaluates on hold-out test set
  7. Saves all artifacts to models/

Usage
-----
  python o6_train.py

Run o5_collect.py at least once first to populate data/raw_matches.parquet.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from o1_config import (
    DATA_DIR, MODEL_DIR,
    CS2_LAUNCH_DATE, TRAIN_END_DATE, VAL_END_DATE,
    XGB_ESTIMATORS, XGB_MAX_DEPTH, XGB_LR, XGB_SUBSAMPLE,
    XGB_COLSAMPLE, XGB_MIN_CHILD, XGB_GAMMA, XGB_EARLY_STOP,
    BLEND_STEPS, RANDOM_SEED,
)
from o3_features import EloRatings, build_training_data, feature_columns
from o4_neural_net import MatchPredictor, train_pytorch, pytorch_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
) -> tuple:
    log.info("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=XGB_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LR,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        min_child_weight=XGB_MIN_CHILD,
        gamma=XGB_GAMMA,
        eval_metric="logloss",
        early_stopping_rounds=XGB_EARLY_STOP,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        device="cpu",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
    val_probs = model.predict_proba(X_va)[:, 1]
    val_auc   = roc_auc_score(y_va, val_probs) if len(set(y_va)) > 1 else 0.0
    log.info("XGBoost best val AUC: %.4f", val_auc)
    return model, val_auc


# ---------------------------------------------------------------------------
# Ensemble blend weight
# ---------------------------------------------------------------------------

def find_blend_weight(
    pt_probs: np.ndarray, xgb_probs: np.ndarray, y_true: np.ndarray
) -> float:
    best_w, best_auc = 0.5, 0.0
    for w in np.linspace(0, 1, BLEND_STEPS):
        blend = w * pt_probs + (1 - w) * xgb_probs
        auc   = roc_auc_score(y_true, blend) if len(set(y_true)) > 1 else 0.0
        if auc > best_auc:
            best_auc, best_w = auc, w
    log.info("Best blend weight (PyTorch): %.2f  ensemble val AUC: %.4f", best_w, best_auc)
    return float(best_w)


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    pt_model:     MatchPredictor,
    xgb_model:    XGBClassifier,
    scaler:       StandardScaler,
    feature_cols: List[str],
    elo:          EloRatings,
    blend_w:      float,
):
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(pt_model.state_dict(), MODEL_DIR / "pytorch_model.pt")
    xgb_model.save_model(str(MODEL_DIR / "xgb_model.json"))
    for obj, fname in [
        (scaler,            "scaler.pkl"),
        (feature_cols,      "feature_cols.pkl"),
        (elo,               "elo.pkl"),
        (blend_w,           "blend_weight.pkl"),
        (len(feature_cols), "input_dim.pkl"),
    ]:
        with open(MODEL_DIR / fname, "wb") as fh:
            pickle.dump(obj, fh)
    log.info("Artifacts saved to %s/", MODEL_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # --- 1. Load raw data ---
    raw_parquet  = DATA_DIR / "raw_matches.parquet"
    profiles_pkl = DATA_DIR / "team_profiles.pkl"

    if not raw_parquet.exists():
        raise FileNotFoundError(
            f"{raw_parquet} not found — run o5_collect.py first."
        )

    raw_df = pd.read_parquet(raw_parquet)
    log.info("Loaded %d matches from %s", len(raw_df), raw_parquet)

    pre_cs2 = (raw_df["match_date"] < CS2_LAUNCH_DATE).sum()
    raw_df   = raw_df[raw_df["match_date"] >= CS2_LAUNCH_DATE].reset_index(drop=True)
    log.info("Dropped %d pre-CS2 matches (before %s) — %d remain",
             pre_cs2, CS2_LAUNCH_DATE.date(), len(raw_df))

    profiles: Dict[int, dict] = {}
    if profiles_pkl.exists():
        with open(profiles_pkl, "rb") as fh:
            profiles = pickle.load(fh)
        log.info("Loaded profiles for %d teams", len(profiles))

    team_name_to_id: Dict[str, int] = {}
    for _, row in raw_df.iterrows():
        for col in ("team1_name", "team2_name"):
            name = row[col]
            if name and name not in team_name_to_id:
                team_name_to_id[name] = -1

    # --- 2. Feature engineering ---
    training_df, elo = build_training_data(raw_df, profiles, team_name_to_id)
    training_df.to_parquet(DATA_DIR / "training_data.parquet", index=False)

    feat_cols = feature_columns(training_df)

    # --- 3. Time-based splits ---
    train_df = training_df[training_df["match_date"] <= TRAIN_END_DATE]
    val_df   = training_df[
        (training_df["match_date"] > TRAIN_END_DATE) &
        (training_df["match_date"] <= VAL_END_DATE)
    ]
    test_df  = training_df[training_df["match_date"] > VAL_END_DATE]

    log.info("Split  train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    def _xy(split):
        X = split[feat_cols].fillna(0).values.astype(np.float32)
        y = split["label"].values.astype(np.float32)
        return X, y

    X_tr, y_tr = _xy(train_df)
    X_va, y_va = _xy(val_df)
    X_te, y_te = _xy(test_df)

    scaler  = StandardScaler()
    X_tr_s  = np.nan_to_num(scaler.fit_transform(X_tr)).astype(np.float32)
    X_va_s  = np.nan_to_num(scaler.transform(X_va)).astype(np.float32)
    X_te_s  = np.nan_to_num(scaler.transform(X_te)).astype(np.float32)
    input_dim = X_tr_s.shape[1]

    # --- 4 & 5. Train models ---
    pt_model,  pt_auc  = train_pytorch(X_tr_s, y_tr, X_va_s, y_va, input_dim)
    xgb_model, xgb_auc = train_xgboost(X_tr_s, y_tr, X_va_s, y_va)

    # --- 6. Ensemble blend ---
    if len(y_va) > 0 and len(set(y_va)) > 1:
        pt_val_probs  = pytorch_predict(pt_model, X_va_s)
        xgb_val_probs = xgb_model.predict_proba(X_va_s)[:, 1]
        blend_w       = find_blend_weight(pt_val_probs, xgb_val_probs, y_va)
    else:
        blend_w = 0.5

    # --- 7. Test evaluation ---
    if len(y_te) > 0 and len(set(y_te)) > 1:
        pt_te_probs  = pytorch_predict(pt_model, X_te_s)
        xgb_te_probs = xgb_model.predict_proba(X_te_s)[:, 1]
        ens_te_probs = blend_w * pt_te_probs + (1 - blend_w) * xgb_te_probs

        log.info("Test set results:")
        log.info("  PyTorch AUC  : %.4f", roc_auc_score(y_te, pt_te_probs))
        log.info("  XGBoost AUC  : %.4f", roc_auc_score(y_te, xgb_te_probs))
        log.info("  Ensemble AUC : %.4f", roc_auc_score(y_te, ens_te_probs))
        log.info("\n%s", classification_report(y_te, (ens_te_probs > 0.5).astype(int)))

    # --- Feature importance ---
    imp = pd.DataFrame({
        "feature":    feat_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(DATA_DIR / "feature_importance.csv", index=False)
    log.info("Top 25 features:\n%s", imp.head(25).to_string(index=False))

    # --- 8. Save ---
    save_artifacts(pt_model, xgb_model, scaler, feat_cols, elo, blend_w)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
