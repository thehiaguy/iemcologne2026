"""Walk-forward backtest of the match model on CS2-only history.

Trains on the past, predicts the next chronological block, and pools the
out-of-sample predictions to measure accuracy, log-loss, Brier and ROC-AUC against
rating-only baselines, plus a calibration (reliability) curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit

from .config import FEATURE_COLUMNS
from .model import MatchModel, RATING_MEMBERS


def _metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": accuracy_score(y, p > 0.5),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        "n": len(y),
    }


def backtest(frame: pd.DataFrame, n_splits: int = 3) -> dict:
    frame = frame.sort_values("date").reset_index(drop=True)
    cols = list(dict.fromkeys(FEATURE_COLUMNS + RATING_MEMBERS + ["same_region"]))

    yt, pm, pe, pv, xr = [], [], [], [], []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for k, (tr, va) in enumerate(tscv.split(frame), 1):
        model = MatchModel().fit(frame.iloc[tr], verbose=False)
        fv = frame.iloc[va]
        yt.append(fv["y"].to_numpy())
        pm.append(model.predict_proba(fv[cols]))
        pe.append(fv["elo_p"].to_numpy())
        pv.append(fv["vrs_p"].to_numpy())
        xr.append((fv["same_region"].to_numpy() == 0))
        print(f"  fold {k}: train={len(tr)} test={len(va)} blend_w={model.vrs_blend:.2f}")

    y = np.concatenate(yt)
    p_model = np.concatenate(pm)
    p_elo = np.concatenate(pe)
    p_vrs = np.concatenate(pv)
    cross = np.concatenate(xr)

    out = {
        "ensemble": _metrics(y, p_model),
        "elo_only": _metrics(y, p_elo),
        "vrs_only": _metrics(y, p_vrs),
        "ensemble_cross_region": _metrics(y[cross], p_model[cross]),
        "elo_cross_region": _metrics(y[cross], p_elo[cross]),
        "_reliability": (y, p_model),
    }
    return out


def compare_models(frame: pd.DataFrame, test_frac: float = 0.2,
                   new_member: str = "knn") -> pd.DataFrame:
    """Time-holdout comparison of every base learner vs the stacked ensemble.

    Trains on the earliest ``1 - test_frac`` of matches and scores the most recent
    ``test_frac`` (out-of-sample, time-respecting). Returns a metrics table for:
      * each individual base learner (pre-ensemble),
      * the Elo / VRS rating signals (reference),
      * the ensemble **without** ``new_member`` (pre adding the new model),
      * the **full** ensemble (post).
    """
    from sklearn.base import clone

    fr = frame.sort_values("date").reset_index(drop=True)
    k = int(len(fr) * (1 - test_frac))
    tr, te = fr.iloc[:k], fr.iloc[k:]
    Xtr, ytr = tr[FEATURE_COLUMNS].to_numpy(float), tr["y"].to_numpy()
    Xte, yte = te[FEATURE_COLUMNS].to_numpy(float), te["y"].to_numpy()
    cols = list(dict.fromkeys(FEATURE_COLUMNS + RATING_MEMBERS + ["same_region"]))

    rows = {}
    for name, mdl in MatchModel().base.items():               # individual learners
        m = clone(mdl).fit(Xtr, ytr)
        rows[name] = _metrics(yte, m.predict_proba(Xte)[:, 1])
    for col in ("elo_p", "vrs_p"):                            # rating references
        rows[col] = _metrics(yte, te[col].to_numpy())

    e0 = MatchModel()                                         # ensemble pre new model
    e0.base.pop(new_member, None)
    e0.fit(tr, verbose=False)
    rows[f"ENSEMBLE (no {new_member})"] = _metrics(yte, e0.predict_proba(te[cols]))

    e1 = MatchModel().fit(tr, verbose=False)                  # full ensemble (post)
    rows["ENSEMBLE (full)"] = _metrics(yte, e1.predict_proba(te[cols]))

    return (pd.DataFrame(rows).T[["accuracy", "log_loss", "brier", "auc", "n"]]
            .sort_values("log_loss"))


def print_report(out: dict) -> None:
    print("\n=== Walk-forward backtest (out-of-sample) ===")
    header = f"{'model':24} {'n':>7} {'acc':>7} {'logloss':>9} {'brier':>7} {'auc':>7}"
    print(header)
    print("-" * len(header))
    for name in ["ensemble", "elo_only", "vrs_only",
                 "ensemble_cross_region", "elo_cross_region"]:
        m = out[name]
        print(f"{name:24} {m['n']:>7} {m['accuracy']:>7.3f} "
              f"{m['log_loss']:>9.4f} {m['brier']:>7.4f} {m['auc']:>7.3f}")
    e, b = out["ensemble"]["log_loss"], out["elo_only"]["log_loss"]
    print(f"\nensemble beats Elo-only on log-loss by {b - e:+.4f} "
          f"({'PASS' if e < b else 'FAIL'})")
