"""Stacked, calibrated match-outcome model (the "ensemble of both").

Members
-------
* Rating models  : Elo, Glicko-2 and VRS expectations (computed leak-free upstream
                   in :mod:`iemcs.dataset` and passed in as member probabilities).
* Base learners  : logistic regression, random forest, hist-gradient-boosting
                   (+ XGBoost / LightGBM if importable) on the engineered features.
* Meta-learner   : logistic regression stacked on out-of-fold member probabilities.
* Calibration    : isotonic regression on out-of-fold stack predictions.

The model's native target is "win a (Bo3-equivalent) encounter"; :mod:`iemcs.series`
turns that into Bo1/Bo3/Bo5 series odds.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=UserWarning)

RATING_MEMBERS = ["elo_p", "glicko_p", "vrs_p"]


def _make_base_learners() -> dict:
    learners = {
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=30,
            n_jobs=-1, random_state=0,
        ),
        "gbm": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=0,
        ),
    }
    try:  # optional boosters
        from xgboost import XGBClassifier
        learners["xgb"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=0,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        learners["lgbm"] = LGBMClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1,
        )
    except Exception:
        pass
    return learners


class MatchModel:
    def __init__(self):
        self.base = _make_base_learners()
        self.meta = LogisticRegression(max_iter=2000)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.stack_columns: list[str] = []
        # Cross-region VRS correction: at a Major most matchups are cross-region,
        # a regime where region-inflated Elo misleads and the official VRS rating
        # (opponent-network aware) is more reliable. Weight tuned in fit().
        self.vrs_blend = 0.0
        self.fitted = False

    # ------------------------------------------------------------------ #
    def fit(self, frame: pd.DataFrame, n_splits: int = 5, verbose: bool = True):
        frame = frame.sort_values("date").reset_index(drop=True)
        X = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = frame["y"].to_numpy(dtype=int)
        n = len(frame)

        base_names = list(self.base)
        oof = {name: np.full(n, np.nan) for name in base_names}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for tr, va in tscv.split(X):
            for name, mdl in self.base.items():
                m = clone(mdl)
                m.fit(X[tr], y[tr])
                oof[name][va] = m.predict_proba(X[va])[:, 1]

        mask = ~np.isnan(oof[base_names[0]])  # rows with OOF predictions
        self.stack_columns = RATING_MEMBERS + base_names
        stack_X = np.column_stack(
            [frame[c].to_numpy(dtype=float)[mask] for c in RATING_MEMBERS]
            + [oof[name][mask] for name in base_names]
        )
        self.meta.fit(stack_X, y[mask])
        stack_oof = self.meta.predict_proba(stack_X)[:, 1]
        self.calibrator.fit(stack_oof, y[mask])
        cal_oof = self.calibrator.predict(stack_oof)

        # Tune the cross-region VRS blend on out-of-fold predictions, restricted to
        # cross-region matches between two ranked teams (the Major-like regime).
        from sklearn.metrics import log_loss
        ym = y[mask]
        vrs_oof = frame["vrs_p"].to_numpy(dtype=float)[mask]
        xr = (frame["same_region"].to_numpy(dtype=float)[mask] == 0) & (
            frame["both_ranked"].to_numpy(dtype=float)[mask] == 1
        )
        best_w, best_ll = 0.0, np.inf
        if xr.sum() > 200:
            for w in np.linspace(0.0, 0.9, 19):
                bl011 = np.clip((1 - w) * cal_oof[xr] + w * vrs_oof[xr], 1e-6, 1 - 1e-6)
                ll = log_loss(ym[xr], bl011)
                if ll < best_ll:
                    best_ll, best_w = ll, float(w)
        self.vrs_blend = best_w

        # Refit base learners on all data for inference.
        for name, mdl in self.base.items():
            mdl.fit(X, y)
        self.fitted = True

        if verbose:
            print(f"  base learners: {base_names}")
            print(f"  meta coefs   : "
                  + ", ".join(f"{c}={w:+.2f}"
                              for c, w in zip(self.stack_columns, self.meta.coef_[0])))
            print(f"  OOF log-loss : stack={log_loss(ym, stack_oof):.4f} "
                  f"calibrated={log_loss(ym, cal_oof):.4f}")
            print(f"  cross-region VRS blend w={self.vrs_blend:.2f} "
                  f"(n_xr={int(xr.sum())}); xr log-loss "
                  f"{log_loss(ym[xr], cal_oof[xr]):.4f} -> "
                  f"{log_loss(ym[xr], np.clip((1-self.vrs_blend)*cal_oof[xr]+self.vrs_blend*vrs_oof[xr],1e-6,1-1e-6)):.4f}")
        return self

    # ------------------------------------------------------------------ #
    def _stack_input(self, feat: pd.DataFrame) -> np.ndarray:
        X = feat[FEATURE_COLUMNS].to_numpy(dtype=float)
        cols = [feat[c].to_numpy(dtype=float) for c in RATING_MEMBERS]
        for name in self.base:
            cols.append(self.base[name].predict_proba(X)[:, 1])
        return np.column_stack(cols)

    def predict_proba(self, feat: pd.DataFrame) -> np.ndarray:
        """Calibrated P(team_a wins a Bo3-equivalent encounter), vectorised.

        For cross-region matchups the prediction is blended toward the VRS estimate
        (weight tuned in fit) to counter region-inflated Elo.
        """
        if not self.fitted:
            raise RuntimeError("model not fitted")
        cal = self.calibrator.predict(self.meta.predict_proba(self._stack_input(feat))[:, 1])
        if self.vrs_blend > 0:
            xr = feat["same_region"].to_numpy(dtype=float) == 0
            vrs_p = feat["vrs_p"].to_numpy(dtype=float)
            cal = np.where(
                xr, (1 - self.vrs_blend) * cal + self.vrs_blend * vrs_p, cal
            )
        return np.clip(cal, 1e-6, 1 - 1e-6)

    def predict_one(self, feat: dict) -> float:
        return float(self.predict_proba(pd.DataFrame([feat]))[0])
