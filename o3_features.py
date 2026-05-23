"""
[3/6] features.py
-----------------
Feature engineering shared by train.py and simulate.py.

Exports
-------
  EloRatings              — Elo rating system with history
  rolling_team_features() — rolling-window player-stat averages
  h2h_features()          — head-to-head win rate / count
  build_training_data()   — build full feature DataFrame from raw matches
  feature_columns()       — ordered list of model input columns
"""

import logging
import warnings
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from o1_config import ELO_K, ELO_INIT, ROLLING_WINDOWS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ELO rating system
# ---------------------------------------------------------------------------

class EloRatings:
    def __init__(self, k: int = ELO_K, init: int = ELO_INIT):
        self.k    = k
        self.init = init
        self.ratings: Dict[str, float]                        = defaultdict(partial(float, init))
        self.history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, team_a: str, team_b: str, score_a: float, date: datetime):
        ra, rb = self.ratings[team_a], self.ratings[team_b]
        ea     = self.expected(ra, rb)
        self.ratings[team_a] = ra + self.k * (score_a - ea)
        self.ratings[team_b] = rb + self.k * ((1 - score_a) - (1 - ea))
        self.history[team_a].append((date, self.ratings[team_a]))
        self.history[team_b].append((date, self.ratings[team_b]))

    def get_at(self, team: str, before: datetime) -> float:
        """Return the Elo rating for `team` as of just before `before`."""
        val = float(self.init)
        for dt, elo in self.history[team]:
            if dt < before:
                val = elo
            else:
                break
        return val


# ---------------------------------------------------------------------------
# Rolling window features
# ---------------------------------------------------------------------------

def rolling_team_features(
    df: pd.DataFrame, team_name: str, before_date: datetime, window: int
) -> Dict[str, float]:
    """
    Return rolling averages for `team_name` over the last `window` matches
    played strictly before `before_date`.
    """
    mask = (
        ((df["team1_name"] == team_name) | (df["team2_name"] == team_name))
        & (df["match_date"] < before_date)
    )
    recent = df[mask].sort_values("match_date", ascending=False).head(window)
    if recent.empty:
        return {}

    stats = {k: [] for k in (
        "rating", "adr", "kast", "opening_kd", "opening_kills",
        "multi_kills", "clutches", "flash_assists", "hs_kills",
    )}
    wins = []

    for _, row in recent.iterrows():
        slot = "team1" if row["team1_name"] == team_name else "team2"
        for k in stats:
            col = f"{slot}_avg_{k}"
            if col in row.index:
                stats[k].append(row[col])
        won = (slot == "team1" and row["label"] == 1) or \
              (slot == "team2" and row["label"] == 0)
        wins.append(int(won))

    def _m(lst):
        if not lst:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanmean(lst)
        return float(v) if np.isfinite(v) else np.nan

    def _s(lst):
        if len(lst) < 2:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanstd(lst))

    p = f"w{window}"
    return {
        f"{p}_avg_rating":        _m(stats["rating"]),
        f"{p}_avg_adr":           _m(stats["adr"]),
        f"{p}_avg_kast":          _m(stats["kast"]),
        f"{p}_avg_opening_kd":    _m(stats["opening_kd"]),
        f"{p}_avg_opening_kills": _m(stats["opening_kills"]),
        f"{p}_avg_multi_kills":   _m(stats["multi_kills"]),
        f"{p}_avg_clutches":      _m(stats["clutches"]),
        f"{p}_avg_flash_assists": _m(stats["flash_assists"]),
        f"{p}_avg_hs_kills":      _m(stats["hs_kills"]),
        f"{p}_win_rate":          _m(wins),
        f"{p}_form_last5":        float(sum(wins[:5])),
        f"{p}_rating_std":        _s(stats["rating"]),
        f"{p}_adr_std":           _s(stats["adr"]),
    }


# ---------------------------------------------------------------------------
# Head-to-head features
# ---------------------------------------------------------------------------

def h2h_features(
    df: pd.DataFrame, team_a: str, team_b: str,
    before_date: datetime, window: int = 20
) -> Dict[str, float]:
    """Return H2H stats for team_a vs team_b before a given date."""
    mask = (
        (
            ((df["team1_name"] == team_a) & (df["team2_name"] == team_b)) |
            ((df["team1_name"] == team_b) & (df["team2_name"] == team_a))
        )
        & (df["match_date"] < before_date)
    )
    h2h   = df[mask].sort_values("match_date", ascending=False).head(window)
    total = len(h2h)
    if total == 0:
        return {"h2h_win_rate": 0.5, "h2h_total": 0.0, "h2h_last5_wins": 2.0}

    def _a_won(r):
        return (r["team1_name"] == team_a and r["label"] == 1) or \
               (r["team2_name"] == team_a and r["label"] == 0)

    a_wins  = sum(1 for _, r in h2h.iterrows()         if _a_won(r))
    a_last5 = sum(1 for _, r in h2h.head(5).iterrows() if _a_won(r))
    return {
        "h2h_win_rate":   a_wins / total,
        "h2h_total":      float(total),
        "h2h_last5_wins": float(a_last5),
    }


# ---------------------------------------------------------------------------
# Full training dataset construction
# ---------------------------------------------------------------------------

def build_training_data(
    df: pd.DataFrame,
    profiles: Dict[int, dict],
    team_name_to_id: Dict[str, int],
) -> Tuple[pd.DataFrame, EloRatings]:
    """
    Build a feature row for every match in `df`.
    Returns (feature_df, fitted EloRatings).
    """
    elo = EloRatings()
    for _, row in df.iterrows():
        elo.update(row["team1_name"], row["team2_name"], float(row["label"]), row["match_date"])

    rows  = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            log.info("  Feature engineering %d / %d", i, total)

        date = row["match_date"]
        t1   = row["team1_name"]
        t2   = row["team2_name"]

        feat1: Dict[str, float] = {}
        feat2: Dict[str, float] = {}
        for w in ROLLING_WINDOWS:
            feat1.update(rolling_team_features(df, t1, date, w))
            feat2.update(rolling_team_features(df, t2, date, w))

        h2h = h2h_features(df, t1, t2, date)

        elo1 = elo.get_at(t1, date)
        elo2 = elo.get_at(t2, date)

        t1_id = team_name_to_id.get(t1)
        t2_id = team_name_to_id.get(t2)
        p1    = profiles.get(t1_id, {}) if t1_id else {}
        p2    = profiles.get(t2_id, {}) if t2_id else {}

        combined: dict = {
            "match_id":   row["match_id"],
            "match_date": date,
            "team1_name": t1,
            "team2_name": t2,
            "label":      int(row["label"]),
        }
        for k, v in feat1.items():
            combined[f"t1_{k}"] = v
        for k, v in feat2.items():
            combined[f"t2_{k}"] = v

        combined.update(h2h)

        combined["t1_elo"]           = elo1
        combined["t2_elo"]           = elo2
        combined["elo_diff"]         = elo1 - elo2
        combined["t1_world_ranking"] = float(p1.get("worldRanking",      999))
        combined["t2_world_ranking"] = float(p2.get("worldRanking",      999))
        combined["t1_valve_ranking"] = float(p1.get("valveRanking",      999))
        combined["t2_valve_ranking"] = float(p2.get("valveRanking",      999))
        combined["t1_weeks_top30"]   = float(p1.get("weeksInTop30",        0))
        combined["t2_weeks_top30"]   = float(p2.get("weeksInTop30",        0))
        combined["t1_avg_age"]       = float(p1.get("averagePlayerAge",   25))
        combined["t2_avg_age"]       = float(p2.get("averagePlayerAge",   25))
        combined["ranking_diff"]     = combined["t1_world_ranking"] - combined["t2_world_ranking"]

        for w in ROLLING_WINDOWS:
            for m in ["avg_rating", "avg_adr", "avg_kast", "avg_opening_kd",
                      "avg_multi_kills", "avg_clutches", "win_rate"]:
                a = feat1.get(f"w{w}_{m}") or 0.0
                b = feat2.get(f"w{w}_{m}") or 0.0
                combined[f"diff_{m}_{w}"] = a - b

        rows.append(combined)

    out = pd.DataFrame(rows)
    log.info("Training dataset: %s", out.shape)
    return out, elo


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the ordered list of numeric feature column names."""
    skip = {"match_id", "match_date", "team1_name", "team2_name", "label"}
    return [c for c in df.columns if c not in skip]
