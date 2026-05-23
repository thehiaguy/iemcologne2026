"""Leak-free, chronological feature construction.

A single forward pass over the CS2 match history maintains online state (Elo,
Glicko-2, recent form, activity, streaks, head-to-head) plus as-of lookups into the
monthly VRS standings (points, roster stability). For every match the *pre-match*
features are recorded, then the result is observed to update the state. After the
pass the same `Context` holds each team's present strength for prediction.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import config
from .ratings import Elo, Glicko2

FORM_ALPHA = 0.20            # EWMA weight on most recent result
ACTIVITY_WINDOW_DAYS = 90
VRS_SCALE = 250.0           # logistic scale converting VRS-point diff -> prob
VRS_UNRANKED = 1000.0       # imputed strength for teams absent from the standings
ROSTER_CAP_DAYS = 540.0
H2H_CAP = 5                 # clamp head-to-head differential
SOS_ALPHA = 0.05            # EWMA weight for opponent-strength (strength of schedule)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# Module-level factories so a fitted Context is picklable (lambdas are not).
def _form_default() -> float:
    return 0.5


def _h2h_default() -> list:
    return [0, 0]


class Context:
    """Evolving team state + as-of standings lookups."""

    def __init__(self, standings: pd.DataFrame, region_map: dict[str, str]):
        self.elo = Elo()
        self.glicko = Glicko2()
        self.form: dict[str, float] = defaultdict(_form_default)
        self.streak: dict[str, int] = defaultdict(int)
        self.activity: dict[str, deque] = defaultdict(deque)
        self.h2h: dict[tuple, list] = defaultdict(_h2h_default)
        self.sos: dict[str, float] = {}        # EWMA of opponents' VRS points
        self.region = region_map

        # Per-team VRS point time series (sorted) for as-of lookups.
        self._vrs_dates: dict[str, np.ndarray] = {}
        self._vrs_points: dict[str, np.ndarray] = {}
        self._roster_dates: dict[str, np.ndarray] = {}
        self._roster_vals: dict[str, list] = {}
        for team, grp in standings.sort_values("date").groupby("team"):
            self._vrs_dates[team] = grp["date"].values.astype("datetime64[ns]")
            self._vrs_points[team] = grp["points"].to_numpy(dtype=float)
            self._roster_dates[team] = grp["date"].values.astype("datetime64[ns]")
            self._roster_vals[team] = grp["roster"].tolist()

    # ---- as-of standings lookups ------------------------------------------ #
    def vrs_points(self, team: str, date) -> float:
        d = self._vrs_dates.get(team)
        if d is None:
            return VRS_UNRANKED
        i = np.searchsorted(d, np.datetime64(date), side="left") - 1
        return float(self._vrs_points[team][i]) if i >= 0 else VRS_UNRANKED

    def has_vrs(self, team: str, date) -> bool:
        d = self._vrs_dates.get(team)
        if d is None:
            return False
        return np.searchsorted(d, np.datetime64(date), side="left") - 1 >= 0

    def roster_stability(self, team: str, date) -> float:
        """Days the current roster has been intact as of `date` (capped)."""
        d = self._roster_dates.get(team)
        if d is None:
            return 0.0
        nd = np.datetime64(date)
        i = np.searchsorted(d, nd, side="left") - 1
        if i < 0:
            return 0.0
        cur = self._roster_vals[team][i]
        j = i
        while j - 1 >= 0 and self._roster_vals[team][j - 1] == cur:
            j -= 1
        days = (nd - d[j]) / np.timedelta64(1, "D")
        return float(min(days, ROSTER_CAP_DAYS))

    # ---- online helpers ---------------------------------------------------- #
    def _activity(self, team: str, date) -> int:
        dq = self.activity[team]
        cutoff = pd.Timestamp(date) - pd.Timedelta(days=ACTIVITY_WINDOW_DAYS)
        return sum(1 for d in dq if d >= cutoff)

    # ---- feature vector ---------------------------------------------------- #
    def compute_features(self, a: str, b: str, date) -> dict:
        vrs_a, vrs_b = self.vrs_points(a, date), self.vrs_points(b, date)
        vrs_diff = vrs_a - vrs_b
        ha, hb = self.h2h[(a, b)][0], self.h2h[(a, b)][1]
        h2h_diff = max(-H2H_CAP, min(H2H_CAP, ha - hb))
        same_region = float(
            self.region.get(a, "?") == self.region.get(b, "?") != "?"
        )
        feats = {
            "elo_diff": self.elo.rating(a) - self.elo.rating(b),
            "glicko_diff": self.glicko.rating(a) - self.glicko.rating(b),
            "glicko_rd_sum": self.glicko.rd(a) + self.glicko.rd(b),
            "vrs_diff": vrs_diff,
            "sos_diff": self.sos.get(a, VRS_UNRANKED) - self.sos.get(b, VRS_UNRANKED),
            "form_diff": self.form[a] - self.form[b],
            "h2h_diff": float(h2h_diff),
            "activity_diff": self._activity(a, date) - self._activity(b, date),
            "streak_diff": float(self.streak[a] - self.streak[b]),
            "roster_stab_diff": (
                self.roster_stability(a, date) - self.roster_stability(b, date)
            ),
            "same_region": same_region,
        }
        # Rating-model member probabilities (inputs to the stacking meta-learner)
        feats["elo_p"] = self.elo.expected(a, b)
        feats["glicko_p"] = self.glicko.expected(a, b)
        feats["vrs_p"] = _sigmoid(vrs_diff / VRS_SCALE)
        return feats

    # ---- observe a result -------------------------------------------------- #
    def observe(self, winner: str, loser: str, date) -> None:
        # Strength-of-schedule: blend in each side's opponent VRS *before* updating
        # ratings, so a team that only beats weak opponents is flagged as such.
        vw, vl = self.vrs_points(winner, date), self.vrs_points(loser, date)
        self.sos[winner] = (1 - SOS_ALPHA) * self.sos.get(winner, VRS_UNRANKED) + SOS_ALPHA * vl
        self.sos[loser] = (1 - SOS_ALPHA) * self.sos.get(loser, VRS_UNRANKED) + SOS_ALPHA * vw

        self.elo.update(winner, loser)
        self.glicko.update(winner, loser)
        self.form[winner] = (1 - FORM_ALPHA) * self.form[winner] + FORM_ALPHA * 1.0
        self.form[loser] = (1 - FORM_ALPHA) * self.form[loser] + FORM_ALPHA * 0.0
        self.streak[winner] = max(1, self.streak[winner] + 1)
        self.streak[loser] = min(-1, self.streak[loser] - 1)
        ts = pd.Timestamp(date)
        for t in (winner, loser):
            self.activity[t].append(ts)
        self.h2h[(winner, loser)][0] += 1
        self.h2h[(loser, winner)][1] += 1


def build(matches: pd.DataFrame, standings: pd.DataFrame,
          region_map: dict[str, str], seed: int = config.RANDOM_SEED):
    """Return (feature_frame, fitted Context).

    Orientation (which team is "a") is randomised per match so the binary target is
    balanced and the signed-difference features carry all the directional signal.
    """
    rng = np.random.default_rng(seed)
    ctx = Context(standings, region_map)
    rows = []
    for date, winner, loser in matches[["date", "winner", "loser"]].itertuples(
        index=False
    ):
        a, b = (winner, loser) if rng.random() < 0.5 else (loser, winner)
        feats = ctx.compute_features(a, b, date)
        feats["y"] = 1 if a == winner else 0
        feats["date"] = date
        feats["team_a"] = a
        feats["team_b"] = b
        feats["both_ranked"] = int(ctx.has_vrs(a, date) and ctx.has_vrs(b, date))
        rows.append(feats)
        ctx.observe(winner, loser, date)
    frame = pd.DataFrame(rows)
    return frame, ctx
