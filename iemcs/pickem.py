"""Stage-1 Pick'Em optimisation.

The Major Pick'Em asks for 2 teams to go 3-0, 6 teams to advance (top-8), and 2 teams
to go 0-3; you must get at least `threshold` of the 10 picks right. Because the picks
are correlated (exactly two teams finish 3-0 and two finish 0-3) and the win condition
is a *threshold*, the best pick-set maximises ``P(>= threshold correct)`` — which is not
the same as taking the single most-likely outcome in each slot.

We Monte-Carlo the real Stage-1 Swiss bracket, then search pick-sets over the simulated
outcomes (so correlations are handled exactly).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import tournament
from .swiss import simulate_swiss
from .teams import stage_teams


def simulate_stage1(model, ctx, vrs_pts, as_of_date, n_sims=40_000, seed=7):
    """Run the Stage-1 Swiss `n_sims` times (VRS seeding = the official bracket).

    Returns (teams_in_seed_order, M3, MA, M03) where each M* is an (n_sims x 16)
    0/1 array of 3-0 / advance / 0-3 outcomes.
    """
    pmap = tournament.build_pmap(model, ctx, as_of_date)

    def pm(a, b):
        return pmap[(a, b)]

    teams = sorted(stage_teams(1), key=lambda t: -vrs_pts[t])
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    M3 = np.zeros((n_sims, n), np.int8)
    MA = np.zeros((n_sims, n), np.int8)
    M03 = np.zeros((n_sims, n), np.int8)
    rng = np.random.default_rng(seed)
    for k in range(n_sims):
        adv, _elim, rec = simulate_swiss(teams, pm, rng, with_records=True)
        for t in adv:
            MA[k, idx[t]] = 1
        for t, (w, l) in rec.items():
            if (w, l) == (3, 0):
                M3[k, idx[t]] = 1
            elif (w, l) == (0, 3):
                M03[k, idx[t]] = 1
    return teams, M3, MA, M03


def probabilities(teams, M3, MA, M03) -> pd.DataFrame:
    return pd.DataFrame({
        "team": teams, "P_advance": MA.mean(0),
        "P_3_0": M3.mean(0), "P_0_3": M03.mean(0),
    }).sort_values("P_advance", ascending=False).reset_index(drop=True)


def _p_at_least(M3, MA, M03, a3, adv, a03, thr) -> float:
    cnt = M3[:, a3].sum(1) + MA[:, adv].sum(1) + M03[:, a03].sum(1)
    return float((cnt >= thr).mean())


def _names(teams, a3, adv, a03) -> dict:
    return {"3-0": [teams[i] for i in a3],
            "advance": [teams[i] for i in adv],
            "0-3": [teams[i] for i in a03]}


def optimize(teams, M3, MA, M03, threshold=5, n30=2, nadv=6, n03=2) -> dict:
    """Compare the naive 'most-likely per slot' picks against the pick-set that
    maximises P(>= threshold correct) (found by searching the simulated outcomes)."""
    P3, PA, P03 = M3.mean(0), MA.mean(0), M03.mean(0)
    nteams = len(teams)

    # Naive: most likely outcome in each slot, greedily.
    a03 = list(np.argsort(-P03)[:n03])
    rem = [i for i in range(nteams) if i not in a03]
    a3 = sorted(rem, key=lambda i: -P3[i])[:n30]
    rem2 = [i for i in rem if i not in a3]
    adv = sorted(rem2, key=lambda i: -PA[i])[:nadv]
    naive = {"picks": _names(teams, a3, adv, a03),
             "p": _p_at_least(M3, MA, M03, a3, adv, a03, threshold)}

    # Search: strongest teams tend to belong in 'advance', so the live candidates are
    # the top P_advance teams; 3-0 / 0-3 from the best remaining specialists.
    c03 = list(np.argsort(-P03)[:6])
    c30 = list(np.argsort(-P3)[:8])
    cadv = list(np.argsort(-PA)[: nadv + 3])
    best = (-1.0, None)
    for z in itertools.combinations(c03, n03):
        for th in itertools.combinations(c30, n30):
            if set(th) & set(z):
                continue
            pool = [i for i in cadv if i not in z and i not in th]
            for ad in itertools.combinations(pool, nadv):
                p = _p_at_least(M3, MA, M03, list(th), list(ad), list(z), threshold)
                if p > best[0]:
                    best = (p, (list(th), list(ad), list(z)))
    th, ad, z = best[1]
    optimal = {"picks": _names(teams, th, ad, z), "p": best[0]}
    curve = {f"P>={k}": _p_at_least(M3, MA, M03, th, ad, z, k)
             for k in (threshold - 1, threshold, threshold + 1)}
    return {"naive": naive, "optimal": optimal, "curve": curve, "threshold": threshold}
