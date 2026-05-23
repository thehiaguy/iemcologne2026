"""Monte-Carlo simulation of the whole IEM Cologne 2026 Major.

Chains Stage 1 -> Stage 2 (+8 seeds) -> Stage 3 (+8 seeds) -> playoffs, many times,
and aggregates each team's probability of clearing every stage and winning the title.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config, series
from .playoffs import simulate_playoffs
from .swiss import simulate_swiss
from .teams import DISPLAY_TO_VRS, FIELD, stage_teams

# milestone columns (monotonic: each implies the previous for a given team)
METRICS = ["adv_s1", "adv_s2", "adv_s3", "semifinal", "final", "champion"]
M = {m: i for i, m in enumerate(METRICS)}


# --------------------------------------------------------------------------- #
# Per-team strength inputs
# --------------------------------------------------------------------------- #
def build_pmap(model, ctx, as_of_date) -> dict:
    """Symmetric per-map win-probability table keyed by (display_a, display_b)."""
    names = [t.name for t in FIELD]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    feats = [
        ctx.compute_features(DISPLAY_TO_VRS[a], DISPLAY_TO_VRS[b], as_of_date)
        for a, b in pairs
    ]
    df = pd.DataFrame(feats)
    enc_ab = model.predict_proba(df)
    feats_ba = [
        ctx.compute_features(DISPLAY_TO_VRS[b], DISPLAY_TO_VRS[a], as_of_date)
        for a, b in pairs
    ]
    enc_ba = model.predict_proba(pd.DataFrame(feats_ba))

    pmap: dict[tuple[str, str], float] = {}
    for (a, b), pab, pba in zip(pairs, enc_ab, enc_ba):
        pe = 0.5 * (pab + (1.0 - pba))          # symmetrise the two orientations
        p = series.map_prob_from_model(pe)
        pmap[(a, b)] = p
        pmap[(b, a)] = 1.0 - p
    return pmap


def _seed(invitees, advancers, vrs):
    """Seed invitees 1..k then advancers k+1..n, each ordered by VRS points."""
    inv = sorted(invitees, key=lambda t: -vrs[t])
    adv = sorted(advancers, key=lambda t: -vrs[t])
    return inv + adv


# --------------------------------------------------------------------------- #
# One realisation of the event
# --------------------------------------------------------------------------- #
def run_once(rng, pmap, vrs) -> list[tuple[str, str]]:
    """Return list of (team, milestone) achieved this run."""
    def pm(a, b):
        return pmap[(a, b)]

    achieved: list[tuple[str, str]] = []

    s1 = _seed(stage_teams(1), [], vrs)
    adv1, _ = simulate_swiss(s1, pm, rng, all_bo3=config.STAGE1.all_bo3)
    for t in adv1:
        achieved.append((t, "adv_s1"))

    s2 = _seed(stage_teams(2), adv1, vrs)
    adv2, _ = simulate_swiss(s2, pm, rng, all_bo3=config.STAGE2.all_bo3)
    for t in adv2:
        achieved.append((t, "adv_s2"))

    s3 = _seed(stage_teams(3), adv2, vrs)
    adv3, _ = simulate_swiss(s3, pm, rng, all_bo3=config.STAGE3.all_bo3)
    for t in adv3:
        achieved.append((t, "adv_s3"))

    res = simulate_playoffs(adv3, pm, rng)      # adv3 already in finishing/seed order
    for team, far in res["reached"].items():
        if far in ("SF", "Final", "Champion"):
            achieved.append((team, "semifinal"))
        if far in ("Final", "Champion"):
            achieved.append((team, "final"))
        if far == "Champion":
            achieved.append((team, "champion"))
    return achieved


def _run_chunk(n, seed, pmap, vrs, names):
    rng = np.random.default_rng(seed)
    idx = {nm: i for i, nm in enumerate(names)}
    counts = np.zeros((len(names), len(METRICS)), dtype=np.int64)
    for _ in range(n):
        for team, metric in run_once(rng, pmap, vrs):
            counts[idx[team], M[metric]] += 1
    return counts


# --------------------------------------------------------------------------- #
# Monte-Carlo driver
# --------------------------------------------------------------------------- #
def run_monte_carlo(pmap, vrs, n_sims=config.N_SIMS, n_jobs=config.N_JOBS,
                    seed=config.RANDOM_SEED) -> pd.DataFrame:
    names = [t.name for t in FIELD]
    n_workers = max(1, (n_jobs if n_jobs > 0 else 8))
    per = [n_sims // n_workers] * n_workers
    for i in range(n_sims - sum(per)):
        per[i] += 1

    chunks = Parallel(n_jobs=n_jobs)(
        delayed(_run_chunk)(per[w], seed + 1 + w, pmap, vrs, names)
        for w in range(n_workers) if per[w] > 0
    )
    counts = np.sum(chunks, axis=0)
    N = n_sims

    stage_of = {t.name: t.stage for t in FIELD}
    z = 1.959963985
    rows = []
    for i, name in enumerate(names):
        row = {"team": name, "stage": stage_of[name], "vrs": vrs[name]}
        for metric in METRICS:
            p = counts[i, M[metric]] / N
            se = np.sqrt(max(p * (1 - p), 0.0) / N)
            row[metric] = p
            row[f"{metric}_lo"] = max(0.0, p - z * se)
            row[f"{metric}_hi"] = min(1.0, p + z * se)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("team")

    # Blank milestones a team cannot contest (it enters at a later stage).
    for name in names:
        st = stage_of[name]
        if st >= 2:
            for m in ("adv_s1",):
                for suffix in ("", "_lo", "_hi"):
                    df.loc[name, m + suffix] = np.nan
        if st >= 3:
            for m in ("adv_s1", "adv_s2"):
                for suffix in ("", "_lo", "_hi"):
                    df.loc[name, m + suffix] = np.nan
    return df.sort_values("champion", ascending=False)
