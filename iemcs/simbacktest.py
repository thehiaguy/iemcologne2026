"""End-to-end simulator backtest on real CS2 Majors.

For each past Major we know the exact 8-team playoff bracket and the actual result.
We train the match model on data **strictly before** the playoffs, build the team
state as of that date, then Monte-Carlo the *real* bracket and compare the predicted
reach-SF / reach-final / champion probabilities to what actually happened. This tests
the whole stack (ratings → ensemble → series math → bracket sim) on real outcomes,
not just individual match predictions.

Only the playoff stage is replayed — it's the format common to every recent Major and
its results are unambiguous. Both backtested Majors were won by Vitality, so champion
accuracy is partly a given; the discriminating signal is the reach-SF / reach-final
Brier (did the model rank the right four / two teams?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dataset, series
from .config import RECENCY_HALF_LIFE_DAYS
from .model import MatchModel
from .teams import normalize_name

# QF pairs are ordered so that (qf0, qf1) feed semifinal 1 and (qf2, qf3) feed
# semifinal 2 — i.e. the real bracket structure of each event.
MAJORS = {
    "BLAST Austin 2025": {
        "cutoff": "2025-06-19",
        "qf": [("Vitality", "Natus Vincere"), ("MOUZ", "Spirit"),
               ("The MongolZ", "FaZe"), ("paiN", "FURIA")],
        "champion": "Vitality",
        "runner_up": "The MongolZ",
        "semifinal": ["MOUZ", "paiN"],          # lost in the semifinals
    },
    "StarLadder Budapest 2025": {
        "cutoff": "2025-12-11",
        "qf": [("Vitality", "The MongolZ"), ("Spirit", "Falcons"),
               ("FaZe", "MOUZ"), ("Natus Vincere", "FURIA")],
        "champion": "Vitality",
        "runner_up": "FaZe",
        "semifinal": ["Spirit", "Natus Vincere"],
    },
}


def _pmap_for(model, ctx, teams, date) -> dict:
    """Symmetric per-map win-prob table for `teams` (VRS names) as of `date`."""
    pm = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            pe = 0.5 * (
                model.predict_one(ctx.compute_features(a, b, date))
                + (1.0 - model.predict_one(ctx.compute_features(b, a, date)))
            )
            p = series.map_prob_from_model(pe)
            pm[(a, b)] = p
            pm[(b, a)] = 1.0 - p
    return pm


def _sim_once(qf, pmap, rng) -> dict:
    """Return level per team: 0=lost QF, 1=reached SF, 2=reached final, 3=champion."""
    lvl = {t: 0 for pair in qf for t in pair}
    sf = []
    for a, b in qf:
        w = a if rng.random() < series.series_winprob(pmap[(a, b)], 3) else b
        lvl[w] = 1
        sf.append(w)
    fin = []
    for a, b in ((sf[0], sf[1]), (sf[2], sf[3])):
        w = a if rng.random() < series.series_winprob(pmap[(a, b)], 3) else b
        lvl[w] = 2
        fin.append(w)
    champ = fin[0] if rng.random() < series.series_winprob(pmap[(fin[0], fin[1])], 5) else fin[1]
    lvl[champ] = 3
    return lvl


def _actual_level(major) -> dict:
    a = {normalize_name(major["champion"]): 3, normalize_name(major["runner_up"]): 2}
    for t in major["semifinal"]:
        a[normalize_name(t)] = 1
    for pair in major["qf"]:
        for t in pair:
            a.setdefault(normalize_name(t), 0)
    return a


def run_major(matches, standings, region, major, recency_half_life,
              n_sims=20_000, seed=42):
    cut = pd.Timestamp(major["cutoff"])
    frame, ctx = dataset.build(matches[matches["date"] < cut], standings, region)
    model = MatchModel(recency_half_life=recency_half_life).fit(frame, verbose=False)

    qf = [(normalize_name(a), normalize_name(b)) for a, b in major["qf"]]
    teams = [t for pair in qf for t in pair]
    pmap = _pmap_for(model, ctx, teams, cut)

    rng = np.random.default_rng(seed)
    counts = {t: np.zeros(3) for t in teams}     # reach SF / final / champion
    for _ in range(n_sims):
        for t, lvl in _sim_once(qf, pmap, rng).items():
            if lvl >= 1:
                counts[t][0] += 1
            if lvl >= 2:
                counts[t][1] += 1
            if lvl >= 3:
                counts[t][2] += 1
    actual = _actual_level(major)
    return pd.DataFrame([
        {"team": t, "P_reach_SF": counts[t][0] / n_sims,
         "P_reach_final": counts[t][1] / n_sims, "P_champion": counts[t][2] / n_sims,
         "actual": {0: "QF", 1: "SF", 2: "Final", 3: "Champion"}[actual[t]]}
        for t in teams
    ])


def run(matches, standings, region, recency_half_life=RECENCY_HALF_LIFE_DAYS,
        n_sims=20_000, seed=42) -> pd.DataFrame:
    out = []
    for name, major in MAJORS.items():
        df = run_major(matches, standings, region, major, recency_half_life,
                       n_sims, seed)
        df.insert(0, "major", name)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def metrics(results: pd.DataFrame) -> dict:
    """Brier scores vs actual outcomes + the actual champion's mean predicted prob."""
    r = results
    champ = (r["actual"] == "Champion").astype(float)
    finalist = r["actual"].isin(["Champion", "Final"]).astype(float)
    semi = r["actual"].isin(["Champion", "Final", "SF"]).astype(float)
    return {
        "brier_champion": float(((r["P_champion"] - champ) ** 2).mean()),
        "brier_reach_final": float(((r["P_reach_final"] - finalist) ** 2).mean()),
        "brier_reach_SF": float(((r["P_reach_SF"] - semi) ** 2).mean()),
        "actual_champion_mean_prob": float(r.loc[champ == 1, "P_champion"].mean()),
        "n_teams": int(len(r)),
    }


if __name__ == "__main__":          # run with: python -m iemcs.simbacktest
    import pandas as pd

    from . import vrs

    pd.set_option("display.width", 200)
    _m, _st, _reg = vrs.parse_matches(), vrs.parse_standings(), vrs.build_region_map()
    _res = run(_m, _st, _reg)
    print(_res.round(3).to_string(index=False))
    print("\nmetrics:", {k: round(v, 4) for k, v in metrics(_res).items()})
