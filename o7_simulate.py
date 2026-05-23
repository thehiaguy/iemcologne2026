"""
[7/7] simulate.py
-----------------
Monte Carlo simulation of IEM Cologne 2026.

Loads trained model artifacts (run o6_train.py first) and runs:
  Stage 1  : 16 teams (Swiss BO1, 3W/3L) → top 8 advance
  Stage 2  : 8 Stage2 seeds + 8 Stage1 qualifiers (Swiss BO1) → top 8 advance
  Stage 3  : 8 Stage3 seeds + 8 Stage2 qualifiers (Swiss BO3) → top 8 advance
  Playoffs :  8 teams → seeded single-elim BO3 bracket

Prediction uses an ensemble of:
  - PyTorch residual MLP
  - XGBoost booster
  (blend weight loaded from training artifacts)

Usage
-----
  python o7_simulate.py --tournament --n 50000
  python o7_simulate.py --match "Team Vitality" "Team Spirit" --n 10000
  python o7_simulate.py --fetch-stats          # force-refresh team stats
  python o7_simulate.py --list-teams
"""

import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from o1_config import (
    DATA_DIR, MODEL_DIR,
    ALL_TEAMS, STAGE3_TEAMS, STAGE2_TEAMS, STAGE1_TEAMS,
    TEAM_IDS, ROLLING_WINDOWS, RANDOM_SEED,
)
from o3_features import EloRatings, rolling_team_features, h2h_features
from o4_neural_net import MatchPredictor, pytorch_predict
from o2_scraper import HLTVScraper
from o5_collect import _result_to_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STATS_CACHE = DATA_DIR / "current_team_stats.pkl"


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_artifacts():
    required = [
        "pytorch_model.pt", "xgb_model.json", "scaler.pkl",
        "feature_cols.pkl", "elo.pkl", "blend_weight.pkl", "input_dim.pkl",
    ]
    for f in required:
        if not (MODEL_DIR / f).exists():
            raise FileNotFoundError(
                f"{MODEL_DIR / f} not found — run o6_train.py first."
            )

    from xgboost import XGBClassifier
    input_dim = pickle.load(open(MODEL_DIR / "input_dim.pkl",   "rb"))
    pt_model  = MatchPredictor(input_dim)
    pt_model.load_state_dict(
        torch.load(MODEL_DIR / "pytorch_model.pt", map_location="cpu")
    )
    pt_model.eval()

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / "xgb_model.json"))

    scaler       = pickle.load(open(MODEL_DIR / "scaler.pkl",       "rb"))
    feature_cols = pickle.load(open(MODEL_DIR / "feature_cols.pkl", "rb"))
    elo          = pickle.load(open(MODEL_DIR / "elo.pkl",          "rb"))
    blend_w      = pickle.load(open(MODEL_DIR / "blend_weight.pkl", "rb"))

    return pt_model, xgb_model, scaler, feature_cols, elo, blend_w


# ---------------------------------------------------------------------------
# Fetch current team stats
# ---------------------------------------------------------------------------

def fetch_all_team_stats(
    force: bool = False,
    teams: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Scrape the last 20 matches for each team in `teams` (defaults to ALL_TEAMS)
    and cache incrementally to disk.  Already-cached teams are skipped unless
    force=True.  Returns dict: team_name → raw_df.
    """
    teams_to_fetch = teams if teams is not None else ALL_TEAMS

    cached: Dict[str, pd.DataFrame] = {}
    if STATS_CACHE.exists() and not force:
        cached = pickle.load(open(STATS_CACHE, "rb"))
        missing = [t for t in teams_to_fetch if t not in cached]
        if not missing:
            log.info("All requested teams found in cache (%s)", STATS_CACHE)
            return cached
        log.info("%d team(s) not yet cached — fetching now", len(missing))
    else:
        missing = list(teams_to_fetch)

    DATA_DIR.mkdir(exist_ok=True)
    team_match_dfs: Dict[str, pd.DataFrame] = dict(cached)

    with HLTVScraper() as scraper:
        for team in missing:
            tid = TEAM_IDS.get(team)
            if not tid:
                log.warning("No ID for %s — skipping", team)
                continue
            log.info("Fetching stats for %s (id=%d)...", team, tid)
            results = scraper.get_team_results(tid, limit=20)
            if not results:
                continue

            rows = []
            for res in results:
                ms_ids = scraper.get_mapstats_ids(res["matchUrl"]) if res.get("matchUrl") else []
                player_rows = []
                for ms_id, slug in ms_ids:
                    player_rows.extend(scraper.get_map_stats(ms_id, slug))

                rec = _result_to_record(res, player_rows or None)
                if rec:
                    rec["match_date"] = res["matchDate"]
                    rows.append(rec)

            if rows:
                df = pd.DataFrame(rows)
                df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
                team_match_dfs[team] = df

    pickle.dump(team_match_dfs, open(STATS_CACHE, "wb"))
    log.info("Cached stats for %d teams", len(team_match_dfs))
    return team_match_dfs


# ---------------------------------------------------------------------------
# Feature vector assembly
# ---------------------------------------------------------------------------

def build_feature_vector(
    team_a: str,
    team_b: str,
    team_dfs: Dict[str, pd.DataFrame],
    elo_obj:  EloRatings,
    feature_cols: List[str],
) -> np.ndarray:
    """Assemble a single feature row matching the training schema."""
    feat: Dict[str, float] = {}

    df_a = team_dfs.get(team_a, pd.DataFrame())
    df_b = team_dfs.get(team_b, pd.DataFrame())
    combined = pd.concat([df_a, df_b], ignore_index=True)
    if not combined.empty and "match_date" in combined.columns:
        combined = combined.sort_values("match_date")

    horizon = pd.Timestamp("2099-01-01")

    feat1: Dict[str, float] = {}
    feat2: Dict[str, float] = {}
    if not combined.empty:
        for w in ROLLING_WINDOWS:
            feat1.update(rolling_team_features(combined, team_a, horizon, w))
            feat2.update(rolling_team_features(combined, team_b, horizon, w))

    for k, v in feat1.items():
        feat[f"t1_{k}"] = v or 0.0
    for k, v in feat2.items():
        feat[f"t2_{k}"] = v or 0.0

    feat["h2h_win_rate"]   = 0.5
    feat["h2h_total"]      = 0.0
    feat["h2h_last5_wins"] = 2.0

    feat["t1_elo"]   = float(elo_obj.ratings.get(team_a, 1000))
    feat["t2_elo"]   = float(elo_obj.ratings.get(team_b, 1000))
    feat["elo_diff"] = feat["t1_elo"] - feat["t2_elo"]

    for key in ("t1_world_ranking", "t2_world_ranking",
                "t1_valve_ranking", "t2_valve_ranking"):
        feat[key] = 999.0
    feat["t1_weeks_top30"] = 0.0
    feat["t2_weeks_top30"] = 0.0
    feat["t1_avg_age"]     = 25.0
    feat["t2_avg_age"]     = 25.0
    feat["ranking_diff"]   = 0.0

    for w in ROLLING_WINDOWS:
        for m in ["avg_rating", "avg_adr", "avg_kast", "avg_opening_kd",
                  "avg_multi_kills", "avg_clutches", "win_rate"]:
            a = feat1.get(f"w{w}_{m}") or 0.0
            b = feat2.get(f"w{w}_{m}") or 0.0
            feat[f"diff_{m}_{w}"] = a - b

    row = np.array([feat.get(col, 0.0) for col in feature_cols], dtype=np.float32)
    return np.nan_to_num(row).reshape(1, -1)


# ---------------------------------------------------------------------------
# Win probability — single query (used by h2h deep dive)
# ---------------------------------------------------------------------------

def win_prob(
    team_a: str, team_b: str,
    team_dfs:     Dict[str, pd.DataFrame],
    elo_obj:      EloRatings,
    pt_model:     MatchPredictor,
    xgb_model,
    scaler,
    feature_cols: List[str],
    blend_w:      float,
) -> float:
    """Return P(team_a beats team_b) in [0, 1]."""
    X_raw = build_feature_vector(team_a, team_b, team_dfs, elo_obj, feature_cols)
    X_s   = scaler.transform(np.nan_to_num(X_raw))

    pt_model.eval()
    with torch.no_grad():
        pt_prob = torch.sigmoid(pt_model(torch.FloatTensor(X_s))).item()

    xgb_prob = float(xgb_model.predict_proba(X_s)[0, 1])
    return blend_w * pt_prob + (1 - blend_w) * xgb_prob


# ---------------------------------------------------------------------------
# Pre-compute all pairwise probabilities in one batched forward pass
# ---------------------------------------------------------------------------

def precompute_win_probs(
    teams:        List[str],
    team_dfs:     Dict[str, pd.DataFrame],
    elo_obj:      EloRatings,
    pt_model:     MatchPredictor,
    xgb_model,
    scaler,
    feature_cols: List[str],
    blend_w:      float,
) -> Dict[Tuple[str, str], float]:
    """
    Build feature vectors for every ordered pair (a, b) where a != b,
    run them through the model in a single batched forward pass (GPU if
    available), and return a {(a, b): prob} lookup table.
    """
    pairs  = [(a, b) for a in teams for b in teams if a != b]
    X_list = [build_feature_vector(a, b, team_dfs, elo_obj, feature_cols)
              for a, b in pairs]
    X      = np.vstack(X_list)
    X_s    = scaler.transform(np.nan_to_num(X)).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info("Running batch inference on GPU (%s)", torch.cuda.get_device_name(0))
    else:
        log.info("Running batch inference on CPU")

    pt_model.eval().to(device)
    with torch.no_grad():
        pt_probs = torch.sigmoid(
            pt_model(torch.from_numpy(X_s).to(device))
        ).cpu().numpy().flatten()
    pt_model.to("cpu")

    xgb_probs = xgb_model.predict_proba(X_s)[:, 1]
    blended   = blend_w * pt_probs + (1 - blend_w) * xgb_probs

    return {pair: float(p) for pair, p in zip(pairs, blended)}


# ---------------------------------------------------------------------------
# Swiss stage simulator
# ---------------------------------------------------------------------------

def simulate_swiss(
    teams: List[str],
    target_wins: int,
    target_losses: int,
    wp_fn,
    rng: np.random.Generator,
    top_n: Optional[int] = None,
    best_of: int = 1,
) -> Tuple[List[str], List[str]]:
    """
    Run a Swiss stage until all teams reach target_wins or target_losses.
    Pairing: same W-L record, adjacent seeds.
    best_of=1 → single map, best_of=3 → BO3 series.
    Returns (advanced, eliminated).
    """
    wins   = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}

    while True:
        active = [t for t in teams
                  if wins[t] < target_wins and losses[t] < target_losses]
        if len(active) < 2:
            break
        active.sort(key=lambda t: wins[t] - losses[t], reverse=True)
        pairs = [(active[i], active[i + 1]) for i in range(0, len(active) - 1, 2)]
        for a, b in pairs:
            if best_of == 3:
                winner = simulate_bo3(a, b, wp_fn, rng)
                if winner == a:
                    wins[a] += 1;  losses[b] += 1
                else:
                    wins[b] += 1;  losses[a] += 1
            else:
                if rng.random() < wp_fn(a, b):
                    wins[a] += 1;  losses[b] += 1
                else:
                    wins[b] += 1;  losses[a] += 1

    if top_n is not None:
        ranked = sorted(teams, key=lambda t: wins[t], reverse=True)
        return ranked[:top_n], ranked[top_n:]
    return (
        [t for t in teams if wins[t] >= target_wins],
        [t for t in teams if losses[t] >= target_losses],
    )


# ---------------------------------------------------------------------------
# BO3 match
# ---------------------------------------------------------------------------

def simulate_bo3(team_a: str, team_b: str, wp_fn, rng: np.random.Generator) -> str:
    p = wp_fn(team_a, team_b)
    wa, wb = 0, 0
    while wa < 2 and wb < 2:
        if rng.random() < p:
            wa += 1
        else:
            wb += 1
    return team_a if wa == 2 else team_b


# ---------------------------------------------------------------------------
# Single-elimination bracket (BO3)
# ---------------------------------------------------------------------------

def simulate_bracket(
    seeds: List[str], wp_fn, rng: np.random.Generator
) -> Dict[str, str]:
    """Seeded bracket: 1v8, 2v7, 3v6, 4v5."""
    n = len(seeds)
    if n == 8:
        pairs = [
            (seeds[0], seeds[7]), (seeds[1], seeds[6]),
            (seeds[2], seeds[5]), (seeds[3], seeds[4]),
        ]
    else:
        pairs = [(seeds[i], seeds[i + 1]) for i in range(0, n, 2)]

    finish: Dict[str, str] = {}
    bracket = []
    for a, b in pairs:
        w = simulate_bo3(a, b, wp_fn, rng)
        finish[b if w == a else a] = "top8"
        bracket.append(w)

    sf_pairs = [(bracket[0], bracket[3]), (bracket[1], bracket[2])] \
               if len(bracket) >= 4 else []
    finalists = []
    for a, b in sf_pairs:
        w = simulate_bo3(a, b, wp_fn, rng)
        finish[b if w == a else a] = "top4"
        finalists.append(w)

    if len(finalists) == 2:
        champion = simulate_bo3(finalists[0], finalists[1], wp_fn, rng)
        runner   = finalists[1] if champion == finalists[0] else finalists[0]
        finish[champion] = "champion"
        finish[runner]   = "runner_up"

    return finish


# ---------------------------------------------------------------------------
# Full Monte Carlo tournament
# ---------------------------------------------------------------------------

def simulate_tournament(
    team_dfs:     Dict[str, pd.DataFrame],
    elo_obj:      EloRatings,
    pt_model:     MatchPredictor,
    xgb_model,
    scaler,
    feature_cols: List[str],
    blend_w:      float,
    n_sims:       int = 10_000,
) -> Dict[str, Dict[str, float]]:
    rng    = np.random.default_rng(RANDOM_SEED)
    counts: Dict[str, Dict[str, int]] = {t: defaultdict(int) for t in ALL_TEAMS}

    log.info("Pre-computing pairwise win probabilities for %d teams...", len(ALL_TEAMS))
    prob_cache = precompute_win_probs(
        ALL_TEAMS, team_dfs, elo_obj, pt_model, xgb_model,
        scaler, feature_cols, blend_w,
    )
    log.info("Cached %d matchup probabilities — starting simulations", len(prob_cache))

    def wp_fn(a: str, b: str) -> float:
        return prob_cache.get((a, b), 0.5)

    for sim_i in range(n_sims):
        if sim_i % 5000 == 0:
            log.info("  Simulation %d / %d", sim_i, n_sims)

        # Stage 1 — 16 teams, Swiss BO1, top 8 advance
        s1_adv, s1_elim = simulate_swiss(
            list(STAGE1_TEAMS), 3, 3, wp_fn, rng, top_n=8, best_of=1
        )
        for t in s1_elim:
            counts[t]["elim_stage1"] += 1

        # Stage 2 — 8 Stage2 seeds + 8 Stage1 qualifiers, Swiss BO1, top 8 advance
        s2_pool = list(STAGE2_TEAMS) + s1_adv
        s2_adv, s2_elim = simulate_swiss(s2_pool, 3, 3, wp_fn, rng, top_n=8, best_of=1)
        for t in s2_elim:
            counts[t]["elim_stage2"] += 1

        # Stage 3 — 8 Stage3 seeds + 8 Stage2 qualifiers, Swiss BO3, top 8 advance
        s3_pool = list(STAGE3_TEAMS) + s2_adv
        s3_adv, s3_elim = simulate_swiss(s3_pool, 3, 3, wp_fn, rng, top_n=8, best_of=3)
        for t in s3_elim:
            counts[t]["elim_stage3"] += 1

        # Playoffs — 8 teams, seeded single-elim BO3
        finish = simulate_bracket(s3_adv[:8], wp_fn, rng)
        for team, result in finish.items():
            counts[team][result] += 1
            if result in ("champion", "runner_up", "top4"):
                counts[team]["reached_top4"] += 1
            if result in ("champion", "runner_up", "top4", "top8"):
                counts[team]["reached_top8"] += 1

    summary: Dict[str, Dict[str, float]] = {}
    for team in ALL_TEAMS:
        c = counts[team]
        summary[team] = {
            "champion_pct":    round(100 * c["champion"]       / n_sims, 2),
            "runner_up_pct":   round(100 * c["runner_up"]      / n_sims, 2),
            "top4_pct":        round(100 * c["reached_top4"]   / n_sims, 2),
            "top8_pct":        round(100 * c["reached_top8"]   / n_sims, 2),
            "elim_stage3_pct": round(100 * c["elim_stage3"]    / n_sims, 2),
            "elim_stage2_pct": round(100 * c["elim_stage2"]    / n_sims, 2),
            "elim_stage1_pct": round(100 * c["elim_stage1"]    / n_sims, 2),
        }
    return summary


# ---------------------------------------------------------------------------
# H2H deep dive
# ---------------------------------------------------------------------------

def h2h_deep_dive(
    team_a: str, team_b: str,
    team_dfs:     Dict[str, pd.DataFrame],
    elo_obj:      EloRatings,
    pt_model:     MatchPredictor,
    xgb_model,
    scaler,
    feature_cols: List[str],
    blend_w:      float,
    n:            int = 10_000,
) -> dict:
    rng = np.random.default_rng()
    p   = win_prob(
        team_a, team_b, team_dfs, elo_obj,
        pt_model, xgb_model, scaler, feature_cols, blend_w
    )

    bo1_wins = sum(1 for _ in range(n) if rng.random() < p)

    def run_bo3_plain() -> bool:
        wa, wb = 0, 0
        while wa < 2 and wb < 2:
            if rng.random() < p:
                wa += 1
            else:
                wb += 1
        return wa == 2

    bo3_wins = sum(1 for _ in range(n) if run_bo3_plain())
    return {
        "team_a":         team_a,
        "team_b":         team_b,
        "map_win_prob_a": round(p * 100, 2),
        "bo1_win_pct_a":  round(bo1_wins / n * 100, 2),
        "bo3_win_pct_a":  round(bo3_wins / n * 100, 2),
        "bo1_win_pct_b":  round((n - bo1_wins) / n * 100, 2),
        "bo3_win_pct_b":  round((n - bo3_wins) / n * 100, 2),
        "n_simulations":  n,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_SEP = "=" * 78


def print_tournament_results(summary: Dict[str, Dict[str, float]], n: int):
    print(f"\n{_SEP}")
    print(f"  IEM COLOGNE 2026  --  Ensemble MC ({n:,} simulations)")
    print(_SEP)
    hdr = (
        f"{'Team':<22} {'Champion':>9} {'2nd':>6} {'Top4':>6} {'Top8':>6} "
        f"{'ES3':>6} {'ES2':>6} {'ES1':>6}"
    )
    print(hdr)
    print("-" * 78)
    for team, s in sorted(summary.items(), key=lambda x: x[1]["champion_pct"], reverse=True):
        print(
            f"{team:<22} {s['champion_pct']:>8.1f}% {s['runner_up_pct']:>5.1f}% "
            f"{s['top4_pct']:>5.1f}% {s['top8_pct']:>5.1f}% "
            f"{s['elim_stage3_pct']:>5.1f}% {s['elim_stage2_pct']:>5.1f}% "
            f"{s['elim_stage1_pct']:>5.1f}%"
        )
    print(_SEP)
    print("  ES3=elim at Stage 3, ES2=Stage 2, ES1=Stage 1\n")


def print_h2h(r: dict):
    a, b = r["team_a"], r["team_b"]
    print(f"\n{_SEP}")
    print(f"  {a}  vs  {b}   ({r['n_simulations']:,} simulations)")
    print(_SEP)
    print(f"  Map win probability  {a}: {r['map_win_prob_a']:>5.1f}%"
          f"   {b}: {100 - r['map_win_prob_a']:>5.1f}%")
    print(f"  BO1 win probability  {a}: {r['bo1_win_pct_a']:>5.1f}%"
          f"   {b}: {r['bo1_win_pct_b']:>5.1f}%")
    print(f"  BO3 win probability  {a}: {r['bo3_win_pct_a']:>5.1f}%"
          f"   {b}: {r['bo3_win_pct_b']:>5.1f}%")
    print(_SEP + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IEM Cologne 2026 CS2 Simulator")
    parser.add_argument("--match",       nargs=2, metavar=("TEAM_A", "TEAM_B"))
    parser.add_argument("--tournament",  action="store_true")
    parser.add_argument("--fetch-stats", action="store_true",
                        help="Force-refresh team stats (re-opens Chrome)")
    parser.add_argument("--list-teams",  action="store_true")
    parser.add_argument("--n",           type=int, default=10_000,
                        help="Number of Monte Carlo simulations (default 10000)")
    args = parser.parse_args()

    if args.list_teams:
        print("\nStage 3 seeds (8):", ", ".join(STAGE3_TEAMS))
        print("Stage 2 seeds (8):", ", ".join(STAGE2_TEAMS))
        print("Stage 1 seeds (16):", ", ".join(STAGE1_TEAMS))
        return

    log.info("Loading model artifacts...")
    pt_model, xgb_model, scaler, feature_cols, elo, blend_w = load_artifacts()
    log.info("Ensemble blend: %.0f%% PyTorch + %.0f%% XGBoost",
             blend_w * 100, (1 - blend_w) * 100)

    log.info("Loading team stats (use --fetch-stats to refresh)...")
    teams_needed = list(args.match) if args.match else None
    team_dfs = fetch_all_team_stats(force=args.fetch_stats, teams=teams_needed)

    if args.match:
        team_a, team_b = args.match
        unknown = [t for t in (team_a, team_b) if t not in TEAM_IDS]
        if unknown:
            log.error("Unknown team(s): %s  --  use --list-teams", unknown)
            return
        result = h2h_deep_dive(
            team_a, team_b, team_dfs, elo,
            pt_model, xgb_model, scaler, feature_cols, blend_w, n=args.n,
        )
        print_h2h(result)

    elif args.tournament:
        log.info("Running tournament Monte Carlo (%d simulations)...", args.n)
        summary = simulate_tournament(
            team_dfs, elo, pt_model, xgb_model,
            scaler, feature_cols, blend_w, n_sims=args.n,
        )
        print_tournament_results(summary, args.n)
        DATA_DIR.mkdir(exist_ok=True)
        out = DATA_DIR / "tournament_odds.csv"
        pd.DataFrame(summary).T.reset_index().rename(
            columns={"index": "team"}
        ).to_csv(out, index=False)
        log.info("Odds saved to %s", out)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
