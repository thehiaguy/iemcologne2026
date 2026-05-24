"""
[8/8] pickem.py
---------------
Pick'Em optimizer for the IEM Cologne 2026 Stage 1 (Opening Stage) Swiss.

Maximises P(>=5 correct) over all candidate pick sets:
  - 2 teams to go 3-0   (correct ONLY if the team finishes exactly 3-0)
  - 6 teams to ADVANCE  (correct if the team finishes 3-0 / 3-1 / 3-2)
  - 2 teams to go 0-3   (correct ONLY if the team finishes exactly 0-3)

The three slots are mutually exclusive (10 picks, 10 distinct teams).

Two simulation paths are selected automatically based on --n-sims:
  Standard  (<= CHUNK_THRESHOLD): stores full boolean arrays in RAM.
  Chunked   (>  CHUNK_THRESHOLD): constant RAM regardless of n_sims — processes
            sims in parallel batches and accumulates per-combo score histograms
            instead of keeping all results in memory simultaneously.

Usage
-----
  python o8_pickem.py
  python o8_pickem.py --n-sims 1000000
  python o8_pickem.py --n-sims 1000000 --n-workers 8 --chunk-size 25000
"""

import argparse
import itertools
import logging
import multiprocessing as mp
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch

from o1_config import DATA_DIR, MODEL_DIR, STAGE1_TEAMS, RANDOM_SEED
from o4_neural_net import MatchPredictor
from o7_simulate import precompute_win_probs, fetch_all_team_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from numba import njit as _njit, prange as _prange
    _NUMBA = True
except ImportError:
    _NUMBA = False
    log.warning("numba not found — using pure-Python simulation (install with: pip install numba)")

CHUNK_SIZE      = 10_000   # sims per worker per round in chunked mode
CHUNK_THRESHOLD = 200_000  # auto-switch to chunked path above this
_COMBO_BATCH    = 500      # combos to evaluate per numpy call in _accumulate_scores


# ---------------------------------------------------------------------------
# Numba-compiled Swiss simulation (10-50x faster than pure Python)
# cache=True writes the compiled code to __pycache__ so worker processes
# load it instantly instead of recompiling on every spawn.
# ---------------------------------------------------------------------------

if _NUMBA:
    @_njit(cache=True)
    def _swiss_numba(wp_matrix, n, target=3):
        wins    = np.zeros(n, dtype=np.int32)
        losses  = np.zeros(n, dtype=np.int32)
        active  = np.zeros(n, dtype=np.int32)
        records = np.zeros(n, dtype=np.int32)
        while True:
            m = 0
            for i in range(n):
                if wins[i] < target and losses[i] < target:
                    active[m]  = i
                    records[m] = wins[i] - losses[i]
                    m += 1
            if m < 2:
                break
            # insertion sort descending by record
            for p in range(1, m):
                kr = records[p]; ka = active[p]
                q  = p - 1
                while q >= 0 and records[q] < kr:
                    records[q + 1] = records[q]
                    active[q + 1]  = active[q]
                    q -= 1
                records[q + 1] = kr
                active[q + 1]  = ka
            for k in range(0, m - 1, 2):
                a = active[k]; b = active[k + 1]
                if np.random.random() < wp_matrix[a, b]:
                    wins[a] += 1;   losses[b] += 1
                else:
                    wins[b] += 1;   losses[a] += 1
        return wins, losses

    @_njit(cache=True)
    def _chunk_numba(wp_matrix, n, chunk_size, seed):
        np.random.seed(seed)
        tz  = np.zeros((chunk_size, n), dtype=np.bool_)
        adv = np.zeros((chunk_size, n), dtype=np.bool_)
        oz  = np.zeros((chunk_size, n), dtype=np.bool_)
        for s in range(chunk_size):
            wins, losses = _swiss_numba(wp_matrix, n)
            for i in range(n):
                w = wins[i]; l = losses[i]
                if w == 3 and l == 0:
                    tz[s, i] = True; adv[s, i] = True
                elif w == 3:
                    adv[s, i] = True
                elif l == 3 and w == 0:
                    oz[s, i] = True
        return tz, adv, oz

    @_njit(cache=True, parallel=True)
    def _accumulate_numba(tz, adv, oz, i30_arr, iadv_arr, i03_arr, score_hists):
        """
        JIT-compiled, parallel accumulation of per-combo score histograms.

        Parallelises over combos (prange) — each combo owns a unique row in
        score_hists so there are no write conflicts between threads.
        No intermediate arrays: each sim/combo pair resolves to a single
        histogram bucket increment.
        """
        n_sims   = tz.shape[0]
        n_combos = i30_arr.shape[0]
        k30      = i30_arr.shape[1]
        kadv     = iadv_arr.shape[1]
        k03      = i03_arr.shape[1]
        for c in _prange(n_combos):
            for s in range(n_sims):
                score = np.int64(0)
                for k in range(k30):
                    if tz[s, i30_arr[c, k]]:
                        score += 1
                for k in range(kadv):
                    if adv[s, iadv_arr[c, k]]:
                        score += 1
                for k in range(k03):
                    if oz[s, i03_arr[c, k]]:
                        score += 1
                score_hists[c, score] += 1


# ---------------------------------------------------------------------------
# Swiss simulation that returns per-team records (Python fallback)
# ---------------------------------------------------------------------------

def _simulate_swiss_records(
    teams: List[str],
    wp_fn,
    rng: np.random.Generator,
    target: int = 3,
) -> Dict[str, Tuple[int, int]]:
    wins   = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}
    while True:
        active = [t for t in teams
                  if wins[t] < target and losses[t] < target]
        if len(active) < 2:
            break
        active.sort(key=lambda t: wins[t] - losses[t], reverse=True)
        for i in range(0, len(active) - 1, 2):
            a, b = active[i], active[i + 1]
            if rng.random() < wp_fn(a, b):
                wins[a] += 1;  losses[b] += 1
            else:
                wins[b] += 1;  losses[a] += 1
    return {t: (wins[t], losses[t]) for t in teams}


# ---------------------------------------------------------------------------
# Multiprocessing worker — runs a chunk of simulations in one process
# ---------------------------------------------------------------------------

def _run_chunk(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    teams, prob_cache, chunk_size, seed = args
    n = len(teams)

    if _NUMBA:
        wp_matrix = np.empty((n, n), dtype=np.float64)
        for i, a in enumerate(teams):
            for j, b in enumerate(teams):
                wp_matrix[i, j] = prob_cache.get((a, b), 0.5) if i != j else 0.5
        return _chunk_numba(wp_matrix, n, chunk_size, seed % (2 ** 31))

    # --- pure-Python fallback ---
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}

    def wp_fn(a: str, b: str) -> float:
        return prob_cache.get((a, b), 0.5)

    tz  = np.zeros((chunk_size, n), dtype=np.bool_)
    adv = np.zeros((chunk_size, n), dtype=np.bool_)
    oz  = np.zeros((chunk_size, n), dtype=np.bool_)

    for s in range(chunk_size):
        records = _simulate_swiss_records(teams, wp_fn, rng)
        for t, (w, l) in records.items():
            i = idx[t]
            if w == 3 and l == 0:
                tz[s, i] = adv[s, i] = True
            elif w == 3:
                adv[s, i] = True
            elif l == 3 and w == 0:
                oz[s, i] = True

    return tz, adv, oz


# ---------------------------------------------------------------------------
# Standard path: run all sims, keep full arrays in RAM
# ---------------------------------------------------------------------------

def run_simulations(
    teams:      List[str],
    prob_cache: Dict,
    n_sims:     int = 50_000,
    n_workers:  int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns boolean arrays of shape (n_sims, n_teams):
      three_zero_arr  — team went exactly 3-0
      advanced_arr    — team advanced (3-0 / 3-1 / 3-2)
      zero_three_arr  — team went exactly 0-3

    Simulations are split evenly across n_workers processes.
    Use for n_sims <= CHUNK_THRESHOLD to keep the full arrays available for
    the vectorised optimizer.  Above that threshold use run_simulations_chunked.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = min(n_workers, n_sims)

    base  = n_sims // n_workers
    extra = n_sims  % n_workers
    chunk_sizes = [base + (1 if i < extra else 0) for i in range(n_workers)]

    worker_args = [
        (teams, prob_cache, chunk_sizes[i], RANDOM_SEED + i)
        for i in range(n_workers)
    ]

    log.info("Running %d sims across %d workers...", n_sims, n_workers)
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_run_chunk, worker_args)
    log.info("All workers finished.")

    three_zero_arr = np.concatenate([r[0] for r in results], axis=0)
    advanced_arr   = np.concatenate([r[1] for r in results], axis=0)
    zero_three_arr = np.concatenate([r[2] for r in results], axis=0)

    return three_zero_arr, advanced_arr, zero_three_arr


# ---------------------------------------------------------------------------
# Chunked path: constant RAM regardless of n_sims
# ---------------------------------------------------------------------------

def _accumulate_scores(
    tz:         np.ndarray,
    adv:        np.ndarray,
    oz:         np.ndarray,
    i30_arr:    np.ndarray,
    iadv_arr:   np.ndarray,
    i03_arr:    np.ndarray,
    score_hists: np.ndarray,
) -> None:
    if _NUMBA:
        _accumulate_numba(tz, adv, oz, i30_arr, iadv_arr, i03_arr, score_hists)
        return

    # --- pure-Python/numpy fallback ---
    n_combos = len(i30_arr)
    tz8  = tz.view(np.uint8)
    adv8 = adv.view(np.uint8)
    oz8  = oz.view(np.uint8)
    for start in range(0, n_combos, _COMBO_BATCH):
        end = min(start + _COMBO_BATCH, n_combos)
        s = (tz8[:,  i30_arr[start:end]].sum(-1, dtype=np.uint8) +
             adv8[:, iadv_arr[start:end]].sum(-1, dtype=np.uint8) +
             oz8[:,  i03_arr[start:end]].sum(-1, dtype=np.uint8))
        for k in range(11):
            score_hists[start:end, k] += (s == k).sum(0)


def run_simulations_chunked(
    teams:      List[str],
    prob_cache: Dict,
    n_sims:     int,
    n_workers:  int  = None,
    chunk_size: int  = CHUNK_SIZE,
    converge:   bool = True,
    min_sims:   int  = 50_000,
    tol:        float = 0.001,
    patience:   int  = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
    """
    Memory-bounded simulation + combo pre-enumeration.

    Instead of storing (n_sims, n_teams) boolean arrays this function:
      1. Runs a small warm-up batch to estimate marginals for top-k filtering.
      2. Enumerates all candidate pick-set combinations (~22k) up front.
      3. Processes sims in parallel chunks, accumulating per-combo score
         histograms and marginal counts — discarding each chunk after use.

    With converge=True (default) the loop stops early once the optimal pick set
    and its P(>=5) estimate have been stable for `patience` consecutive rounds
    (each round = chunk_size * n_workers sims), provided at least `min_sims`
    have run.  Stability is defined as: same best combo AND |delta P(>=5)| < tol.

    Peak RAM = O(chunk_size * n_teams) per round + O(n_combos * 11) for hists.
    Both are constant regardless of n_sims.

    Returns:
      p30, padv, p03  — marginal outcome probabilities, shape (n_teams,)
      combos          — list of (i30_tuple, iadv_tuple, i03_tuple) index tuples
      score_hists     — int64 array of shape (n_combos, 11)
                        (score_hists[c].sum() == actual sims run, not n_sims if
                        early convergence triggered)
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n = len(teams)

    # --- warm-up to estimate marginals for candidate filtering ---
    warmup_n = min(10_000, max(1_000, n_sims // 50))
    log.info("Warm-up: %d sims to estimate marginals...", warmup_n)
    tz_w, adv_w, oz_w = _run_chunk((teams, prob_cache, warmup_n, RANDOM_SEED))

    # --- enumerate candidate combos using warm-up marginals ---
    top_k_30, top_k_03, top_k_adv = 8, 8, 10
    cand_30  = list(np.argsort(tz_w.mean(0))[::-1][:top_k_30])
    cand_03  = list(np.argsort(oz_w.mean(0))[::-1][:top_k_03])
    cand_adv = list(np.argsort(adv_w.mean(0))[::-1][:top_k_adv])

    combos = []
    for i30 in itertools.combinations(cand_30, 2):
        for i03 in itertools.combinations(cand_03, 2):
            pool_idx = [i for i in cand_adv if i not in set(i30) | set(i03)]
            if len(pool_idx) >= 6:
                for iadv in itertools.combinations(pool_idx, 6):
                    combos.append((i30, iadv, i03))

    n_combos = len(combos)
    log.info("%d candidate combinations | %d total sims | %d workers | chunk %d",
             n_combos, n_sims, n_workers, chunk_size)

    i30_arr  = np.array([list(c[0]) for c in combos], dtype=np.int32)
    iadv_arr = np.array([list(c[1]) for c in combos], dtype=np.int32)
    i03_arr  = np.array([list(c[2]) for c in combos], dtype=np.int32)

    # --- accumulators ---
    score_hists = np.zeros((n_combos, 11), dtype=np.int64)
    marg_tz  = np.zeros(n, dtype=np.int64)
    marg_adv = np.zeros(n, dtype=np.int64)
    marg_oz  = np.zeros(n, dtype=np.int64)
    convergence_history = []

    def _absorb(tz, adv, oz):
        nonlocal marg_tz, marg_adv, marg_oz
        marg_tz  += tz.sum(0)
        marg_adv += adv.sum(0)
        marg_oz  += oz.sum(0)
        _accumulate_scores(tz, adv, oz, i30_arr, iadv_arr, i03_arr, score_hists)

    _absorb(tz_w, adv_w, oz_w)
    del tz_w, adv_w, oz_w

    # warm-up snapshot
    p5_wu  = score_hists[:, 5:].sum(1) / warmup_n
    wu_idx = int(p5_wu.argmax())
    wu_i30, wu_iadv, wu_i03 = combos[wu_idx]
    convergence_history.append({
        "sims":      warmup_n,
        "best_p5":   float(p5_wu[wu_idx]),
        "picks_30":  tuple(teams[i] for i in wu_i30),
        "picks_adv": tuple(teams[i] for i in wu_iadv),
        "picks_03":  tuple(teams[i] for i in wu_i03),
    })

    # --- main loop: process remaining sims in parallel chunks ---
    import time
    remaining      = n_sims - warmup_n
    processed      = warmup_n
    chunk_seed     = RANDOM_SEED + 1
    stable_count   = 0
    prev_best_idx  = -1
    prev_p5        = -1.0
    t_start        = time.time()
    round_num      = 0

    with mp.Pool(processes=n_workers) as pool:
        while remaining > 0:
            round_num += 1
            batch_n = min(chunk_size * n_workers, remaining)
            base    = batch_n // n_workers
            extra   = batch_n  % n_workers
            sizes   = [base + (1 if i < extra else 0) for i in range(n_workers)]
            args    = [(teams, prob_cache, sizes[i], chunk_seed + i)
                       for i in range(n_workers)]
            chunk_seed += n_workers

            log.info("Round %d | simulating %d sims on %d workers...",
                     round_num, batch_n, n_workers)
            t_sim = time.time()
            results = pool.map(_run_chunk, args)
            log.info("  ...sims done in %.1fs | accumulating %d combos...",
                     time.time() - t_sim, n_combos)

            t_acc = time.time()
            for tz_c, adv_c, oz_c in results:
                _absorb(tz_c, adv_c, oz_c)
            log.info("  ...accumulation done in %.1fs", time.time() - t_acc)

            processed += batch_n
            remaining -= batch_n

            elapsed = time.time() - t_start
            rate    = processed / elapsed if elapsed > 0 else 1
            eta_s   = remaining / rate
            log.info("  Progress: %d / %d sims (%.1f%%) | %.0f sims/s | ETA ~%.0fs",
                     processed, n_sims,
                     100.0 * processed / n_sims,
                     rate, eta_s)

            # --- current best (always, for history + convergence) ---
            p5_all  = score_hists[:, 5:].sum(1) / processed
            cur_idx = int(p5_all.argmax())
            cur_p5  = float(p5_all[cur_idx])
            snap_i30, snap_iadv, snap_i03 = combos[cur_idx]
            convergence_history.append({
                "sims":      processed,
                "best_p5":   cur_p5,
                "picks_30":  tuple(teams[i] for i in snap_i30),
                "picks_adv": tuple(teams[i] for i in snap_iadv),
                "picks_03":  tuple(teams[i] for i in snap_i03),
            })

            # --- convergence check ---
            if converge and processed >= min_sims:
                delta = abs(cur_p5 - prev_p5)

                if cur_idx == prev_best_idx and delta < tol:
                    stable_count += 1
                    log.info("  Convergence: stable %d/%d | P(>=5)=%.4f | delta=%.5f",
                             stable_count, patience, cur_p5, delta)
                    if stable_count >= patience:
                        log.info("Converged after %d sims (saved %d).",
                                 processed, n_sims - processed)
                        break
                else:
                    stable_count = 0
                    log.info("  Convergence: not stable | P(>=5)=%.4f | delta=%.5f",
                             cur_p5, delta)
                prev_best_idx = cur_idx
                prev_p5       = cur_p5

    # divide by actual sims run (may be < n_sims if converged early)
    p30  = marg_tz  / processed
    padv = marg_adv / processed
    p03  = marg_oz  / processed

    return p30, padv, p03, combos, score_hists, convergence_history


# ---------------------------------------------------------------------------
# Optimizer — standard path (full boolean arrays)
# ---------------------------------------------------------------------------

def optimize_pickem(
    teams:          List[str],
    three_zero_arr: np.ndarray,
    advanced_arr:   np.ndarray,
    zero_three_arr: np.ndarray,
    top_k_30:  int = 8,
    top_k_03:  int = 8,
    top_k_adv: int = 12,
) -> Tuple[dict, float]:
    p30  = three_zero_arr.mean(0)
    padv = advanced_arr.mean(0)
    p03  = zero_three_arr.mean(0)

    cand_30  = list(np.argsort(p30)[::-1][:top_k_30])
    cand_03  = list(np.argsort(p03)[::-1][:top_k_03])
    cand_adv = list(np.argsort(padv)[::-1][:top_k_adv])

    best_p5    = -1.0
    best_picks = None
    n_combos   = 0

    for i30 in itertools.combinations(cand_30, 2):
        for i03 in itertools.combinations(cand_03, 2):
            excluded = set(i30) | set(i03)
            adv_pool = [i for i in cand_adv if i not in excluded]
            if len(adv_pool) < 6:
                continue
            for iadv in itertools.combinations(adv_pool, 6):
                n_combos += 1
                score = (
                    three_zero_arr[:, list(i30)].sum(1)  +
                    advanced_arr[:,   list(iadv)].sum(1) +
                    zero_three_arr[:, list(i03)].sum(1)
                )
                p5 = float((score >= 5).mean())
                if p5 > best_p5:
                    best_p5   = p5
                    best_picks = {
                        "three_zero": [teams[i] for i in i30],
                        "advance":    [teams[i] for i in iadv],
                        "zero_three": [teams[i] for i in i03],
                    }

    log.info("Evaluated %d pick-set combinations", n_combos)
    return best_picks, best_p5


def score_distribution(
    picks:          dict,
    teams:          List[str],
    three_zero_arr: np.ndarray,
    advanced_arr:   np.ndarray,
    zero_three_arr: np.ndarray,
) -> Dict[int, float]:
    idx  = {t: i for i, t in enumerate(teams)}
    i30  = [idx[t] for t in picks["three_zero"]]
    iadv = [idx[t] for t in picks["advance"]]
    i03  = [idx[t] for t in picks["zero_three"]]
    score = (
        three_zero_arr[:, i30].sum(1)  +
        advanced_arr[:,   iadv].sum(1) +
        zero_three_arr[:, i03].sum(1)
    )
    n = len(score)
    return {k: float((score == k).sum() / n) for k in range(11)}


# ---------------------------------------------------------------------------
# Optimizer — chunked path (score histograms)
# ---------------------------------------------------------------------------

def optimize_from_histograms(
    teams:       List[str],
    combos:      list,
    score_hists: np.ndarray,
    n_sims:      int = None,
) -> Tuple[dict, float, int]:
    """Find the best pick set from pre-accumulated score histograms.

    n_sims is derived from the histogram totals when not supplied, so callers
    don't need to track the actual sim count after early convergence.
    """
    if n_sims is None:
        n_sims = int(score_hists[0].sum())
    p5_all   = score_hists[:, 5:].sum(1) / n_sims
    best_idx = int(p5_all.argmax())
    i30, iadv, i03 = combos[best_idx]
    best_picks = {
        "three_zero": [teams[i] for i in i30],
        "advance":    [teams[i] for i in iadv],
        "zero_three": [teams[i] for i in i03],
    }
    log.info("Best P(>=5): %.3f  (combo %d / %d, n=%d)",
             p5_all[best_idx], best_idx, len(combos), n_sims)
    return best_picks, float(p5_all[best_idx]), best_idx


def score_dist_from_hist(
    best_idx:    int,
    score_hists: np.ndarray,
    n_sims:      int = None,
) -> Dict[int, float]:
    if n_sims is None:
        n_sims = int(score_hists[best_idx].sum())
    counts = score_hists[best_idx]
    return {k: float(counts[k] / n_sims) for k in range(11)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims",     type=int, default=50_000)
    parser.add_argument("--n-workers",  type=int, default=None,
                        help="CPU workers (default: all logical cores)")
    parser.add_argument("--chunk-size",  type=int,   default=CHUNK_SIZE,
                        help=f"Sims per worker per round in chunked mode (default: {CHUNK_SIZE})")
    parser.add_argument("--no-converge", action="store_true",
                        help="Disable early stopping; always run all --n-sims")
    parser.add_argument("--tol",         type=float, default=0.001,
                        help="P(>=5) change tolerance for convergence (default: 0.001)")
    parser.add_argument("--patience",    type=int,   default=3,
                        help="Stable rounds needed to declare convergence (default: 3)")
    parser.add_argument("--min-sims",    type=int,   default=50_000,
                        help="Minimum sims before convergence can trigger (default: 50000)")
    args = parser.parse_args()

    teams = list(STAGE1_TEAMS)
    log.info("Stage 1 teams: %s", teams)

    # Load model artifacts
    from xgboost import XGBClassifier
    input_dim    = pickle.load(open(MODEL_DIR / "input_dim.pkl",    "rb"))
    pt_model     = MatchPredictor(input_dim)
    pt_model.load_state_dict(
        torch.load(MODEL_DIR / "pytorch_model.pt", map_location="cpu")
    )
    pt_model.eval()
    xgb_model    = XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / "xgb_model.json"))
    scaler       = pickle.load(open(MODEL_DIR / "scaler.pkl",       "rb"))
    feature_cols = pickle.load(open(MODEL_DIR / "feature_cols.pkl", "rb"))
    blend_w      = pickle.load(open(MODEL_DIR / "blend_weight.pkl", "rb"))
    elo          = pickle.load(open(MODEL_DIR / "elo.pkl",          "rb"))

    team_dfs = fetch_all_team_stats(teams=teams)

    log.info("Pre-computing pairwise win probabilities...")
    prob_cache = precompute_win_probs(
        teams, team_dfs, elo, pt_model, xgb_model,
        scaler, feature_cols, blend_w,
    )

    # Print win-prob matrix
    print("\nWin probability matrix (row beats column):")
    header = f"{'':>22}" + "".join(f"{t[:6]:>8}" for t in teams)
    print(header)
    for a in teams:
        row = f"{a:<22}" + "".join(
            f"{prob_cache.get((a, b), 0.5):>7.1%}" if a != b else f"{'—':>8}"
            for b in teams
        )
        print(row)

    # --- simulation path ---
    log.info("Running %d Stage 1 Swiss simulations...", args.n_sims)

    if args.n_sims > CHUNK_THRESHOLD:
        log.info("Chunked mode: RAM stays bounded regardless of sim count.")
        p30, padv, p03, combos, score_hists, _ = run_simulations_chunked(
            teams, prob_cache, args.n_sims,
            n_workers=args.n_workers, chunk_size=args.chunk_size,
            converge=not args.no_converge, tol=args.tol,
            patience=args.patience, min_sims=args.min_sims,
        )
        best_picks, best_p5, best_idx = optimize_from_histograms(
            teams, combos, score_hists
        )
        dist = score_dist_from_hist(best_idx, score_hists)

    else:
        three_zero_arr, advanced_arr, zero_three_arr = run_simulations(
            teams, prob_cache, args.n_sims, n_workers=args.n_workers
        )
        p30  = three_zero_arr.mean(0)
        padv = advanced_arr.mean(0)
        p03  = zero_three_arr.mean(0)
        log.info("Searching for optimal pick set...")
        best_picks, best_p5 = optimize_pickem(
            teams, three_zero_arr, advanced_arr, zero_three_arr
        )
        dist = score_distribution(
            best_picks, teams, three_zero_arr, advanced_arr, zero_three_arr
        )

    # --- marginal probabilities ---
    print("\nMarginal outcome probabilities:")
    print(f"{'Team':<24} {'P(3-0)':>8} {'P(adv)':>8} {'P(0-3)':>8}")
    print("-" * 52)
    for i, t in enumerate(teams):
        print(f"{t:<24} {p30[i]:>7.1%} {padv[i]:>7.1%} {p03[i]:>7.1%}")

    # --- results ---
    sep = "=" * 62
    print(f"\n{sep}")
    print("  OPTIMAL PICK'EM — IEM Cologne 2026 Opening Stage")
    print(sep)
    print(f"  3-0 picks  : {', '.join(best_picks['three_zero'])}")
    print(f"  Advance    : {', '.join(best_picks['advance'])}")
    print(f"  0-3 picks  : {', '.join(best_picks['zero_three'])}")
    print(sep)
    print(f"  P(>=4 correct) : {sum(v for k,v in dist.items() if k>=4):.1%}")
    print(f"  P(>=5 correct) : {sum(v for k,v in dist.items() if k>=5):.1%}")
    print(f"  P(>=6 correct) : {sum(v for k,v in dist.items() if k>=6):.1%}")
    print(f"  E[correct]    : {sum(k*v for k,v in dist.items()):.2f}")
    print(sep)
    print("\nScore distribution:")
    for k in range(11):
        bar = "#" * int(dist[k] * 50)
        print(f"  {k:2d} correct: {dist[k]:5.1%}  {bar}")
    print()


if __name__ == "__main__":
    mp.freeze_support()   # needed for Windows spawn
    main()
