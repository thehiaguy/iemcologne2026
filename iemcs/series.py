"""Series probability math.

The match model is trained to predict the winner of a (Bo3-equivalent) encounter.
To simulate the real bracket we need Bo1, Bo3 and Bo5 odds. We treat the trained
probability as Bo3, invert it to an implied per-map probability `p`, and recompute
the odds for whatever format a given match actually uses.
"""
from __future__ import annotations

from math import comb

from .config import PROB_SHRINK, TRAIN_ENCOUNTER_FORMAT

_MAPS = {1: 1, 3: 3, 5: 5}


def series_winprob(p_map: float, bo: int) -> float:
    """P(win a best-of-`bo`) given per-map win probability `p_map`."""
    if bo == 1:
        return p_map
    need = bo // 2 + 1
    return sum(
        comb(bo, k) * p_map ** k * (1 - p_map) ** (bo - k)
        for k in range(need, bo + 1)
    )


def map_prob_from_encounter(p_enc: float, enc_bo: int = 3) -> float:
    """Invert a best-of-`enc_bo` win probability to the implied per-map probability."""
    if enc_bo == 1:
        return min(max(p_enc, 1e-6), 1 - 1e-6)
    lo, hi = 1e-6, 1 - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        if series_winprob(mid, enc_bo) < p_enc:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _shrink(p: float) -> float:
    return (1 - PROB_SHRINK) * p + PROB_SHRINK * 0.5


def map_prob_from_model(p_encounter: float) -> float:
    """Per-map probability implied by the model output, with overconfidence shrink."""
    enc_bo = _MAPS[3 if TRAIN_ENCOUNTER_FORMAT == "bo3" else 1]
    p = map_prob_from_encounter(p_encounter, enc_bo)
    return _shrink(p)
