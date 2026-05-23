"""8-team single-elimination playoff bracket (Bo3, Bo5 grand final)."""
from __future__ import annotations

from .config import PLAYOFF_FINAL_BO5
from .series import series_winprob


def _play(a: str, b: str, pmap, rng, bo: int) -> tuple[str, str]:
    p = series_winprob(pmap(a, b), bo)
    return (a, b) if rng.random() < p else (b, a)


def simulate_playoffs(seeds: list[str], pmap, rng) -> dict:
    """Standard 1v8/4v5/2v7/3v6 bracket.

    `seeds` are the 8 playoff teams in seed order (1..8, best first).
    Returns dict with keys: champion, runner_up, semifinalists (4), and
    `reached` mapping team -> furthest round in {"QF","SF","Final","Champion"}.
    """
    s = seeds
    reached = {t: "QF" for t in s}                  # all 8 reached the quarterfinals
    qf = [(s[0], s[7]), (s[3], s[4]), (s[1], s[6]), (s[2], s[5])]

    sf_teams = []
    for a, b in qf:
        w, _ = _play(a, b, pmap, rng, bo=3)
        sf_teams.append(w)
        reached[w] = "SF"

    finalists = []
    for a, b in ((sf_teams[0], sf_teams[1]), (sf_teams[2], sf_teams[3])):
        w, _ = _play(a, b, pmap, rng, bo=3)
        finalists.append(w)
        reached[w] = "Final"

    final_bo = 5 if PLAYOFF_FINAL_BO5 else 3
    champ, runner = _play(finalists[0], finalists[1], pmap, rng, bo=final_bo)
    reached[champ] = "Champion"

    return {
        "champion": champ,
        "runner_up": runner,
        "semifinalists": sf_teams,
        "reached": reached,
    }
