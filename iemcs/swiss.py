"""Major Swiss-stage engine.

Implements the CS2 Major Swiss system used at IEM Cologne 2026:
  * 16 teams, first to 3 wins advances, 3 losses eliminates (top 8 advance).
  * Round 1 pairs initial seed i vs i+8.
  * Later rounds pair within win-loss groups, re-seeded by Buchholz then seed,
    high-vs-low, avoiding rematches.
  * Bo1 except matches that would advance or eliminate a team (a 2-x or x-2
    record) are Bo3; Stage 3 is entirely Bo3.
"""
from __future__ import annotations

from .series import series_winprob


class _T:
    __slots__ = ("name", "seed", "w", "l", "opps")

    def __init__(self, name: str, seed: int):
        self.name = name
        self.seed = seed
        self.w = 0
        self.l = 0
        self.opps: list["_T"] = []

    def buchholz(self) -> int:
        return sum(o.w - o.l for o in self.opps)


def _high_low_pairs(group: list[_T]) -> list[tuple[_T, _T]]:
    """Pair a record group high-vs-low, avoiding rematches where possible."""
    # Re-seed within the group: best Buchholz first, then best initial seed.
    order = sorted(group, key=lambda t: (-t.buchholz(), t.seed))
    pool = list(order)
    pairs = []
    while pool:
        top = pool.pop(0)
        chosen = None
        for j in range(len(pool) - 1, -1, -1):       # search from the bottom up
            if pool[j] not in top.opps:
                chosen = j
                break
        if chosen is None:                            # forced rematch
            chosen = len(pool) - 1
        opp = pool.pop(chosen)
        pairs.append((top, opp))
    return pairs


def _pairings(active: list[_T], rnd: int) -> list[tuple[_T, _T]]:
    if rnd == 1:
        order = sorted(active, key=lambda t: t.seed)
        half = len(order) // 2
        return list(zip(order[:half], order[half:]))
    pairs = []
    groups: dict[tuple[int, int], list[_T]] = {}
    for t in active:
        groups.setdefault((t.w, t.l), []).append(t)
    for record in sorted(groups):                     # deterministic group order
        pairs.extend(_high_low_pairs(groups[record]))
    return pairs


def simulate_swiss(seeded_teams, pmap, rng, all_bo3: bool = False,
                   with_records: bool = False):
    """Run one Swiss stage.

    Parameters
    ----------
    seeded_teams : list[str]   team names in seed order (best first)
    pmap         : callable    pmap(a, b) -> per-map P(a beats b)
    rng          : np.random.Generator
    all_bo3      : bool         Stage 3 flag
    with_records : bool         also return {team: (wins, losses)}

    Returns (advanced, eliminated) name lists in finishing order; if
    `with_records`, returns (advanced, eliminated, records).
    """
    teams = {n: _T(n, i + 1) for i, n in enumerate(seeded_teams)}
    active = list(teams.values())
    advanced: list[_T] = []
    eliminated: list[_T] = []

    rnd = 0
    while active:
        rnd += 1
        for a, b in _pairings(active, rnd):
            if all_bo3:
                bo = 3
            else:
                bo = 3 if (a.w == 2 or a.l == 2) else 1
            p = series_winprob(pmap(a.name, b.name), bo)
            if rng.random() < p:
                win, lose = a, b
            else:
                win, lose = b, a
            win.w += 1
            lose.l += 1
            win.opps.append(lose)
            lose.opps.append(win)

        still = []
        for t in active:
            if t.w >= 3:
                advanced.append(t)
            elif t.l >= 3:
                eliminated.append(t)
            else:
                still.append(t)
        active = still

    # Finishing order: advancers by (wins desc, buchholz desc, seed); reverse for out.
    advanced.sort(key=lambda t: (-t.w, -t.buchholz(), t.seed))
    eliminated.sort(key=lambda t: (t.l, -t.buchholz(), t.seed))
    adv_names = [t.name for t in advanced]
    elim_names = [t.name for t in eliminated]
    if with_records:
        records = {t.name: (t.w, t.l) for t in advanced + eliminated}
        return adv_names, elim_names, records
    return adv_names, elim_names
