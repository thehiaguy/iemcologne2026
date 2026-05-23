"""Swiss-stage engine invariants."""
import numpy as np

from iemcs.swiss import simulate_swiss


def _teams():
    return [f"T{i:02d}" for i in range(16)]


def test_eight_advance_eight_eliminated():
    teams = _teams()
    rng = np.random.default_rng(0)
    adv, elim = simulate_swiss(teams, lambda a, b: 0.5, rng)
    assert len(adv) == 8
    assert len(elim) == 8
    assert set(adv) | set(elim) == set(teams)
    assert not (set(adv) & set(elim))


def test_transitive_strength_extremes():
    """Strongest team always advances; weakest is always eliminated."""
    teams = _teams()
    strength = {t: 16 - i for i, t in enumerate(teams)}  # T00 strongest

    def pmap(a, b):
        return 1.0 if strength[a] > strength[b] else 0.0

    for seed in range(8):
        rng = np.random.default_rng(seed)
        adv, elim = simulate_swiss(teams, pmap, rng)
        assert teams[0] in adv
        assert teams[-1] in elim
        assert len(adv) == 8


def test_all_bo3_runs():
    teams = _teams()
    rng = np.random.default_rng(3)
    adv, elim = simulate_swiss(teams, lambda a, b: 0.55, rng, all_bo3=True)
    assert len(adv) == 8 and len(elim) == 8
