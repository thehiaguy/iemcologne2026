"""Full-tournament Monte-Carlo invariants (synthetic strengths, no data needed)."""
import numpy as np

from iemcs import tournament
from iemcs.teams import FIELD


def _synthetic_inputs():
    names = [t.name for t in FIELD]
    strength = {n: len(names) - i for i, n in enumerate(names)}
    pmap = {}
    for a in names:
        for b in names:
            if a != b:
                d = strength[a] - strength[b]
                pmap[(a, b)] = 1.0 / (1.0 + 10 ** (-d / 12.0))
    vrs = {n: 1000.0 + strength[n] for n in names}
    return pmap, vrs


def test_probability_invariants():
    pmap, vrs = _synthetic_inputs()
    res = tournament.run_monte_carlo(pmap, vrs, n_sims=800, n_jobs=1)
    assert abs(res["champion"].sum() - 1.0) < 1e-9
    assert abs(res["adv_s3"].sum() - 8.0) < 1e-9
    assert abs(res["semifinal"].sum() - 4.0) < 1e-9
    assert abs(res["final"].sum() - 2.0) < 1e-9
    for col in ("adv_s1", "adv_s2", "adv_s3", "champion"):
        v = res[col].dropna()
        assert (v >= 0).all() and (v <= 1).all()


def test_stage_seeds_skip_earlier_stages():
    pmap, vrs = _synthetic_inputs()
    res = tournament.run_monte_carlo(pmap, vrs, n_sims=300, n_jobs=1)
    for t in FIELD:
        if t.stage >= 2:
            assert np.isnan(res.loc[t.name, "adv_s1"])
        if t.stage >= 3:
            assert np.isnan(res.loc[t.name, "adv_s2"])
