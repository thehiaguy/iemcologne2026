"""Series-probability math."""
from iemcs import series


def test_bo1_identity():
    assert series.series_winprob(0.73, 1) == 0.73


def test_format_amplifies_favorite():
    p = 0.60
    assert series.series_winprob(p, 1) < series.series_winprob(p, 3) \
        < series.series_winprob(p, 5)


def test_format_dampens_underdog():
    p = 0.40
    assert series.series_winprob(p, 1) > series.series_winprob(p, 3) \
        > series.series_winprob(p, 5)


def test_symmetry():
    for p in (0.25, 0.5, 0.8):
        for bo in (1, 3, 5):
            assert abs(series.series_winprob(p, bo)
                       + series.series_winprob(1 - p, bo) - 1) < 1e-9


def test_inversion_round_trip():
    for p in (0.52, 0.7, 0.9):
        for bo in (3, 5):
            P = series.series_winprob(p, bo)
            assert abs(series.map_prob_from_encounter(P, bo) - p) < 1e-3
