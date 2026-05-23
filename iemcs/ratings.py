"""Online rating systems trained chronologically on CS2 results: Elo and Glicko-2.

Both are updated match-by-match so that, at any point in a chronological pass, the
current ratings reflect only past results (no leakage). After processing the whole
history the ratings represent each team's present strength.
"""
from __future__ import annotations

import math

from .config import ELO, GLICKO

_PI2 = math.pi ** 2


class Elo:
    def __init__(self, base: float | None = None, k: float | None = None,
                 scale: float | None = None):
        self.base = ELO.base_rating if base is None else base
        self.k = ELO.k_factor if k is None else k
        self.scale = ELO.scale if scale is None else scale
        self.r: dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self.r.get(team, self.base)

    def expected(self, a: str, b: str) -> float:
        """P(a beats b) under Elo."""
        return 1.0 / (1.0 + 10 ** ((self.rating(b) - self.rating(a)) / self.scale))

    def update(self, winner: str, loser: str) -> None:
        ew = self.expected(winner, loser)
        self.r[winner] = self.rating(winner) + self.k * (1.0 - ew)
        self.r[loser] = self.rating(loser) + self.k * (0.0 - (1.0 - ew))


class Glicko2:
    """Glicko-2 with one game per rating period (updated every match).

    Stores Glicko-2 internal (mu, phi, sigma). `rating()` / `rd()` return the
    human-scale (Elo-like) values.
    """

    def __init__(self):
        self.cfg = GLICKO
        self.mu: dict[str, float] = {}
        self.phi: dict[str, float] = {}
        self.sig: dict[str, float] = {}

    def _get(self, t: str) -> tuple[float, float, float]:
        if t not in self.mu:
            self.mu[t] = 0.0
            self.phi[t] = self.cfg.base_rd / self.cfg.scale
            self.sig[t] = self.cfg.base_vol
        return self.mu[t], self.phi[t], self.sig[t]

    def rating(self, t: str) -> float:
        mu, _, _ = self._get(t)
        return mu * self.cfg.scale + self.cfg.base_rating

    def rd(self, t: str) -> float:
        _, phi, _ = self._get(t)
        return phi * self.cfg.scale

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / _PI2)

    def expected(self, a: str, b: str) -> float:
        """P(a beats b), folding both teams' deviation into one g-factor."""
        mu_a, phi_a, _ = self._get(a)
        mu_b, phi_b, _ = self._get(b)
        g = self._g(math.sqrt(phi_a ** 2 + phi_b ** 2))
        return 1.0 / (1.0 + math.exp(-g * (mu_a - mu_b)))

    def _new_sigma(self, phi: float, sigma: float, v: float, delta: float) -> float:
        a = math.log(sigma ** 2)
        tau = self.cfg.tau

        def f(x):
            ex = math.exp(x)
            num = ex * (delta ** 2 - phi ** 2 - v - ex)
            den = 2.0 * (phi ** 2 + v + ex) ** 2
            return num / den - (x - a) / tau ** 2

        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau
        fa, fb = f(A), f(B)
        for _ in range(100):
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            if fc * fb < 0:
                A, fa = B, fb
            else:
                fa /= 2.0
            B, fb = C, fc
            if abs(B - A) < 1e-6:
                break
        return math.exp(A / 2.0)

    def update(self, winner: str, loser: str) -> None:
        # Update both teams against each other (one game this period).
        for me, opp, s in ((winner, loser, 1.0), (loser, winner, 0.0)):
            mu, phi, sigma = self._get(me)
            mu_o, phi_o, _ = self._get(opp)
            g = self._g(phi_o)
            e = 1.0 / (1.0 + math.exp(-g * (mu - mu_o)))
            v = 1.0 / (g ** 2 * e * (1.0 - e))
            delta = v * g * (s - e)
            sigma_p = self._new_sigma(phi, sigma, v, delta)
            phi_star = math.sqrt(phi ** 2 + sigma_p ** 2)
            phi_p = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
            mu_p = mu + phi_p ** 2 * g * (s - e)
            # stash (apply after both computed to avoid using updated opp mid-step)
            self._pending = getattr(self, "_pending", {})
            self._pending[me] = (mu_p, phi_p, sigma_p)
        for t, (mu_p, phi_p, sigma_p) in self._pending.items():
            self.mu[t], self.phi[t], self.sig[t] = mu_p, phi_p, sigma_p
        self._pending = {}
