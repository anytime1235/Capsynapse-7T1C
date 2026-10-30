"""Compact model of the 6T1C / 7T1C capacitive synapse (Supplemental Note S4), in NumPy.

Conductances are in uS. step_median is the number of states of the median cell over [Gmin, Gmax].
lambda is in 1/V: the offset G - G_boundary is converted to V_cap with (Gmax - Gmin) <-> VDD.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

VDD = 3.0
VOV_MEDIAN = 0.35
VOV_SIGMA = 0.043
GMAX_MU, GMIN_MU, G_SIGMA = 4.5, -4.5, 0.15
LAMBDA = 0.0324
W_REL_SIGMA = 1.0 / 30.0        # channel width, 3 sigma = 10 %
STEP_MEDIAN = 1000


@dataclass
class CellParams:
    """Per-cell parameter arrays, shape (n,)."""
    gmax: np.ndarray
    gmin: np.ndarray
    beta_pot: np.ndarray
    beta_dep: np.ndarray
    alpha_pot: np.ndarray
    alpha_dep: np.ndarray
    lam_pot: np.ndarray
    lam_dep: np.ndarray
    step_median: float = STEP_MEDIAN
    vdd: float = VDD

    @property
    def n(self) -> int:
        return self.gmax.shape[0]

    @property
    def gref(self) -> np.ndarray:
        return 0.5 * (self.gmax + self.gmin)

    @property
    def gb_dep(self) -> np.ndarray:  # Eq. S3 / S8
        return self.gmin + (self.gmax - self.gmin) * self.beta_dep

    @property
    def gb_pot(self) -> np.ndarray:  # Eq. S4 / S9
        return self.gmax - (self.gmax - self.gmin) * self.beta_pot


def _alpha_from_beta(beta: np.ndarray, rng: np.random.Generator,
                     vov_median: float = VOV_MEDIAN, vdd: float = VDD) -> np.ndarray:
    """Eq. S14: alpha = U (VDD beta / Vov_median)^2, U = 1 + Z/30."""
    u = 1.0 + W_REL_SIGMA * rng.standard_normal(beta.shape)
    return u * (vdd * beta / vov_median) ** 2


def sample_params(n: int, arch: str, rng: np.random.Generator | int | None = None,
                  vov_median: float = VOV_MEDIAN, vov_sigma: float = VOV_SIGMA,
                  vdd: float = VDD, lam: float = LAMBDA,
                  lam_rel_sigma_7t: float = 1.0 / 30.0,
                  g_sigma: float = G_SIGMA,
                  step_median: float = STEP_MEDIAN) -> CellParams:
    """Sample n cells. 6T1C: (beta, alpha) drawn separately for potentiation (N2) and
    depression (N4). 7T1C: one (beta, alpha) for both directions (N7); lambda_pot and lambda_dep
    drawn separately. alpha is derived from beta (Eq. S14).
    """
    rng = np.random.default_rng(rng)
    gmax = rng.normal(GMAX_MU, g_sigma, n)
    gmin = rng.normal(GMIN_MU, g_sigma, n)
    b_mu, b_sd = vov_median / vdd, vov_sigma / vdd
    if arch.upper() == "6T1C":
        beta_pot = rng.normal(b_mu, b_sd, n)
        beta_dep = rng.normal(b_mu, b_sd, n)
        alpha_pot = _alpha_from_beta(beta_pot, rng, vov_median, vdd)
        alpha_dep = _alpha_from_beta(beta_dep, rng, vov_median, vdd)
        lam_pot = np.full(n, lam)
        lam_dep = np.full(n, lam)
    elif arch.upper() == "7T1C":
        beta = rng.normal(b_mu, b_sd, n)
        alpha = _alpha_from_beta(beta, rng, vov_median, vdd)
        beta_pot = beta_dep = beta
        alpha_pot = alpha_dep = alpha
        lam_pot = rng.normal(lam, lam * lam_rel_sigma_7t, n)
        lam_dep = rng.normal(lam, lam * lam_rel_sigma_7t, n)
    else:
        raise ValueError(arch)
    return CellParams(gmax, gmin, beta_pot, beta_dep, alpha_pot, alpha_dep,
                      lam_pot, lam_dep, step_median, vdd)


def delta_g_dep(g: np.ndarray, p: CellParams) -> np.ndarray:
    """Eq. S1 / S6: magnitude of the conductance decrease of one depression pulse."""
    rng_ = p.gmax - p.gmin
    base = rng_ * p.alpha_dep / p.step_median
    k = p.vdd / rng_                      # uS -> V
    lin = (g - p.gmin) * p.alpha_dep / (p.beta_dep * p.step_median)
    sat = base * (1.0 + p.lam_dep * k * (g - p.gb_dep))
    return np.where(g <= p.gb_dep, lin, sat)


def delta_g_pot(g: np.ndarray, p: CellParams) -> np.ndarray:
    """Eq. S2 / S7: magnitude of the conductance increase of one potentiation pulse."""
    rng_ = p.gmax - p.gmin
    base = rng_ * p.alpha_pot / p.step_median
    k = p.vdd / rng_                      # uS -> V
    sat = base * (1.0 + p.lam_pot * k * (p.gb_pot - g))
    lin = (p.gmax - g) * p.alpha_pot / (p.beta_pot * p.step_median)
    return np.where(g <= p.gb_pot, sat, lin)


def run_pulses(p: CellParams, n_pairs: int, g0: np.ndarray | None = None, record: bool = False):
    """Apply n_pairs of (pot, dep) pulses to every cell starting at g0 (default Gref).
    Returns final G (and trajectory if record)."""
    g = p.gref.copy() if g0 is None else np.array(g0, dtype=float)
    traj = [g.copy()] if record else None
    for _ in range(n_pairs):
        g = g + delta_g_pot(g, p)
        g = np.minimum(g, p.gmax)
        g = g - delta_g_dep(g, p)
        g = np.maximum(g, p.gmin)
        if record:
            traj.append(g.copy())
    return (g, np.array(traj)) if record else g


def symmetry_point(p: CellParams, n_pairs: int = 20000, normalized: bool = True) -> np.ndarray:
    """G after n_pairs potentiation/depression pulse pairs from Gref. normalized=True returns
    (G - Gref)/(Gmax - Gmin), i.e. V_cap/VDD."""
    g = run_pulses(p, n_pairs)
    if normalized:
        return (g - p.gref) / (p.gmax - p.gmin)
    return g
