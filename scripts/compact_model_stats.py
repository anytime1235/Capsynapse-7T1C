"""Statistics of the compact model (Table S10), N cells per architecture: symmetry point, V_cap change after
10 potentiation (depression) pulses from V_cap = 0, and their difference.

  python scripts/compact_model_stats.py [--device table-s9|spice] [-n 2000] [--seed 0]
"""
import argparse, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from varsim import sample_params, symmetry_point, delta_g_pot, delta_g_dep

DEVICES = {
    "spice":    dict(),                                                         # Note S4
    "table-s9": dict(vov_median=0.51, vov_sigma=0.05, lam=0.05, lam_rel_sigma_7t=1 / 20, step_median=300),
}
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="table-s9", choices=sorted(DEVICES))
ap.add_argument("-n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pulses", type=int, default=10)
a = ap.parse_args()

VDD = 3.0


def pulses(p, pot, n):
    g = p.gref.copy()
    for _ in range(n):
        g = np.minimum(g + delta_g_pot(g, p), p.gmax) if pot else np.maximum(g - delta_g_dep(g, p), p.gmin)
    return VDD * 1e3 * (g - p.gref) / (p.gmax - p.gmin)      # mV


print(f"{a.device} device, N = {a.n}, seed {a.seed}")
print(f"{'':6s} {'':26s} {'mean (mV)':>10s} {'s.d. (mV)':>10s}")
for arch in ("6T1C", "7T1C"):
    p = sample_params(a.n, arch, a.seed, **DEVICES[a.device])
    sym = VDD * 1e3 * symmetry_point(p)
    up = pulses(p, True, a.pulses)
    dn = -pulses(p, False, a.pulses)
    for name, v in (("symmetry point", sym), ("dV_cap, potentiation", up), ("dV_cap, depression", dn),
                    ("pot - dep", up - dn)):
        print(f"{arch:6s} {name:26s} {v.mean():10.2f} {v.std(ddof=1):10.2f}")
