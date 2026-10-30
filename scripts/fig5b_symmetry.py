"""Symmetry-point distributions of the 6T1C and 7T1C compact model (Fig. 5B), N = 2000 cells each.

  python scripts/fig5b_symmetry.py [--device spice|table-s9] [-n 2000] [--seed 0]

Writes figures/fig5b_model_<device>.png.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from varsim import sample_params, symmetry_point

DEVICES = {
    "spice":    dict(),                                                         # Note S4
    "table-s9": dict(vov_median=0.51, vov_sigma=0.05, lam=0.05, lam_rel_sigma_7t=1 / 20, step_median=300),
}
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="spice", choices=sorted(DEVICES))
ap.add_argument("-n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

VDD = 3.0
LIM = {"6T1C": 1500.0, "7T1C": 150.0}           # x range (mV)
COL = {"6T1C": "#D95319", "7T1C": "#0072BD"}
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
for ax, arch in zip(axes, ("6T1C", "7T1C")):
    p = sample_params(a.n, arch, a.seed, **DEVICES[a.device])
    v = VDD * 1e3 * symmetry_point(p)            # mV
    print(f"{arch}: N={a.n}  mean {v.mean():+.2f} mV  s.d. {v.std(ddof=1):.2f} mV")
    lim = LIM[arch]
    ax.hist(np.clip(v, -lim, lim), bins=np.linspace(-lim, lim, 121), color=COL[arch])
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("V_cap at the symmetry point (mV)")
    ax.set_ylabel("cells")
    ax.set_title(arch)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
out = os.path.join(ROOT, "figures", f"fig5b_model_{a.device}.png")
fig.savefig(out, dpi=200)
print("saved", out)
