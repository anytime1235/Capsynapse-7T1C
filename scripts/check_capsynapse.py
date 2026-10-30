"""Compare the symmetry points of the AIHWKIT CapSynapseDevice (CPU and CUDA) with the NumPy model.

20,000 potentiation/depression pulse pairs on 2000 cells per architecture. The two samples are drawn
by different generators, so the statistics agree to within sampling noise. --cuda-only skips the
slower CPU tile.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aihwkit.simulator.configs import SingleRPUConfig, CapSynapseDevice
from aihwkit.simulator.configs.utils import PulseType
from aihwkit.simulator.tiles import AnalogTile
from varsim import sample_params, symmetry_point

def run(arch, use_cuda, n=2000, n_pairs=20000, seed=0):
    dev = CapSynapseDevice(shared_transistor=(arch == "7T1C"), dw_min_std=0.0,
                           up_down=0.0, up_down_dtod=0.0, dw_min_dtod=0.0,
                           w_max_dtod=0.0, w_min_dtod=0.0, reset_std=0.0,
                           diffusion=0.0, lifetime=0.0)
    cfg = SingleRPUConfig(device=dev)
    cfg.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT   # one pulse per update
    cfg.update.desired_bl = 1
    cfg.update.update_bl_management = False
    cfg.update.update_management = False
    cfg.forward.is_perfect = True; cfg.backward.is_perfect = True
    torch.manual_seed(seed)
    tile = AnalogTile(n, 1, cfg)
    if use_cuda: tile = tile.cuda()
    tile.set_weights(torch.zeros(n, 1))
    devc = "cuda" if use_cuda else "cpu"
    x = torch.ones(1, 1, device=devc)
    tile.set_learning_rate(1.0)
    d_pot = -torch.ones(1, n, device=devc)
    d_dep = +torch.ones(1, n, device=devc)
    for _ in range(n_pairs):
        tile.update(x, d_pot); tile.update(x, d_dep)
    w = tile.get_weights()[0].detach().cpu().numpy().ravel()
    return w / 2.0  # (G - Gref)/(Gmax - Gmin)

for arch in ("6T1C", "7T1C"):
    p = sample_params(2000, arch, 0); ref = symmetry_point(p)
    for cuda in ((True,) if "--cuda-only" in sys.argv else (False, True)):
        s = run(arch, cuda)
        v = 3 * s; vr = 3 * ref
        print(f"{arch} {'CUDA' if cuda else 'CPU '}: mean={v.mean():+.4f} std={v.std():.4f} "
              f"|V|<0.5 frac={np.mean(abs(v)<0.5):.2f} | numpy ref: mean={vr.mean():+.4f} std={vr.std():.4f} frac={np.mean(abs(vr)<0.5):.2f}")
