"""Compare the CUDA write-disturbance kernel with a NumPy reference on short pulse sequences.

Node group of a cell: potentiation phase, row pulsed -> CP, else CN; depression phase, row pulsed
-> CN, else CP. Every CP <-> CN transition decays the weight, w <- (1 - gamma) w. With pulse
scheduling the group flips only when the row pulse flips (twice per flip); without it the phase
alternates at every slot (two events per slot, minus one when the row pulse also flips).
Inputs in {0, 1} and errors in {0, +-1} with lr = BL * dw_min give deterministic pulses.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aihwkit.simulator.configs import SingleRPUConfig, CapSynapseDevice
from aihwkit.simulator.configs.utils import PulseType
from aihwkit.simulator.tiles import AnalogTile

BL, DW = 1, 2.0 / 1000
GAMMA = 0.02          # large, so that the decay is visible in a short sequence
R, BETA, LAM_EFF = 2.0, 0.35 / 3, 0.0324 * 1.5   # median cell; lam * lam_scale


def make_tile(gamma, ps, d=4, x=3, seed=0):
    dev = CapSynapseDevice(
        shared_transistor=True, dw_min_std=0.0, hs_gamma=gamma, hs_pulse_scheduling=ps,
        vov_std=0.0, w_rel_std=0.0, bound_std=0.0, lam_rel_std=0.0)
    cfg = SingleRPUConfig(device=dev)
    cfg.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
    cfg.update.desired_bl = BL
    cfg.update.fixed_bl = True
    cfg.update.update_bl_management = False
    cfg.update.update_management = True
    cfg.forward.is_perfect = True
    cfg.backward.is_perfect = True
    torch.manual_seed(seed)
    t = AnalogTile(d, x, cfg).cuda()
    t.set_learning_rate(BL * DW)
    return t


def ref_update(w, pot):
    """Median 7T1C cell (alpha = 1)."""
    su = R / 1000
    if pot:
        gb = 1 - R * BETA
        dw = su * (1 + LAM_EFF * (gb - w)) if w <= gb else su * (1 - w) / (R * BETA)
        return min(w + dw, 1.0)
    gb = -1 + R * BETA
    dw = su * (1 + LAM_EFF * (w - gb)) if w > gb else su * (w + 1) / (R * BETA)
    return max(w - dw, -1.0)


def reference(seq, w0, gamma, ps, d_size, x_size):
    """seq: list of (x_vals, d_vals) mat-vecs applied in order."""
    w = np.full((d_size, x_size), w0)
    prev = np.full(x_size, -1)                       # previous row bit per input line
    for xv, dv in seq:
        for j in range(x_size):
            xb = 1 if xv[j] != 0 else 0
            if prev[j] < 0:
                n_ev = 0 if ps else 1
            else:
                flip = prev[j] != xb
                n_ev = (2 if flip else 0) if ps else (1 if flip else 2)
            w[:, j] *= (1.0 - gamma) ** n_ev
            prev[j] = xb
            if xb:
                for i in range(d_size):
                    if dv[i] != 0:
                        pot = (xv[j] < 0) != (dv[i] < 0)
                        w[i, j] = ref_update(w[i, j], pot)
    return w


def run(name, seq, w0, gamma, ps):
    d_size, x_size = len(seq[0][1]), len(seq[0][0])
    t = make_tile(gamma, ps, d_size, x_size)
    t.set_weights(torch.full((d_size, x_size), w0))
    for xv, dv in seq:
        t.update(torch.tensor([xv], dtype=torch.float32).cuda(),
                 torch.tensor([dv], dtype=torch.float32).cuda())
    gpu = t.get_weights()[0].numpy()
    ref = reference(seq, w0, gamma, ps, d_size, x_size)
    err = np.abs(gpu - ref).max()
    print(f"{name:34s} max|dw| = {err:.2e}   gpu[0]={np.round(gpu[0], 5)}  ref[0]={np.round(ref[0], 5)}")
    return err


if __name__ == "__main__":
    errs = []
    # A: constant row pulses: no decay with pulse scheduling
    seq = [([1, 1, 1], [-1, -1, -1, -1])] * 4
    errs.append(run("A PS, no row flip (no decay)", seq, 0.5, GAMMA, True))
    # B: alternating row pulses: 2 events per flip
    seq = [([1, 1, 1], [0, 0, 0, 0]), ([0, 0, 0], [0, 0, 0, 0])] * 3
    errs.append(run("B PS, row flips only", seq, 0.5, GAMMA, True))
    # C: same without pulse scheduling
    errs.append(run("C no PS, row flips only", seq, 0.5, GAMMA, False))
    # D, E: mixed rows and signs, updates and half-selects
    seq = [([1, 0, 1], [1, 0, -1, 1]), ([1, 1, 0], [0, -1, 1, 0]), ([0, 1, 1], [-1, 1, 0, -1])] * 2
    errs.append(run("D PS, mixed updates + flips", seq, 0.3, GAMMA, True))
    errs.append(run("E no PS, mixed updates + flips", seq, 0.3, GAMMA, False))
    # F: gamma = 0 uses the standard kernel
    errs.append(run("F gamma=0 (fast kernel path)", seq, 0.3, 0.0, True))
    # G: gamma of the paper
    errs.append(run("G PS, gamma=0.000112", seq, 0.3, 0.000112, True))
    print(f"\nworst deviation over all cases: {max(errs):.2e}")
