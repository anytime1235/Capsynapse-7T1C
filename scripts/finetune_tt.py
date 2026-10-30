"""Layer-wise Tiki-Taka fine-tuning of a pretrained ResNet-18 on CIFAR-10 with a core NVM array
and a 6T1C / 7T1C auxiliary array (AIHWKIT with the CapSynapseDevice of aihwkit_ext/).

Conv layers 2-17 are analog; the stem conv and the classifier stay digital. The layers are
fine-tuned one at a time, one epoch each, and the pass over the layers is repeated --iters times.
The test accuracy is recorded after every layer.

  python scripts/finetune_tt.py --arch 7T1C --nonideal variation,disturbance \\
      --fast-lr 0.24 --lr 0.01 --seed 0 --out results_rerun/7T1C_dist_s0.json
"""
import argparse, json, os, sys, time
import torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from varsim.nn.resnet_cifar import resnet18_cifar, conv_layers, loaders, evaluate
from varsim.nn.program import program_core_noise

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogConv2d
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (UnitCellRPUConfig, TransferCompound, CapSynapseDevice,
                                       LinearStepDevice, MappingParameter, IOParameters,
                                       UpdateParameters)
from aihwkit.simulator.parameters.enums import NoiseManagementType, BoundManagementType

# core NVM array and read-out (Methods)
CORE_DW_MIN = 0.002   # mean update step E(|dw|)
CORE_NL = 1.0         # update nonlinearity, Eq. 8 (AIHWKIT LinearStep gamma = NL/2)
CORE_C2C = 0.3        # cycle-to-cycle std of the update
CORE_D2D = 0.06       # device-to-device std of E(|dw|), NL and the conductance range
OUT_NOISE = 0.06      # current-sum std of inference and transfer reads
OMEGA = 0.8           # weight scaling SF = max|w| / 0.8, Eq. 11
BATCH = 128
TAU_D2D = 0.15        # device-to-device std of the retention time constant

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--arch", default="7T1C", choices=["6T1C", "7T1C"])
ap.add_argument("--nonideal", default="variation,disturbance",
                help="comma list of variation, retention, disturbance; or 'ideal'")
ap.add_argument("--vov-median", type=float, default=None, help="median overdrive voltage (V), default 0.35")
ap.add_argument("--vov-std", type=float, default=None, help="std of the overdrive voltage (V), default 0.043")
ap.add_argument("--lam", type=float, default=None, help="channel-length modulation (1/V), default 0.0324")
ap.add_argument("--lam-rel-std", type=float, default=None, help="7T1C relative std of lambda_pot/dep, default 1/30")
ap.add_argument("--step-median", type=float, default=None, help="states of the median cell, default 1000")
ap.add_argument("--gamma", type=float, default=0.000112, help="write-disturbance decay per node-group transition")
ap.add_argument("--tau-min", type=float, default=None, help="retention time constant (min), needed with retention")
ap.add_argument("--dt-batch", type=float, default=None, help="device time of one mini-batch (s), needed with retention")
ap.add_argument("--prog-noise", type=float, default=0.085, help="std of the core programming noise (weight units)")
ap.add_argument("--fast-lr", type=float, default=0.04, help="learning rate of the auxiliary array")
ap.add_argument("--lr", type=float, default=0.01, help="transfer learning rate")
ap.add_argument("--iters", type=int, default=5, help="passes over layers 2-17")
ap.add_argument("--frac", type=float, default=1.0, help="fraction of an epoch per layer (1.0 in the paper)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ckpt", default="data/ckpt/resnet18_cifar10.pt", help="relative to the repository root")
ap.add_argument("--out", default=None, help="output JSON file, relative to the repository root")
args = ap.parse_args()

nonideal = set() if args.nonideal == "ideal" else set(args.nonideal.split(","))
if nonideal - {"variation", "retention", "disturbance"}:
    raise SystemExit(f"unknown --nonideal item(s): {sorted(nonideal - {'variation', 'retention', 'disturbance'})}")
if "retention" in nonideal and (args.tau_min is None or args.dt_batch is None):
    raise SystemExit("retention needs --tau-min and --dt-batch")
torch.manual_seed(args.seed)
dev = "cuda"


def aux_device():
    d = CapSynapseDevice(shared_transistor=(args.arch == "7T1C"), dw_min_std=0.0,
                         up_down=0.0, up_down_dtod=0.0, dw_min_dtod=0.0,
                         w_max_dtod=0.0, w_min_dtod=0.0, reset_std=0.0, diffusion=0.0)
    if "variation" not in nonideal:      # median cell
        d.vov_std = 0.0; d.w_rel_std = 0.0; d.bound_std = 0.0; d.lam_rel_std = 0.0
    for k in ("vov_median", "vov_std", "lam_rel_std", "lam", "step_median"):
        v = getattr(args, k)
        if v is not None:
            setattr(d, k, v)
    if "retention" in nonideal:          # decay toward V_cap = 0, applied per mini-batch
        d.lifetime = args.tau_min * 60.0 / args.dt_batch
        d.lifetime_dtod = TAU_D2D
    if "disturbance" in nonideal:
        d.hs_gamma = args.gamma
        d.hs_pulse_scheduling = True
    return d


def core_device():
    g = CORE_NL / 2.0
    return LinearStepDevice(dw_min=CORE_DW_MIN, dw_min_std=CORE_C2C, mult_noise=True, dw_min_dtod=CORE_D2D,
                            gamma_up=g, gamma_down=g, gamma_up_dtod=CORE_D2D * g, gamma_down_dtod=CORE_D2D * g,
                            w_max=1.0, w_min=-1.0, w_max_dtod=CORE_D2D, w_min_dtod=CORE_D2D,
                            up_down=0.0, up_down_dtod=0.01, reset_std=0.0)


def upd_params():                        # one pulse slot per mat-vec
    u = UpdateParameters()
    u.desired_bl = 1; u.update_bl_management = False; u.fixed_bl = True; u.update_management = True
    return u


rpu = UnitCellRPUConfig(
    device=TransferCompound(
        unit_cell_devices=[aux_device(), core_device()],
        gamma=0.0,                       # forward and backward use the core only
        transfer_every=1.0 / BATCH, units_in_mbatch=True,      # one row per training image
        fast_lr=args.fast_lr, n_reads_per_transfer=1, transfer_lr=1.0, scale_transfer_lr=True,
        transfer_forward=IOParameters(noise_management=NoiseManagementType.NONE,
                                      bound_management=BoundManagementType.NONE, out_noise=OUT_NOISE),
        transfer_update=upd_params(), construction_seed=args.seed + 1),
    forward=IOParameters(out_noise=OUT_NOISE), backward=IOParameters(out_noise=OUT_NOISE),
    update=upd_params(),
    mapping=MappingParameter(weight_scaling_omega=OMEGA, learn_out_scaling=False, digital_bias=True),
)

tr, te = loaders(root=os.path.join(ROOT, "data/cifar"), batch_size=BATCH)
model = resnet18_cifar().to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, args.ckpt), map_location=dev))
names = [n for n, _ in conv_layers(model)]           # 17 convs, index 0 = stem
evaluate(model, te, device=dev)                      # first pass discarded (advances the RNG state)
sw_acc = evaluate(model, te, device=dev)

# pulse-scheduling group = kernel positions of one image (H_out * W_out) per conv layer
_pos = {}
if args.gamma > 0 and "disturbance" in nonideal:
    _h = [m.register_forward_hook(lambda mod, inp, out, n=n: _pos.__setitem__(n, out.shape[-2] * out.shape[-1]))
          for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=dev))
    for _x in _h:
        _x.remove()


def _per_layer_rpu(name, mod, cfg):
    if name in _pos:
        cfg.device.unit_cell_devices[0].hs_ps_group = float(_pos[name])
    return cfg


model = convert_to_analog(model, rpu, conversion_map={nn.Conv2d: AnalogConv2d},
                          specific_rpu_config_fun=_per_layer_rpu,
                          exclude_modules=["conv1", "fc"]).to(dev)
for m in model.analog_modules():                     # write the pretrained weights to the core
    for t in m.analog_tiles():
        program_core_noise(t, args.prog_noise)
mapped_acc = evaluate(model, te, device=dev)
print(f"software {sw_acc:.2f}%  mapped {mapped_acc:.2f}%", flush=True)

assert len(names) == 17, names
steps = list(enumerate(names[1:], start=2))          # (layer number, module name), layers 2-17
last_layer = steps[-1][0]
mods = dict(model.named_modules())
extra = {}                                           # 1x1 shortcut convs, trained with their block's conv1
for n in mods:
    if n.endswith("downsample.0"):
        extra.setdefault(n.rsplit(".", 2)[0] + ".conv1", []).append(n)

opt = AnalogSGD(model.parameters(), lr=args.lr, momentum=0.0)
opt.regroup_param_groups(model)
crit = nn.CrossEntropyLoss()
log = {"args": vars(args), "software": sw_acc, "mapped": mapped_acc,
       "acc": [], "layer": [], "iter_end": [], "ps_group": dict(_pos)}
out = os.path.join(ROOT, args.out or f"results_rerun/{args.arch}_{args.nonideal.replace(',', '+')}_s{args.seed}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
n_batches = max(1, int(args.frac * len(tr)))
t0 = time.time()
for it in range(args.iters):
    for k, name in steps:
        train_set = {name, *extra.get(name, [])}
        for n, p in model.named_parameters():
            p.requires_grad_(any(n.startswith(t + ".") for t in train_set))
        # AIHWKIT updates every tile that backward passed through, so drop the context of the others
        train_tiles = set(id(t) for name_ in train_set for t in mods[name_].analog_tiles())
        other_tiles = [t for m_ in model.analog_modules() for t in m_.analog_tiles() if id(t) not in train_tiles]
        model.eval()                                 # batch-norm statistics stay frozen
        for b, (x, y) in enumerate(tr):
            if b >= n_batches:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            for t in other_tiles:
                t.analog_ctx.reset()
            opt.step()
        acc = evaluate(model, te, device=dev)
        log["acc"].append(acc); log["layer"].append(f"it{it+1}_L{k}"); log["iter_end"].append(k == last_layer)
        print(f"iter {it+1} layer {k:2d} ({name}) acc {acc:.2f}%  [{(time.time()-t0)/60:.1f} min]", flush=True)
        json.dump(log, open(out, "w"), indent=1)
json.dump(log, open(out, "w"), indent=1)
print("saved", out)
