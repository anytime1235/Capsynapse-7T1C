# Capsynapse-7T1C

Simulation code for

> J. Won, J. Kang, J. Park, C. Joe, Y. Roh, M. Kang, H.-M. Lee, and S. Kim,
> "Variation-Resilient Oxide-Semiconductor Capacitive Synaptic Device for Reliable Training in
> Analog In-Memory Computing," *Device* (2026).

The 6T1C / 7T1C capacitive synapse model (Supplemental Note S4) is implemented as a device of
[AIHWKIT](https://github.com/IBM/aihwkit) 1.1.0 (`CapSynapseDevice`), with device-to-device
variation, retention and write disturbance. ResNet-18 pretrained on CIFAR-10 is fine-tuned layer by
layer with a core NVM array and a 6T1C or 7T1C auxiliary array (Tiki-Taka).

| | |
|---|---|
| `aihwkit_ext/` | `CapSynapseDevice` (C++/CUDA), patch registering it in AIHWKIT 1.1.0, build script |
| `varsim/` | compact model in NumPy, ResNet-18 and CIFAR-10 loaders |
| `scripts/` | training (`finetune_tt.py`), run settings (`paper_runs.py`), tests |
| `results/` | runs of Fig. 5D, S13, S24 and S25 (5 seeds each) |
| `data/ckpt/` | pretrained ResNet-18 (94.25 %) |

## Installation

Linux with a CUDA GPU, the CUDA toolkit, a C++ compiler, OpenBLAS and the Python headers
(Ubuntu: `sudo apt install build-essential git libopenblas-dev python3-dev python3-venv`).

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
ARCH=90 ./aihwkit_ext/install.sh        # ARCH: CUDA compute capability, e.g. 80, 86, 89, 90, 100
.venv/bin/python scripts/check_capsynapse.py
.venv/bin/python scripts/check_disturbance.py
```

## Usage

```bash
# one run
.venv/bin/python scripts/finetune_tt.py --arch 7T1C --nonideal variation,disturbance \
    --fast-lr 0.24 --lr 0.01 --seed 0 --out results_rerun/7T1C_s0.json

# all runs in results/ (settings in scripts/paper_runs.py)
.venv/bin/python scripts/run_paper.py --results results_rerun --gpus 0,1 --jobs 4

# mean +- s.d. over seeds (results/ by default; --plot writes figures/)
.venv/bin/python scripts/paper_results.py
.venv/bin/python scripts/paper_results.py --results results_rerun --plot

# compact model statistics (Fig. 5B, Table S10)
.venv/bin/python scripts/fig5b_symmetry.py
.venv/bin/python scripts/compact_model_stats.py
```

CIFAR-10 is downloaded to `data/cifar/` on first use. A run takes a few hours on a recent GPU.

## License

MIT. AIHWKIT is distributed by IBM under the MIT license.
