"""Settings of the training runs (five seeds per panel).

Every run is scripts/finetune_tt.py with the flags below plus --seed.
"""
SEEDS = (0, 1, 2, 3, 4)

RET6 = ["--tau-min", "775", "--dt-batch", "3.93216e-3"]     # 6T1C retention
RET7 = ["--tau-min", "46.03", "--dt-batch", "6.5536e-3"]    # 7T1C retention
WAFER = ["--vov-median", "0.51", "--vov-std", "0.05", "--lam", "0.05", "--lam-rel-std", "0.05",
         "--step-median", "300"]                            # measured-wafer device (Table S9)

# (name, figure, arch, --nonideal, aux lr (--fast-lr), transfer lr (--lr), extra flags)
PANELS = [
    ("ideal",                            "Fig. S25A",           "7T1C", "ideal",                 0.03, 0.0003, []),
    ("6T1C_retention",                   "Fig. S25B",           "6T1C", "retention",             0.03, 0.0003, RET6),
    ("7T1C_retention",                   "Fig. S25C",           "7T1C", "retention",             0.03, 0.0003, RET7),
    ("disturbance",                      "Fig. S25D",           "7T1C", "disturbance",           0.06, 0.006,  []),
    ("6T1C_variation",                   "Fig. S25E",           "6T1C", "variation",             0.04, 0.01,   []),
    ("7T1C_variation",                   "Fig. S25F",           "7T1C", "variation",             0.03, 0.0003, []),
    ("6T1C_variation_disturbance",       "Fig. 5D, S13A, S25G", "6T1C", "variation,disturbance", 0.04, 0.01,   []),
    ("7T1C_variation_disturbance",       "Fig. 5D, S13B, S25H", "7T1C", "variation,disturbance", 0.24, 0.01,   []),
    ("wafer_6T1C_variation_disturbance", "Fig. S24A",           "6T1C", "variation,disturbance", 0.04, 0.01,   WAFER),
    ("wafer_7T1C_variation_disturbance", "Fig. S24B",           "7T1C", "variation,disturbance", 0.04, 0.01,   WAFER),
]


def tag(name, seed):
    return f"{name}_s{seed}"


def flags(arch, nonideal, fast_lr, lr, extra):
    return ["--arch", arch, "--nonideal", nonideal, "--fast-lr", f"{fast_lr:g}", "--lr", f"{lr:g}", *extra]


def runs():
    """Yield (tag, figure, flags) for every run."""
    for name, fig, arch, nonideal, fa, tl, extra in PANELS:
        for s in SEEDS:
            yield tag(name, s), fig, flags(arch, nonideal, fa, tl, extra) + ["--seed", str(s)]


if __name__ == "__main__":
    for t, fig, f in runs():
        print(f"# {fig}\npython scripts/finetune_tt.py {' '.join(f)} --out results/{t}.json")
