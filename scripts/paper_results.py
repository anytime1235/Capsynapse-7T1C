"""Mean +- s.d. over seeds of every run set in scripts/paper_runs.py.

  python scripts/paper_results.py [--results DIR] [--plot]
"""
import argparse, glob, json, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import paper_runs

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(ROOT, "results"))
ap.add_argument("--plot", action="store_true", help="write figures/<run set>.png")
a = ap.parse_args()

KEYS = ("arch", "nonideal", "fast_lr", "lr", "vov_median", "vov_std", "lam", "lam_rel_std", "step_median",
        "tau_min", "dt_batch")


def key(d):
    return tuple(None if d.get(k) is None else (d[k] if isinstance(d[k], str) else round(float(d[k]), 9)) for k in KEYS)


runs = {}                                  # settings -> {seed: run}
for p in sorted(glob.glob(os.path.join(a.results, "*.json"))):
    d = json.load(open(p))
    if len(d["acc"]) == 16 * d["args"]["iters"]:
        runs.setdefault(key(d["args"]), {})[d["args"]["seed"]] = d

W = max(len(row[0]) for row in paper_runs.PANELS)
print(f"{'run set':{W}s} {'figure':22s} {'n':>2s}  final accuracy (%)  after each pass")
sets = []
for name, fig, arch, nonideal, fa, tl, extra in paper_runs.PANELS:
    f = dict(zip(*[iter(paper_runs.flags(arch, nonideal, fa, tl, extra))] * 2))
    want = {k: f.get("--" + k.replace("_", "-")) for k in KEYS}
    cell = runs.get(key({k: (v if k in ("arch", "nonideal") or v is None else float(v)) for k, v in want.items()}), {})
    if not cell:
        print(f"{name:{W}s} {fig:22s}  0")
        continue
    ends = {s: [x for x, e in zip(d["acc"], d["iter_end"]) if e] for s, d in cell.items()}
    final = [e[-1] for e in ends.values()]
    sd = st.stdev(final) if len(final) > 1 else 0.0
    passes = [st.mean(e[i] for e in ends.values()) for i in range(len(next(iter(ends.values()))))]
    print(f"{name:{W}s} {fig:22s} {len(cell):2d}  {st.mean(final):6.2f} +- {sd:4.2f}       "
          + " ".join(f"{x:5.2f}" for x in passes))
    sets.append((name, fig, cell))

if a.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    for name, fig, cell in sets:
        f, ax = plt.subplots(figsize=(4, 3))
        curves = [[d["mapped"]] + d["acc"] for d in cell.values()]
        for c in curves:
            ax.plot(c, color="0.75", lw=0.8)
        ax.plot([st.mean(v) for v in zip(*curves)], color="C0", lw=1.5)
        ax.set_xticks(range(0, len(curves[0]), 16), ["map"] + [str(i) for i in range(1, len(curves[0]) // 16 + 1)])
        ax.set_xlabel("fine-tuning iteration"); ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 100)
        ax.set_title(f"{fig} ({name})", fontsize=8)
        f.tight_layout(); f.savefig(os.path.join(ROOT, "figures", f"{name}.png"), dpi=200); plt.close(f)
