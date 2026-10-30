"""Run all training runs of scripts/paper_runs.py.

  python scripts/run_paper.py --results results_rerun --gpus 0,1 --jobs 4
  python scripts/run_paper.py --dry-run

Runs already complete in the output directory are skipped.
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import paper_runs

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(ROOT, "results_rerun"), help="output directory")
ap.add_argument("--gpus", default="0", help="comma list of GPU indices")
ap.add_argument("--jobs", type=int, default=1, help="concurrent runs")
ap.add_argument("--only", default=None, help="run only tags starting with this prefix")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
a.results = os.path.abspath(a.results)


def finished(tag):
    try:
        d = json.load(open(os.path.join(a.results, f"{tag}.json")))
    except Exception:
        return False
    return len(d.get("acc", [])) >= 16 * d["args"]["iters"]


todo = [(t, f) for t, _, f in paper_runs.runs() if (a.only is None or t.startswith(a.only)) and not finished(t)]
print(f"{len(todo)} run(s) to do")
gpus = a.gpus.split(",")
os.makedirs(a.results, exist_ok=True)
running = []
for i, (tag, f) in enumerate(todo):
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "finetune_tt.py"), *f,
           "--out", os.path.relpath(os.path.join(a.results, f"{tag}.json"), ROOT)]
    gpu = gpus[i % len(gpus)]
    if a.dry_run:
        print(f"CUDA_VISIBLE_DEVICES={gpu}", " ".join(cmd))
        continue
    while sum(p.poll() is None for p in running) >= a.jobs:
        time.sleep(30)
    print(f"[start] {tag} on GPU {gpu}", flush=True)
    log = open(os.path.join(a.results, f"{tag}.log"), "w")
    running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)))
    time.sleep(60)
for p in running:
    p.wait()
