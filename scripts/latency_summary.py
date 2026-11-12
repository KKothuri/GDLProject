"""Median per-step model latency (ms) per method x suite from results/latency (see latency_bench.sbatch).
First step of each episode is dropped (warmup). Speedup is vs ar on the same suite.
STAT=mean env var switches median -> mean (mean matters for KERV: its forced Kalman steps skip the LLM every 4th action)."""
import glob, json
import os
import numpy as np
agg = np.mean if os.environ.get("STAT") == "mean" else np.median

R = "results/latency"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
METHODS = ["ar", "spec", "relaxed_r9", "kerv", "kerv_noKF"]


def steps(method, suite):
    ms = []
    if method.startswith("kerv"):
        for f in glob.glob(f"{R}/{method}/{suite}/logs/*/*stats.json"):
            for task in json.load(open(f)).values():
                for ep in task.values():
                    ms += [1000 * (b - a) for a, b in ep[1:]]
    else:
        for f in glob.glob(f"{R}/{method}/{suite}_task*.json"):
            for e in json.load(open(f))["episodes"]:
                ms += e["step_ms"][1:]
    return ms


print("method       " + "".join(f"{s:>22}" for s in SUITES))
ar = {s: agg(steps("ar", s)) if steps("ar", s) else np.nan for s in SUITES}
for m in METHODS:
    row = ""
    for s in SUITES:
        x = steps(m, s)
        row += f"{'-':>22}" if not x else f"{agg(x):8.0f} ms ({ar[s] / agg(x):4.2f}x)n={len(x):<4}"
    print(f"{m:12} {row}")
