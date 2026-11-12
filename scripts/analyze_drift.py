"""Drift check on paired episodes (same init state) from results/analysis/{spec,relaxed_r9}/*.json (--log_detail runs).
For each task: outcome pairs (exact/r9 success), step where r9's end-effector first departs > 2 cm from exact's,
and r9's accumulated signed bin error (sum of draft - verifier over accepted tokens) per dim, in failed vs
successful r9 episodes. CPU only. Usage: python scripts/analyze_drift.py [results/analysis]"""
import glob, json, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results/analysis"
DIMS = ["y", "z", "roll", "pitch", "yaw"]  # dims 1..5 (x is always from prefill, gripper is binarized)


def load(p):
    try:
        return json.load(open(p))["episodes"]
    except Exception:
        return None


def cum_err(ep):
    """|sum of signed (draft - verifier) bin errors| per dim 1..5 over the episode, and max over time of |running sum|."""
    run = np.zeros(5); peak = np.zeros(5)
    for det in ep["detail"]:
        for d in det:
            if d["src"] == "accept" and 1 <= d["dim"] <= 5:
                run[d["dim"] - 1] += d["draft"] - d["verif"]
        peak = np.maximum(peak, np.abs(run))
    return np.abs(run), peak


allf, alls = [], []
for f in sorted(glob.glob(f"{R}/relaxed_r9/*.json")):
    name = f.split("/")[-1][:-5]
    rel, ex = load(f), load(f"{R}/spec/{name}.json")
    if not rel or not ex:
        continue
    n = min(len(rel), len(ex))
    pairs = {"both": 0, "exact_only": 0, "r9_only": 0, "neither": 0}
    dep = {"exact_only": [], "both": []}
    for a, b in zip(ex[:n], rel[:n]):
        k = "both" if a["success"] and b["success"] else "exact_only" if a["success"] else "r9_only" if b["success"] else "neither"
        pairs[k] += 1
        pa = np.array([s["eef_pos"] for s in a["state"]]); pb = np.array([s["eef_pos"] for s in b["state"]])
        m = min(len(pa), len(pb)); dist = np.linalg.norm(pa[:m] - pb[:m], axis=1)
        idx = np.argmax(dist > 0.02) if (dist > 0.02).any() else m
        if k in dep:
            dep[k].append(idx)
    fail = [cum_err(e)[1] for e in rel[:n] if not e["success"]]
    succ = [cum_err(e)[1] for e in rel[:n] if e["success"]]
    allf += fail; alls += succ
    print(f"\n== {name} (n={n}) pairs {pairs}")
    print(f"  step where r9 departs >2cm from exact: exact_only-success pairs median {np.median(dep['exact_only']) if dep['exact_only'] else '-'}"
          f", both-success pairs median {np.median(dep['both']) if dep['both'] else '-'}")
    if fail and succ:
        print("  peak |running signed bin error| (r9), failed vs success: " + " ".join(
            f"{DIMS[k]} {np.mean([x[k] for x in fail]):.0f}/{np.mean([x[k] for x in succ]):.0f}" for k in range(5)))
if allf and alls:
    print("\nALL tasks peak |running error| failed/success: " + " ".join(
        f"{DIMS[k]} {np.mean([x[k] for x in allf]):.0f}/{np.mean([x[k] for x in alls]):.0f}" for k in range(5)))
