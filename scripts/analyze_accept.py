"""Error analysis of relaxed vs exact acceptance from --log_detail runs (results/analysis/{spec,relaxed_r9}/*.json).
Per task: success; for relaxed: share of accepted draft tokens that differ from the verifier's argmax ("relaxed
accepts"), split by action dim, by success/failure episode, and by gripper contact; verifier probability of those
tokens. Usage: python scripts/analyze_accept.py [results/analysis]"""
import glob, json, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results/analysis"
DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]


def load(path):
    try:
        return json.load(open(path))["episodes"]
    except Exception:
        return None  # file being rewritten by a running job


def tokens(eps):
    """Yield (episode_success, contact, token-dict) for decided action tokens (dim 0..6)."""
    for e in eps:
        for det, st in zip(e["detail"], e["state"]):
            for d in det:
                if d["dim"] <= 6:
                    yield e["success"], st["contact"], d


for f in sorted(glob.glob(f"{R}/relaxed_r9/*.json")):
    name = f.split("/")[-1][:-5]
    rel, ex = load(f), load(f"{R}/spec/{name}.json")
    if not rel or not ex:
        continue
    n = min(len(rel), len(ex))
    print(f"\n== {name}: success exact {sum(e['success'] for e in ex[:n])}/{n}, r9 {sum(e['success'] for e in rel[:n])}/{n}")
    T = list(tokens(rel))
    acc = [(s, c, d) for s, c, d in T if d["src"] == "accept"]
    relax = [(s, c, d) for s, c, d in acc if d["draft"] != d["verif"]]
    print(f"  accepted drafts {len(acc)}, of which relaxed (draft != argmax) {len(relax)} ({100 * len(relax) / max(len(acc), 1):.1f}%)")
    print("  relaxed share by dim: " + " ".join(
        f"{DIMS[k]} {100 * sum(d['dim'] == k for _, _, d in relax) / max(sum(d['dim'] == k for _, _, d in acc), 1):.0f}%"
        for k in range(7)))
    for lab, sel in [("success eps", lambda s, c: s), ("failed eps", lambda s, c: not s),
                     ("contact", lambda s, c: c), ("no contact", lambda s, c: c is False)]:
        a = [d for s, c, d in acc if sel(s, c)]
        r = [d for d in a if d["draft"] != d["verif"]]
        if a:
            pd = np.array([d["p_draft"] for d in r]) if r else np.array([np.nan])
            print(f"  {lab:12} relaxed {100 * len(r) / len(a):5.1f}% of accepts | mean bin dist "
                  f"{np.mean([abs(d['draft'] - d['verif']) for d in r]) if r else 0:4.1f} | p_draft<1e-3: "
                  f"{100 * np.mean(pd < 1e-3):5.1f}% | verifier p_top {np.mean([d['p_top'] for d in a]):.3f}")
