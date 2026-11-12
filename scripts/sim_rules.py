"""Offline speed estimate for acceptance rules STRICTER than r=9, from r9 --log_detail logs
(results/analysis/relaxed_r9/*.json). For each logged verify pass, a stricter rule accepts a prefix of the r9-accepted
drafts; new accept length L' = tokens until the first one the rule rejects. Mean tokens/pass (L'+1) gives verify passes
per action ~ 6 / tokens_per_pass (6 tokens after the prefill token), calibrated so the r9 rule reproduces the measured
r9 passes. Latency model from results/profile (A40): ms = FIXED + passes * PER_PASS (vision+prefill+init draft+other,
and verify+draft per pass). This is a LOWER bound on tokens/pass for strict rules (their own trees would hold better paths;
check: it predicts exact at 4.96 passes vs 4.3 measured, ~15% pessimistic), and it says nothing about success. Usage: python scripts/sim_rules.py [results/analysis/relaxed_r9]"""
import glob, json, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "results/analysis/relaxed_r9"
FIXED, PER_PASS = 119.0, 45.0   # ms; profile: vision 19 + prefill 72 + init draft 9 + other ~19; verify 36 + draft 9
MEASURED_R9_PASSES = 3.2        # profile mean verify passes per action for r9 (n_llm_calls - 1)


def passes(ep):
    """Yield (step, list of accepted-draft token dicts) per verify pass, dims 1..6 only."""
    for t, det in enumerate(ep["detail"]):
        cur = []
        for d in det:
            if d["src"] == "accept":
                cur.append(d)
            elif d["src"] == "bonus":
                yield t, [x for x in cur if x["dim"] <= 6]
                cur = []


def rule_bin(r):
    return lambda d, st: abs(d["draft"] - d["verif"]) <= r


def rule_conf(tau):
    return lambda d, st: d["draft"] == d["verif"] or d["p_draft"] >= tau


def rule_perdim(rz):
    return lambda d, st: abs(d["draft"] - d["verif"]) <= (rz if d["dim"] == 2 else 9)


def rule_budget(B):
    def f(d, st):
        k, e = d["dim"], d["draft"] - d["verif"]
        if e == 0:
            return True
        if abs(st.get(k, 0) + e) <= B:
            st[k] = st.get(k, 0) + e
            return True
        return False
    return f


RULES = {"r9 (logged)": rule_bin(9), "exact": rule_bin(0), "r3": rule_bin(3), "r5": rule_bin(5),
         "conf p>=1e-4": rule_conf(1e-4), "conf p>=1e-3": rule_conf(1e-3), "conf p>=1e-2": rule_conf(1e-2),
         "z r3 / others r9": rule_perdim(3), "budget B=10": rule_budget(10), "budget B=20": rule_budget(20),
         "budget B=40": rule_budget(40)}

eps = []
for f in sorted(glob.glob(f"{D}/*.json")):
    try:
        eps += json.load(open(f))["episodes"]
    except Exception:
        pass
res = {}
for name, rule in RULES.items():
    tok, relaxed_kept = [], 0
    for ep in eps:
        st = {}  # per-episode rule state (budget)
        for _, acc in passes(ep):
            L = 0
            for d in acc:
                if not rule(d, st):
                    break
                L += 1
                relaxed_kept += d["draft"] != d["verif"]
            tok.append(L + 1)
    res[name] = (np.mean(tok), relaxed_kept)
scale = MEASURED_R9_PASSES / (6 / res["r9 (logged)"][0])
print(f"{len(eps)} episodes. est ms = {FIXED:.0f} + passes*{PER_PASS:.0f}; passes calibrated to r9 = {MEASURED_R9_PASSES}")
print(f"{'rule':20} tokens/pass  est passes  est ms  relaxed tokens kept")
for name, (tp, rk) in res.items():
    p = scale * 6 / tp
    print(f"{name:20} {tp:10.2f} {p:11.2f} {FIXED + p * PER_PASS:7.0f}  {100 * rk / max(res['r9 (logged)'][1], 1):6.1f}%")
