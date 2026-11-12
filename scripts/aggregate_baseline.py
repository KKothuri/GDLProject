"""Summarize OpenVLA LIBERO baseline success per suite from results/baseline_openvla/<suite>/logs/*.txt."""
import glob, os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results", "baseline_openvla")
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    eps = succ = tasks = 0
    for f in sorted(glob.glob(f"{root}/{suite}/logs/*.txt")):
        text = open(f).read()
        n = len(re.findall(r"^Success: ", text, re.M))
        s = len(re.findall(r"^Success: True", text, re.M))
        if n:
            tasks += 1; eps += n; succ += s
    rate = f"{100 * succ / eps:.1f}%" if eps else "-"
    print(f"{suite:15s} tasks_with_data={tasks:2d} episodes={eps:4d} success={rate}")
