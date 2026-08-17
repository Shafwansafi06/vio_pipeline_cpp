#!/usr/bin/env python3
"""Print the DOD-vs-official table from a reeval run directory."""
import json, os, sys, glob

R = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/dod/docs/results/trajectories/reeval_20260817/runs"

SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
        "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
        "KAIST_circle", "KAIST_infinity"]


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def wall(p):
    try:
        with open(p) as f:
            return float(f.read().split()[-1])
    except Exception:
        return float("nan")


hdr = ("sequence", "DOD ATE", "OV ATE", "ratio", "DOD p95", "OV p95",
       "DODpath", "OVpath", "GTpath", "DODn", "OVn", "DODs", "OVs")
print("%-16s %8s %8s %6s %8s %8s %8s %8s %8s %6s %6s %6s %6s" % hdr)
rows = []
for s in SEQS:
    d = load(f"{R}/{s}_eval.json")
    o = load(f"{R}/{s}_ov_eval.json")
    if not d or not o:
        print("%-16s  MISSING (dod=%s ov=%s)" % (s, bool(d), bool(o)))
        continue
    ratio = d["ate_rmse_m"] / o["ate_rmse_m"]
    rows.append((s, d, o, ratio))
    print("%-16s %8.4f %8.4f %6.2f %8.3f %8.3f %8.1f %8.1f %8.1f %6d %6d %6.0f %6.0f" % (
        s, d["ate_rmse_m"], o["ate_rmse_m"], ratio,
        d["ate_p95_m"], o["ate_p95_m"],
        d["estimated_path_length_m"], o["estimated_path_length_m"],
        d["groundtruth_path_length_m"],
        d["associated_samples"], o["associated_samples"],
        wall(f"{R}/{s}_dod_wall.txt"), wall(f"{R}/{s}_ov_wall.txt")))

if rows:
    import statistics
    print()
    print("DOD better on %d/%d sequences" % (sum(1 for r in rows if r[3] < 1.0), len(rows)))
    print("mean ATE   DOD %.4f   OV %.4f" % (
        statistics.mean(r[1]["ate_rmse_m"] for r in rows),
        statistics.mean(r[2]["ate_rmse_m"] for r in rows)))
    print("median ratio %.3f  (geometric mean %.3f)" % (
        statistics.median(r[3] for r in rows),
        statistics.geometric_mean(r[3] for r in rows)))
