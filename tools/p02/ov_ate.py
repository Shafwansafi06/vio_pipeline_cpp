#!/usr/bin/env python3
"""Convert OpenVINS save_total_state estimate txt -> evaluate CSV, then print ATEs.

State txt columns: timestamp q(x y z w) p(x y z) v bg ba ...
evaluate_trajectory.py wants: timestamp,px,py,pz,qx,qy,qz,qw
"""
import csv
import glob
import json
import os
import subprocess
import sys

EVAL = "/workspace/acv/head/scripts/evaluate_trajectory.py"
OVDIR = "/workspace/acv/p04/ovodom"
P01 = "/workspace/acv/p01"
P04 = "/workspace/acv/p04"

def convert(src, dst):
    with open(src) as f, open(dst, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw"])
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t, qx, qy, qz, qw, px, py, pz = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]
            w.writerow([t, px, py, pz, qx, qy, qz, qw])

GT = {
    "MH_01": f"{P01}/dod_MH_01_groundtruth.csv",
    "MH_02": f"{P01}/dod_MH_02_groundtruth.csv",
    "MH_03": f"{P01}/dod_MH_03_groundtruth.csv",
    "MH_04": f"{P01}/dod_MH_04_groundtruth.csv",
    "MH_05": f"{P01}/dod_MH_05_groundtruth.csv",
    "V1_01": f"{P04}/asl_V1_01_groundtruth.csv",
    "V1_02": f"{P04}/asl_V1_02_groundtruth.csv",
    "V1_03": f"{P04}/asl_V1_03_groundtruth.csv",
    "circle": f"{P01}/dod_circle_groundtruth.csv",
    "infinite": f"{P01}/dod_infinite_groundtruth.csv",
}

print("%-9s %-10s" % ("seq", "OV_ATE"))
for seq in sorted(GT):
    src = f"{OVDIR}/ov_{seq}_est.txt"
    if not os.path.exists(src):
        print("%-9s missing" % seq)
        continue
    csvp = f"{OVDIR}/ov_{seq}_est.csv"
    convert(src, csvp)
    r = subprocess.run(["python3", EVAL, csvp, GT[seq]], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
        print("%-9s %-10.4f" % (seq, d["ate_rmse_m"]))
    except Exception:
        print("%-9s EVAL_FAIL" % seq)
