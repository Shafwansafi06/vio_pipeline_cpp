#!/usr/bin/env python3
"""Convert official Open_VINS's _filepath_est state dump to the CSV that
scripts/evaluate_trajectory.py reads, so DOD and official are scored by the
exact same code path against the exact same ground truth.

Official's format is whitespace-separated:
    timestamp(s) qx qy qz qw px py pz vx vy vz bg(3) ba(3) ...
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(src) as fin, open(dst, "w") as fout:
    fout.write("timestamp,px,py,pz,qx,qy,qz,qw\n")
    for line in fin:
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 8:
            continue
        ts = f[0]
        qx, qy, qz, qw = f[1:5]
        px, py, pz = f[5:8]
        fout.write(f"{ts},{px},{py},{pz},{qx},{qy},{qz},{qw}\n")
        n += 1
print(f"wrote {n} poses to {dst}")
