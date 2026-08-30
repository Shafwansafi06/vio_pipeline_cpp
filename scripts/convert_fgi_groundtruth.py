#!/usr/bin/env python3
"""Convert FGI Masala's space-delimited ground truth (t x y z qx qy qz qw)
into the timestamp,px,py,pz,qx,qy,qz,qw CSV scripts/evaluate_trajectory.py
expects. One-shot, no library needed -- ponytail: stdlib csv is enough here.
"""
import csv
import sys

def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.txt OUTPUT.csv", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as src, open(sys.argv[2], "w", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(["timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw"])
        for line in src:
            fields = line.split()
            if len(fields) != 8:
                continue
            writer.writerow(fields)
    return 0

if __name__ == "__main__":
    sys.exit(main())
