#!/usr/bin/env python3
"""P-01 latency statistics from DOD and OpenVINS per-frame timing files.

DOD  *_timing.csv: timestamp,tracking_ms,estimator_ms,total_ms,observations
OV   *_timing.txt: "# timestamp (sec),tracking,propagation,msckf update,slam
update,slam delayed,re-tri & marg,total" (seconds)

Pure stdlib. Writes latency_table.txt and latency.json next to the inputs.
"""
import glob
import json
import os
import sys

def load_dod(path):
    vals = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5 or parts[0] == "timestamp":
                continue
            try:
                vals.append(float(parts[3]))
            except ValueError:
                continue
    return vals

def load_ov(path):
    vals = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            try:
                vals.append(float(parts[7]) * 1000.0)
            except ValueError:
                continue
    return vals

def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(q / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]

def stats(vals):
    s = sorted(vals)
    p50, p99 = pct(s, 50), pct(s, 99)
    return {
        "n": len(vals),
        "mean_ms": sum(vals) / len(vals),
        "p50_ms": p50,
        "p99_ms": p99,
        "max_ms": s[-1],
        "jitter_ms": p99 - p50,
    }

def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = {}
    for dod_path in sorted(glob.glob(os.path.join(directory, "dod_*_timing.csv"))):
        base = os.path.basename(dod_path)
        seq = base[len("dod_"):-len("_timing.csv")]
        ov_path = f"{directory}/ov_{seq}_timing.txt"
        if not os.path.exists(ov_path):
            continue
        rows[seq] = {"dod": stats(load_dod(dod_path)), "ov": stats(load_ov(ov_path))}

    lines = []
    lines.append("%-9s | %-36s | %-36s | p50 ratio" % ("seq", "DOD mean/p50/p99/max (ms)", "OpenVINS mean/p50/p99/max (ms)"))
    lines.append("-" * 112)
    for seq in sorted(rows):
        d, o = rows[seq]["dod"], rows[seq]["ov"]
        ratio = o["p50_ms"] / d["p50_ms"] if d["p50_ms"] else float("nan")
        lines.append("%-9s | %7.2f %6.2f %6.2f %8.2f n=%d | %7.2f %6.2f %6.2f %8.2f n=%d | %.2fx"
                     % (seq, d["mean_ms"], d["p50_ms"], d["p99_ms"], d["max_ms"], d["n"],
                        o["mean_ms"], o["p50_ms"], o["p99_ms"], o["max_ms"], o["n"],
                        o["p50_ms"] / d["p50_ms"] if d["p50_ms"] else float("nan")))
    text = "\n".join(lines)
    sys.stdout.write(text + "\n")
    with open(os.path.join(directory, "latency_table.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(directory, "latency.json"), "w") as f:
        json.dump(rows, f, indent=1)

if __name__ == "__main__":
    main()
