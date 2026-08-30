#!/usr/bin/env python3
"""P-01 CDF figure: per-frame total latency, DOD vs OpenVINS, all ten sequences.

Reads dod_*_timing.csv (total_ms column 4) and ov_*_timing.txt (total column 8,
seconds) from a directory of p01 outputs. Writes latency_cdf.png (2x5 grid).
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOD_COLOR = "#1f6feb"
OV_COLOR = "#d97706"

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

def ecdf(ax, vals, label, color):
    s = sorted(vals)
    n = len(s)
    ys = [i / n for i in range(1, n + 1)]
    ax.step(s, ys, where="post", label=label, color=color, linewidth=1.4)

def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "latency_cdf.png"
    seqs = sorted(
        os.path.basename(p)[len("dod_"):-len("_timing.csv")]
        for p in glob.glob(f"{directory}/dod_*_timing.csv")
    )
    fig, axes = plt.subplots(2, 5, figsize=(19, 7), sharey=True)  # noqa: F821
    for ax, seq in zip(axes.flat, seqs):
        d = load_dod(f"{directory}/dod_{seq}_timing.csv")
        o = load_ov(f"{directory}/ov_{seq}_timing.txt")
        ecdf(ax, d, "DOD (ours)", DOD_COLOR)
        ecdf(ax, o, "OpenVINS", OV_COLOR)
        ax.set_title(seq, fontsize=11)
        ax.set_xscale("log")
        ax.set_xlim(1, 200)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)
        if ax in axes[:, 0]:
            ax.set_ylabel("fraction of frames", fontsize=9)
        if ax in axes[1, :]:
            ax.set_xlabel("per-frame total latency (ms, log)", fontsize=9)
        ax.axvline(50.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    axes[0][0].legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.suptitle("Per-frame end-to-end latency, same rosbag transport, ROS1 Noetic container (OpenCV 4.2.0 / g++ 9.4.0)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    print("wrote", out)

if __name__ == "__main__":
    main()
