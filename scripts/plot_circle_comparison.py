#!/usr/bin/env python3
"""Overlay DOD/OpenVINS/Python trajectories against KAIST circle.bag ground truth.

Reads docs/results/*_estimate.csv + dod_circle_groundtruth.csv, aligns each
estimate to ground truth with the same rigid-body fit used for accuracy
scoring (scripts/../docs/results/*_eval.json), and renders begin/middle/end
+ a full top-down view.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "results"


def load_positions(path):
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    t = np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float64)
    p = np.asarray([[float(r["px"]), float(r["py"]), float(r["pz"])] for r in rows], dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(p).all(axis=1)
    order = np.argsort(t[finite])
    return t[finite][order], p[finite][order]


def associate(estimate_times, truth_times, tolerance=0.03):
    right = np.searchsorted(truth_times, estimate_times)
    right = np.clip(right, 0, len(truth_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(truth_times[left] - estimate_times) < np.abs(truth_times[right] - estimate_times)
    indices = np.where(choose_left, left, right)
    valid = np.abs(truth_times[indices] - estimate_times) <= tolerance
    return np.flatnonzero(valid), indices[valid]


def rigid_align(source, target):
    sm, tm = source.mean(axis=0), target.mean(axis=0)
    sc, tc = source - sm, target - tm
    u, _, vt = np.linalg.svd(sc.T @ tc)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    return r, tm - r @ sm


def aligned(name):
    t, p = load_positions(RESULTS / f"{name}_estimate.csv")
    tt, tp = load_positions(RESULTS / "dod_circle_groundtruth.csv")
    ie, it = associate(t, tt)
    r, tr = rigid_align(p[ie], tp[it])
    return t[ie], (r @ p[ie].T).T + tr, tp[it]


def main():
    truth_t, truth_p = load_positions(RESULTS / "dod_circle_groundtruth.csv")
    series = {
        "DOD C++ (ours)": aligned("dod_circle"),
        "OpenVINS (official)": aligned("ov_circle"),
        "Python v2": aligned("py_circle"),
    }
    colors = {"DOD C++ (ours)": "#d62728", "OpenVINS (official)": "#1f77b4", "Python v2": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    ax_full = axes[0, 0]
    ax_full.plot(truth_p[:, 0], truth_p[:, 1], color="black", linewidth=2, label="Ground truth")
    for name, (_, est, _) in series.items():
        ax_full.plot(est[:, 0], est[:, 1], color=colors[name], alpha=0.85, label=name)
    ax_full.set_title("Full trajectory (top-down XY, aligned)")
    ax_full.set_xlabel("x (m)")
    ax_full.set_ylabel("y (m)")
    ax_full.axis("equal")
    ax_full.legend(fontsize=8)
    ax_full.grid(alpha=0.3)

    segments = [("begin", 0.0, 0.12), ("middle", 0.44, 0.56), ("end", 0.88, 1.0)]
    for ax, (label, lo, hi) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], segments):
        n = len(truth_t)
        lo_i, hi_i = int(lo * n), max(int(hi * n), int(lo * n) + 1)
        ax.plot(truth_p[lo_i:hi_i, 0], truth_p[lo_i:hi_i, 1], color="black", linewidth=2, label="Ground truth")
        for name, (est_t, est, _) in series.items():
            t_lo, t_hi = truth_t[lo_i], truth_t[min(hi_i, n - 1)]
            sel = (est_t >= t_lo) & (est_t <= t_hi)
            ax.plot(est[sel, 0], est[sel, 1], color=colors[name], alpha=0.85, label=name)
        ax.set_title(f"{label} segment")
        ax.axis("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("KAIST circle.bag: DOD C++ vs official OpenVINS vs Python v2", fontsize=13)
    fig.tight_layout()
    out = RESULTS.parent / "circle_trajectory_comparison.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
