#!/usr/bin/env python3
"""Side-by-side XY trajectory plot: DOD vs official OpenVINS, both against the
same ground truth and both aligned by the same Umeyama rigid fit that
scripts/evaluate_trajectory.py uses to score them.

    plot_euroc_side_by_side.py DOD_EST OV_EST GROUNDTRUTH OUT.png [title]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_trajectory import associate, load_positions, rigid_align

GT_COLOR = "#444444"
SERIES = [("DOD (ours)", "#d62728"), ("Official OpenVINS", "#1f77b4")]


def aligned(estimate_path, truth_times, truth_positions):
    times, positions = load_positions(Path(estimate_path))
    order = np.argsort(times)
    times, positions = times[order], positions[order]
    est_idx, truth_idx = associate(times, truth_times, 0.03)
    rotation, translation = rigid_align(positions[est_idx], truth_positions[truth_idx])
    fitted = (rotation @ positions[est_idx].T).T + translation
    error = np.linalg.norm(fitted - truth_positions[truth_idx], axis=1)
    return fitted, error


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    dod_path, ov_path, gt_path, out_path = sys.argv[1:5]
    title = sys.argv[5] if len(sys.argv) > 5 else "EuRoC MH_01_easy"

    truth_times, truth_positions = load_positions(Path(gt_path))
    order = np.argsort(truth_times)
    truth_times, truth_positions = truth_times[order], truth_positions[order]

    runs = [aligned(dod_path, truth_times, truth_positions),
            aligned(ov_path, truth_times, truth_positions)]

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)
    for axis, (label, color), (track, error) in zip(axes, SERIES, runs):
        axis.plot(truth_positions[:, 0], truth_positions[:, 1], color=GT_COLOR,
                  linewidth=2.4, alpha=0.45, label="Ground truth (Leica)")
        axis.plot(track[:, 0], track[:, 1], color=color, linewidth=1.3, label=label)
        rmse = float(np.sqrt(np.mean(error ** 2)))
        axis.set_title(f"{label}\nATE RMSE {rmse:.4f} m   "
                       f"median {np.median(error):.3f}   p95 {np.percentile(error, 95):.3f}")
        axis.set_xlabel("x [m]")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("y [m]")

    figure.suptitle(f"{title} — trajectory vs ground truth (SE3-aligned, same evaluator)",
                    fontsize=13)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
