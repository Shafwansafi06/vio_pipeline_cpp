#!/usr/bin/env python3
"""Render DOD-vs-official trajectory comparisons for every benchmarked sequence,
plus a combined grid and an accuracy/timing summary.

Both estimates are SE3-aligned to the same ground truth by the same Umeyama fit
scripts/evaluate_trajectory.py uses to score them, so the ATE printed on each
panel is the scored number.

    plot_all_comparisons.py RUNS_DIR OUT_DIR
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_trajectory import associate, load_positions, rigid_align

GT_COLOR = "#666666"
DOD_COLOR = "#d62728"
OV_COLOR = "#1f77b4"

# label -> (dod prefix, official estimate stem, dataset)
#
# Matches the reeval_20260817/ naming, where DOD, official and ground truth
# all share one base name (SEQ_estimate.csv, SEQ_ov_estimate.csv,
# SEQ_groundtruth.csv) -- both estimators scored off the identical rosbag
# transport and the identical ground truth, per CLAUDE.md's same-transport
# rule. Previous entries here (OK_MH_01/ov_mh01/K_circle/...) pointed at an
# older, ASL-transport run directory and are stale; V1_* under that naming
# never had valid ground truth (see the V1 groundtruth-topic fix in
# 1ec3389), so any plot made with the old tuple silently miscompares.
SEQUENCES = [
    ("MH_01_easy", "MH_01_easy", "MH_01_easy_ov", "EuRoC"),
    ("MH_02_easy", "MH_02_easy", "MH_02_easy_ov", "EuRoC"),
    ("MH_03_medium", "MH_03_medium", "MH_03_medium_ov", "EuRoC"),
    ("MH_04_difficult", "MH_04_difficult", "MH_04_difficult_ov", "EuRoC"),
    ("MH_05_difficult", "MH_05_difficult", "MH_05_difficult_ov", "EuRoC"),
    ("V1_01_easy", "V1_01_easy", "V1_01_easy_ov", "EuRoC"),
    ("V1_02_medium", "V1_02_medium", "V1_02_medium_ov", "EuRoC"),
    ("V1_03_difficult", "V1_03_difficult", "V1_03_difficult_ov", "EuRoC"),
    ("KAIST circle", "KAIST_circle", "KAIST_circle_ov", "KAIST"),
    ("KAIST infinity", "KAIST_infinity", "KAIST_infinity_ov", "KAIST"),
]


def aligned(estimate_path, truth_times, truth_positions):
    times, positions = load_positions(Path(estimate_path))
    order = np.argsort(times)
    times, positions = times[order], positions[order]
    est_idx, truth_idx = associate(times, truth_times, 0.03)
    if len(est_idx) < 10:
        return None, None
    rotation, translation = rigid_align(positions[est_idx], truth_positions[truth_idx])
    fitted = (rotation @ positions[est_idx].T).T + translation
    error = np.linalg.norm(fitted - truth_positions[truth_idx], axis=1)
    return fitted, error


def dod_timing(path):
    """-> (tracking, estimator) mean ms/frame from the runner's timing CSV."""
    rows = np.genfromtxt(path, delimiter=",", names=True)
    return float(np.mean(rows["tracking_ms"])), float(np.mean(rows["estimator_ms"]))


def official_timing(path):
    """OpenVINS timing columns: ts,tracking,prop,msckf,slam,slam-delay,marg,total."""
    data = np.loadtxt(path, delimiter=",", comments="#")
    return float(np.mean(data[:, 1]) * 1000.0), float(np.mean(data[:, 7] - data[:, 1]) * 1000.0)


def panel(axis, truth, dod, ov, label):
    axis.plot(truth[:, 0], truth[:, 1], color=GT_COLOR, linewidth=2.6, alpha=0.4, label="Ground truth")
    if dod[0] is not None:
        axis.plot(dod[0][:, 0], dod[0][:, 1], color=DOD_COLOR, linewidth=1.2, label="DOD (ours)")
    if ov[0] is not None:
        axis.plot(ov[0][:, 0], ov[0][:, 1], color=OV_COLOR, linewidth=1.2, alpha=0.85, label="Official OpenVINS")
    dod_rmse = np.sqrt(np.mean(dod[1] ** 2)) if dod[1] is not None else float("nan")
    ov_rmse = np.sqrt(np.mean(ov[1] ** 2)) if ov[1] is not None else float("nan")
    winner = "DOD" if dod_rmse < ov_rmse else "OpenVINS"
    axis.set_title(f"{label}\nDOD {dod_rmse:.4f} m   OpenVINS {ov_rmse:.4f} m   ({winner} ahead)", fontsize=10)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.tick_params(labelsize=8)
    return dod_rmse, ov_rmse


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    grid, axes = plt.subplots(2, 5, figsize=(26, 11))
    summary = []
    for index, (label, dod_prefix, ov_stem, family) in enumerate(SEQUENCES):
        gt_path = runs / f"{dod_prefix}_groundtruth.csv"
        truth_times, truth_positions = load_positions(gt_path)
        order = np.argsort(truth_times)
        truth_times, truth_positions = truth_times[order], truth_positions[order]

        dod = aligned(runs / f"{dod_prefix}_estimate.csv", truth_times, truth_positions)
        ov = aligned(runs / f"{ov_stem}_estimate.csv", truth_times, truth_positions)

        figure, axis = plt.subplots(figsize=(7.5, 7.0))
        dod_rmse, ov_rmse = panel(axis, truth_positions, dod, ov, label)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.legend(loc="best", fontsize=9)
        figure.tight_layout()
        name = label.replace(" ", "_").lower()
        figure.savefig(out / f"{name}.png", dpi=150)
        plt.close(figure)

        panel(axes.flat[index], truth_positions, dod, ov, label)
        if index == 0:
            axes.flat[index].legend(loc="best", fontsize=8)

        dod_timing_path = runs / f"{dod_prefix}_timing.csv"
        ov_timing_path = runs / f"{ov_stem}_timing.txt"
        timings = ()
        if dod_timing_path.exists() and ov_timing_path.exists():
            timings = dod_timing(dod_timing_path) + official_timing(ov_timing_path)
        summary.append((label, family, dod_rmse, ov_rmse, timings))

    grid.suptitle("DOD vs official OpenVINS — all benchmarked sequences (SE3-aligned to the same ground truth)",
                  fontsize=15)
    grid.tight_layout()
    grid.savefig(out / "all_sequences.png", dpi=130)
    plt.close(grid)

    print(f"{'sequence':<18}{'DOD ATE':>10}{'OV ATE':>10}{'ratio':>8}   "
          f"{'DOD trk':>8}{'DOD est':>8}{'OV trk':>8}{'OV est':>8}{'DOD tot':>9}{'OV tot':>9}")
    for label, family, dod_rmse, ov_rmse, timings in summary:
        line = f"{label:<18}{dod_rmse:>10.4f}{ov_rmse:>10.4f}{dod_rmse / ov_rmse:>8.2f}"
        if timings:
            dt, de, ot, oe = timings
            line += f"   {dt:>8.3f}{de:>8.3f}{ot:>8.3f}{oe:>8.3f}{dt + de:>9.3f}{ot + oe:>9.3f}"
        print(line)
    print(f"\nwrote {len(SEQUENCES)} panels + all_sequences.png to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
