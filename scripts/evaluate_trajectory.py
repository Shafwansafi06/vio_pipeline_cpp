#!/usr/bin/env python3
"""Evaluate a VIO CSV against PoseStamped ground truth without trajectory libraries."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_positions(path: Path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    positions = np.asarray(
        [[float(row["px"]), float(row["py"]), float(row["pz"])] for row in rows],
        dtype=np.float64,
    )
    finite = np.isfinite(timestamps) & np.isfinite(positions).all(axis=1)
    return timestamps[finite], positions[finite]


def associate(estimate_times, truth_times, tolerance):
    right = np.searchsorted(truth_times, estimate_times)
    right = np.clip(right, 0, len(truth_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(truth_times[left] - estimate_times) < np.abs(
        truth_times[right] - estimate_times
    )
    indices = np.where(choose_left, left, right)
    valid = np.abs(truth_times[indices] - estimate_times) <= tolerance
    return np.flatnonzero(valid), indices[valid]


def rigid_align(source, target):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def percentile(values, quantile):
    return float(np.percentile(values, quantile)) if len(values) else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("estimate", type=Path)
    parser.add_argument("groundtruth", type=Path)
    parser.add_argument("--time-offset", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()

    estimate_times, estimate_positions = load_positions(args.estimate)
    truth_times, truth_positions = load_positions(args.groundtruth)
    estimate_indices, truth_indices = associate(
        estimate_times + args.time_offset, truth_times, args.tolerance
    )
    if len(estimate_indices) < 10:
        raise SystemExit(f"only {len(estimate_indices)} timestamp associations")
    estimate_matched = estimate_positions[estimate_indices]
    truth_matched = truth_positions[truth_indices]
    rotation, translation = rigid_align(estimate_matched, truth_matched)
    estimate_aligned = (rotation @ estimate_matched.T).T + translation
    errors = np.linalg.norm(estimate_aligned - truth_matched, axis=1)
    segments = {}
    for name, fraction in (("begin", 0.0), ("middle", 0.5), ("end", 1.0)):
        index = min(len(errors) - 1, int(round(fraction * (len(errors) - 1))))
        segments[name] = {
            "timestamp": float(estimate_times[estimate_indices[index]]),
            "estimate_aligned": estimate_aligned[index].tolist(),
            "groundtruth": truth_matched[index].tolist(),
            "error_m": float(errors[index]),
        }
    step = max(1, int(round(len(errors) / 100.0)))
    relative_error = np.linalg.norm(
        (estimate_aligned[step:] - estimate_aligned[:-step])
        - (truth_matched[step:] - truth_matched[:-step]), axis=1
    )
    result = {
        "estimate_samples": int(len(estimate_times)),
        "groundtruth_samples": int(len(truth_times)),
        "associated_samples": int(len(errors)),
        "duration_seconds": float(estimate_times[-1] - estimate_times[0]),
        "ate_rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "ate_mean_m": float(errors.mean()),
        "ate_median_m": float(np.median(errors)),
        "ate_p95_m": percentile(errors, 95),
        "ate_max_m": float(errors.max()),
        "relative_translation_rmse_m_approx_1pct": float(
            np.sqrt(np.mean(relative_error * relative_error))
        ),
        "estimated_path_length_m": float(
            np.linalg.norm(np.diff(estimate_aligned, axis=0), axis=1).sum()
        ),
        "groundtruth_path_length_m": float(
            np.linalg.norm(np.diff(truth_matched, axis=0), axis=1).sum()
        ),
        "estimated_elevation_change_m": float(estimate_aligned[-1, 2] - estimate_aligned[0, 2]),
        "groundtruth_elevation_change_m": float(truth_matched[-1, 2] - truth_matched[0, 2]),
        "segments": segments,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
