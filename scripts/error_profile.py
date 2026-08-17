#!/usr/bin/env python3
"""Where does the ATE gap against official come from?

For each sequence: align both trajectories to ground truth (Umeyama, the same
alignment evaluate_trajectory.py uses), then report

  * the error time series split into thirds, so an early transient is
    distinguishable from a steady drift,
  * the largest single error excursion and when it happens,
  * relative pose error over a 1 s window, which is alignment-free and
    measures local drift rate rather than accumulated global error.

An ATE gap that lives in the first third is an initialisation problem. One that
tracks the RPE gap is a drift-rate problem. One that is neither is a discrete
excursion, and the timestamp says where to look.
"""
import csv
import sys
from pathlib import Path

import numpy as np

SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
        "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
        "KAIST_circle", "KAIST_infinity"]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["timestamp"]) for r in rows])
    p = np.array([[float(r["px"]), float(r["py"]), float(r["pz"])] for r in rows])
    return t, p


def associate(te, tt, tol=0.02):
    """Nearest-neighbour match of estimate times into truth times."""
    idx = np.searchsorted(tt, te)
    idx = np.clip(idx, 1, len(tt) - 1)
    left, right = tt[idx - 1], tt[idx]
    pick = np.where(np.abs(te - left) < np.abs(te - right), idx - 1, idx)
    good = np.abs(tt[pick] - te) < tol
    return np.nonzero(good)[0], pick[good]


def umeyama(src, dst):
    """Rigid (rotation+translation) alignment, no scale."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def rpe(t, p, tt, tp, delta=1.0):
    """Translation error over a fixed time window, alignment-free."""
    ei, ti = associate(t, tt)
    if len(ei) < 10:
        return float("nan")
    t_a, p_a, g_a = t[ei], p[ei], tp[ti]
    j = np.searchsorted(t_a, t_a + delta)
    ok = j < len(t_a)
    if ok.sum() < 10:
        return float("nan")
    i = np.nonzero(ok)[0]
    j = j[ok]
    d_est = np.linalg.norm(p_a[j] - p_a[i], axis=1)
    d_gt = np.linalg.norm(g_a[j] - g_a[i], axis=1)
    return float(np.sqrt(np.mean((d_est - d_gt) ** 2)))


def profile(est_path, gt_path):
    t, p = load(est_path)
    tt, tp = load(gt_path)
    ei, ti = associate(t, tt)
    if len(ei) < 10:
        return None
    te, pe, pg = t[ei], p[ei], tp[ti]
    R, s = umeyama(pe, pg)
    err = np.linalg.norm((pe @ R.T + s) - pg, axis=1)
    n = len(err)
    thirds = [float(np.sqrt(np.mean(err[a:b] ** 2)))
              for a, b in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n))]
    k = int(np.argmax(err))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "thirds": thirds,
        "max": float(err[k]),
        "max_at": float(te[k] - te[0]),
        "rpe1": rpe(t, p, tt, tp, 1.0),
    }


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else "docs/results/trajectories/reeval_20260817/runs")
    print("%-16s %-4s %7s %7s %7s %7s %7s %7s %7s" % (
        "sequence", "who", "rmse", "1st/3", "2nd/3", "3rd/3", "max", "at[s]", "rpe1s"))
    for seq in SEQS:
        gt = root / f"{seq}_groundtruth.csv"
        for who, est in (("DOD", root / f"{seq}_estimate.csv"),
                         ("OV", root / f"{seq}_ov_estimate.csv")):
            if not est.exists() or not gt.exists():
                continue
            r = profile(est, gt)
            if r is None:
                continue
            print("%-16s %-4s %7.4f %7.4f %7.4f %7.4f %7.3f %7.1f %7.4f" % (
                seq, who, r["rmse"], *r["thirds"], r["max"], r["max_at"], r["rpe1"]))
        print()


if __name__ == "__main__":
    main()
