#!/usr/bin/env python3
"""Compare feature-track lifetime AND quality between DOD and official OpenVINS.

Both pipelines dump the same thing, per frame::

    F <timestamp>
    <cam> <id> <u> <v>      # one line per surviving track, raw distorted pixels
    ...

Everything else is derived here, so the two sides are measured by identical
code with identical undistortion.

DOD side
--------
``core/tracker.cpp`` writes the dump when ``VIO_TRACK_DUMP`` is set.

Official side
-------------
``tools/ov_track_dump.cpp`` reads ``TrackBase::get_last_obs()`` and
``get_last_ids()``, both public -- official's source is not modified.

Metrics
-------
lifetime      how many frames a track survives (right-censored tracks excluded)
parallax      angle between a track's first and last cam0 bearing; this is what
              makes a feature triangulable, so it matters more than lifetime
epipolar      Sampson distance of each stereo pair to the epipolar geometry
              implied by the FIXED, KNOWN EuRoC extrinsics. A correct stereo
              correspondence sits on the epipolar line; a KLT match that has
              slid onto the wrong structure does not. No estimator needed --
              this is ground truth about correspondence quality.
stereo rate   fraction of cam0 tracks that also have a cam1 observation

    python3 scripts/track_lifetime.py dod_tracks.txt ov_tracks.txt
    python3 scripts/track_lifetime.py --selftest
"""

import sys
from collections import Counter, defaultdict

import numpy as np

# mav0/camN/sensor.yaml, same values the dump drivers use.
CAM = {
    0: dict(k=(458.654, 457.296, 367.215, 248.375),
            d=(-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05)),
    1: dict(k=(457.587, 456.134, 379.999, 255.238),
            d=(-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05)),
}
# T_cam_imu for each camera (rows of the 4x4), as in the EuRoC runner.
T_CAM_IMU = {
    0: np.array([[0.0148655429818, 0.999557249008, -0.0257744366974, 0.06522291331214665],
                 [-0.999880929698, 0.0149672133247, 0.00375618835797, -0.02070639072309887],
                 [0.00414029679422, 0.025715529948, 0.999660727178, -0.008054603453164811],
                 [0.0, 0.0, 0.0, 1.0]]),
    1: np.array([[0.0125552670891, 0.999598781151, -0.0253898008918, -0.04490198068735834],
                 [-0.999755099723, 0.0130119051815, 0.0179005838253, -0.02056977306809739],
                 [0.0182237714554, 0.0251588363115, 0.999517347078, -0.008638136949756423],
                 [0.0, 0.0, 0.0, 1.0]]),
}


def undistort(uv, cam):
    """Radtan undistortion by Newton iteration, matching OpenVINS's cam model."""
    fx, fy, cx, cy = CAM[cam]["k"]
    k1, k2, p1, p2 = CAM[cam]["d"]
    xy = np.empty_like(uv)
    xy[:, 0] = (uv[:, 0] - cx) / fx
    xy[:, 1] = (uv[:, 1] - cy) / fy
    out = xy.copy()
    for _ in range(20):
        x, y = out[:, 0], out[:, 1]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        out[:, 0] -= (x * radial + dx) - xy[:, 0]
        out[:, 1] -= (y * radial + dy) - xy[:, 1]
    return out


def essential():
    """E for x1^T E x0 = 0, from the fixed cam0->cam1 extrinsics."""
    T = T_CAM_IMU[1] @ np.linalg.inv(T_CAM_IMU[0])
    R, t = T[:3, :3], T[:3, 3]
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return skew @ R


def load(path):
    """-> (times, frames) where frames[i] is {cam: {id: (u, v)}}."""
    times, frames = [], []
    current = None
    with open(path) as handle:
        for line in handle:
            if line.startswith("F "):
                times.append(float(line[2:]))
                current = {0: {}, 1: {}}
                frames.append(current)
            elif current is not None:
                cam, fid, u, v = line.split()
                current[int(cam)][int(fid)] = (float(u), float(v))
    return times, frames


def lifetimes(frames):
    seen = Counter()
    for frame in frames:
        seen.update(frame[0].keys())
    alive = set(frames[-1][0]) if frames else set()
    return sorted(n for i, n in seen.items() if i not in alive), len(alive)


def parallax_deg(frames):
    """Angle between each completed cam0 track's first and last bearing."""
    first, last = {}, {}
    for frame in frames:
        for fid, uv in frame[0].items():
            first.setdefault(fid, uv)
            last[fid] = uv
    alive = set(frames[-1][0]) if frames else set()
    ids = [i for i in first if i not in alive]
    if not ids:
        return np.zeros(0)
    a = undistort(np.array([first[i] for i in ids]), 0)
    b = undistort(np.array([last[i] for i in ids]), 0)
    va = np.hstack([a, np.ones((len(a), 1))])
    vb = np.hstack([b, np.ones((len(b), 1))])
    va /= np.linalg.norm(va, axis=1, keepdims=True)
    vb /= np.linalg.norm(vb, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((va * vb).sum(1), -1.0, 1.0)))


def epipolar_px(frames):
    """Sampson distance (px) of every stereo pair against the known extrinsics."""
    E = essential()
    focal = 0.5 * (CAM[0]["k"][0] + CAM[1]["k"][0])
    left, right = [], []
    for frame in frames:
        shared = frame[0].keys() & frame[1].keys()
        for fid in shared:
            left.append(frame[0][fid])
            right.append(frame[1][fid])
    if not left:
        return np.zeros(0), 0.0
    x0 = undistort(np.array(left), 0)
    x1 = undistort(np.array(right), 1)
    v0 = np.hstack([x0, np.ones((len(x0), 1))])
    v1 = np.hstack([x1, np.ones((len(x1), 1))])
    Ev0 = v0 @ E.T
    Etv1 = v1 @ E
    num = (v1 * Ev0).sum(1) ** 2
    den = Ev0[:, 0] ** 2 + Ev0[:, 1] ** 2 + Etv1[:, 0] ** 2 + Etv1[:, 1] ** 2
    sampson = np.sqrt(num / np.maximum(den, 1e-12)) * focal
    pairs_per_frame = len(left) / max(1, len(frames))
    return sampson, pairs_per_frame


def quantiles(values, label, unit):
    if len(values) == 0:
        print(f"{label:<22} (none)")
        return
    q = np.percentile(values, [50, 75, 90, 95, 99])
    print(f"{label:<22} med {q[0]:7.3f}  p75 {q[1]:7.3f}  p90 {q[2]:7.3f}  "
          f"p95 {q[3]:7.3f}  p99 {q[4]:7.3f}  {unit}")


def report(name, path):
    times, frames = load(path)
    lives, censored = lifetimes(frames)
    par = parallax_deg(frames)
    epi, pairs = epipolar_px(frames)
    alive = [len(f[0]) for f in frames]
    stereo_rate = np.mean([len(f[0].keys() & f[1].keys()) / max(1, len(f[0])) for f in frames])

    print(f"\n=== {name}   ({len(frames)} frames)")
    print(f"completed tracks       {len(lives)}  (+{censored} censored)")
    print(f"cam0 alive / frame     {np.mean(alive):.1f}  (min {min(alive)}, max {max(alive)})")
    print(f"stereo pairs / frame   {pairs:.1f}   ({100 * stereo_rate:.1f}% of cam0 tracks)")
    if lives:
        print(f"lifetime               med {np.median(lives):.0f}  mean {np.mean(lives):.1f}  "
              f"p95 {np.percentile(lives, 95):.0f} frames   died<=2: "
              f"{100 * sum(1 for x in lives if x <= 2) / len(lives):.1f}%")
    quantiles(par, "parallax", "deg")
    if len(par):
        for thresh in (0.5, 1.0, 2.0):
            print(f"  parallax < {thresh:>3} deg      {100 * np.mean(par < thresh):.1f}% of tracks")
    quantiles(epi, "epipolar (Sampson)", "px")
    if len(epi):
        for thresh in (1.0, 2.0, 5.0):
            print(f"  epipolar > {thresh:>3} px       {100 * np.mean(epi > thresh):.2f}% of stereo pairs")


def distort(xy, cam):
    """Forward radtan projection, the inverse of undistort(); test-side only."""
    fx, fy, cx, cy = CAM[cam]["k"]
    k1, k2, p1, p2 = CAM[cam]["d"]
    x, y = xy[0], xy[1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return (fx * xd + cx, fy * yd + cy)


def project(point_c0, cam):
    """3D point in the cam0 frame -> distorted pixels in the requested camera."""
    if cam == 1:
        T = T_CAM_IMU[1] @ np.linalg.inv(T_CAM_IMU[0])
        point_c0 = T[:3, :3] @ point_c0 + T[:3, 3]
    return distort(point_c0[:2] / point_c0[2], cam)


def selftest():
    import tempfile, os

    # A genuine correspondence: one 3D point projected into both cameras. This
    # exercises distort/undistort as inverses and the extrinsics at once.
    near = np.array([0.4, -0.2, 3.0])
    far = np.array([0.6, -0.2, 3.4])
    l0, r0 = project(near, 0), project(near, 1)
    l1 = project(far, 0)
    text = (f"F 0.0\n0 1 {l0[0]:.4f} {l0[1]:.4f}\n1 1 {r0[0]:.4f} {r0[1]:.4f}\n0 2 300.0 100.0\n"
            f"F 1.0\n0 1 {l1[0]:.4f} {l1[1]:.4f}\n"
            f"F 2.0\n0 1 {l1[0]:.4f} {l1[1]:.4f}\n0 3 400.0 300.0\n")
    path = os.path.join(tempfile.mkdtemp(), "t.txt")
    with open(path, "w") as handle:
        handle.write(text)

    times, frames = load(path)
    assert len(frames) == 3 and times[2] == 2.0
    assert set(frames[0][0]) == {1, 2} and set(frames[0][1]) == {1}
    lives, censored = lifetimes(frames)
    assert lives == [1], lives          # id 2 lived one frame; 1 and 3 censored
    assert censored == 2, censored

    epi, pairs = epipolar_px(frames)
    assert len(epi) == 1 and epi[0] < 0.01, epi  # true correspondence -> ~0 px
    # The same left point matched 40 px off in the right image must not pass.
    bad = [{0: {1: l0}, 1: {1: (r0[0], r0[1] + 40.0)}}]
    assert epipolar_px(bad)[0][0] > 20.0, epipolar_px(bad)

    par = parallax_deg(frames)   # the one completed track is id 2, a single obs
    assert len(par) == 1 and par[0] == 0.0, par
    # A track that moved from `near`'s bearing to `far`'s has real parallax.
    moved = [{0: {9: l0}, 1: {}}, {0: {9: l1}, 1: {}}, {0: {}, 1: {}}]
    assert 2.0 < parallax_deg(moved)[0] < 3.0, parallax_deg(moved)
    print("selftest ok")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return 0
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    report("DOD", sys.argv[1])
    report("OFFICIAL", sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
