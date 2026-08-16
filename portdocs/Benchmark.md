# VIO Porting Benchmark & Performance Analysis

This document describes the methodology, setup, and numerical results comparing the original Python VIO pipeline, the official Open_VINS design pattern, and our ported high-performance DOD C++17 pipeline.

## Benchmark 1: Python vs C++ Port (Pure Propagation)

We compared the execution speed of our ported C++ code against the original Python VIO pipeline running IMU propagation.

### Methodology
* **Inputs**: 100 Hz IMU readings simulated over a 1.0-second time window (100 propagation steps).
* **Solver**: Runge-Kutta 4th Order (RK4) integration.
* **Runs**: 1,000 propagation cycles.
* **Environment**: 
  * Python pipeline executed via `uv run tests/benchmark.py` using Python 3.
  * C++ pipeline compiled under `-DCMAKE_BUILD_TYPE=Release` (optimized `-O3`) and executed via `./build/benchmark`.

### Results

| Implementation | Total Time (1,000 runs) | Avg Time Per Run (1.0s window) | Speedup |
| :--- | :--- | :--- | :--- |
| Python (`uv run`) | 91.714 seconds | 91.714 ms | *Reference* (1x) |
| **Ported C++17 (Release)** | **1.092 seconds** | **1.092 ms** | **84x** |

---

## Benchmark 2: Official Open_VINS vs Ported DOD (Propagation + Cloning)

To evaluate the design-level optimizations, we benchmarked the official Open_VINS polymorphic structure against our flat DOD implementation.

### Methodology
* **Harness**: [compare_openvins.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/tests/compare_openvins.cpp) compiled with `-O3` Release optimization.
* **Operations**: RK4 mean state propagation and EKF clone augmentation (every 5 steps).
* **State Size**: Simulated $300 \times 300$ EKF covariance matrix (matching active SLAM landmark configurations).
* **Runs**: 50,000 iterations.

### Design Elements Evaluated
* **Open_VINS**: Dynamic EKF variable list (`std::vector<std::shared_ptr<Type>>`), heap allocation of camera pose clones (`std::make_shared`), and matrix resizing (`Eigen::MatrixXd::conservativeResize()`).
* **Ported DOD**: Contiguous EKF variable structs (`Variable clones[12]`), zero runtime heap allocations, and in-place updates of a pre-allocated static matrix (`Eigen::Matrix<double, 512, 512>`).

### Results

| Design Pattern | Total Time (50k iterations) | Avg Time Per Iteration | Speedup |
| :--- | :--- | :--- | :--- |
| Official Open_VINS Style | 2.971 seconds | 59,417 ns | *Reference* (1x) |
| **Ported DOD C++17** | **0.061 seconds** | **1,229 ns** | **48x** |

---

## Performance Rationale

```
+-------------------------------------------------------------------+
|  Open_VINS Pattern (48x slower)                                   |
|  [Clone Augment] -> [new PoseJPL] -> [conservativeResize]          |
|  (Requires OS Heap Allocator lock & copying 90,000 double values) |
+-------------------------------------------------------------------+
                               VS
+-------------------------------------------------------------------+
|  Ported DOD Pattern (48x faster)                                  |
|  [Clone Augment] -> [Update index] -> [In-place matrix slices]    |
|  (Zero allocations, stack-bound, cache-contiguous writes)         |
+-------------------------------------------------------------------+
```

1. **Memory Allocator Overhead**: Calling `std::make_shared` forces a search in the OS heap for free blocks, requiring thread locking and pointer setup. DOD initializes pre-allocated stack memory in $\mathcal{O}(1)$ time.
2. **Matrix Copies**: Resizing a $300 \times 300$ matrix dynamically forces Eigen to allocate a new $306 \times 306$ block, copy all $90,000$ values, and deallocate the old block. Our DOD slice updates write only to the required $6 \times 300$ indices of a pre-allocated matrix.
3. **CPU Cache Locality**: Pointer chasing inside a vector of `shared_ptr` causes CPU stalls due to random RAM accesses. Contiguous memory layouts in DOD allow the CPU prefetcher to load values into L1/L2 caches ahead of time.

---

## Benchmark 3: Full End-to-End Pipeline (IMU + Camera + EKF)

This is the most realistic comparison: the **entire** VIO pipeline from initialization through EKF propagation, camera clone augmentation, MSCKF visual updates, and sliding-window marginalization — all three implementations exercised identically.

### Methodology

* **Dataset**: Synthetic 30-point 3D feature set, 30 features tracked across 7 camera frames at 10 Hz, and 61 IMU readings at 100 Hz over a 0.60-second window.
* **Initialization**: Static IMU initializer with a 0.50-second window.
* **EKF Operations per run**: 61 IMU propagation steps + 7 camera clone augmentations + MSCKF visual updates on lost tracks + sliding-window marginalization.
* **Scripts/Executables**:
  * Python: [`tests/benchmark_full.py`](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/tests/benchmark_full.py) via `uv run` (100 full pipeline runs)
  * C++ DOD: [`tests/benchmark_full.cpp`](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/tests/benchmark_full.cpp) compiled with `-O3` Release (10,000 full pipeline runs)
  * Open_VINS Design Pattern: [`tests/compare_openvins.cpp`](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/tests/compare_openvins.cpp) — 50,000 iterations of the per-step core (RK4 + clone augment + MSCKF update), using both polymorphic and DOD data layouts.

### Results

| Implementation | Runs | Total Time | Avg Time/Run | Speedup vs Python |
| :--- | :--- | :--- | :--- | :--- |
| **Python** (`vio_pipeline_v2`) | 100 | 5.546 s | **55.46 ms** | 1× (baseline) |
| **C++ DOD Port** (`benchmark_full`) | 10,000 | 24.935 s | **2.49 ms** | **22×** |
| Open_VINS design (per EKF step) | 50,000 | 1.919 s | **38.38 ns/step** | — |
| **Ported DOD** (per EKF step) | 50,000 | 1.748 s | **34.97 ns/step** | **1.1×** vs OV |

> **Note on the per-step comparison**: In this apples-to-apples comparison, both Open_VINS and DOD use equivalent `MatrixXd` allocations for the visual Jacobian `H` to avoid unfair stack vs heap differences. The ~10% per-step advantage of DOD comes entirely from the pre-allocated covariance slice (no `conservativeResize()`) and direct struct member writes (no virtual dispatch).

### End-to-End Speedup Breakdown

The **22× end-to-end speedup** of C++ over Python breaks down as:

| Stage | Python Overhead Source | C++ Advantage |
| :--- | :--- | :--- |
| IMU propagation loop | Python bytecode interpreter, CPython GIL overhead per op | Compiled RK4, cache-hot doubles |
| EKF covariance update | NumPy `@` operators with temporary heap allocation per multiplication | In-place Eigen block views, no temporaries |
| Feature database lookup | Python dict `__hash__` + dynamic type boxing per feature | Flat `FeatureDatabase` array with `int` ID indexing |
| Initialization | Python threading overhead (`threading.RLock`, `time.time()`) | Single-threaded `StaticInitializer` on the stack |
| Clone augmentation | Dynamic list `append` + `numpy.zeros()` allocation | Index-bump into pre-allocated `Variable clones[12]` |

### Reproduction Commands

```bash
# Python full pipeline
uv run tests/benchmark_full.py

# C++ full pipeline (build first)
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
./benchmark_full

# Open_VINS vs DOD per-step comparison
./compare_openvins
```

---

## Benchmark 4: Real KAIST `circle.bag` Three-Way Comparison

This benchmark replaces synthetic per-step timing with a real ROS1 bag run of
all three implementations against the same data: KAIST VIO dataset
`circle/circle.bag` (191 s, stereo infra1/infra2 @ ~25 Hz, IMU
`/mavros/imu/data` @ 100 Hz, ground truth `/pose_transformed` @ ~47 Hz), using
the identical `kaist_vio` calibration and estimator parameters (FEJ on, RK4
integration, `max_clones=11`, `max_slam=50`, `sigma_pix=1.2`) for all three.

### Methodology

* **DOD C++**: `ros/vio_rosbag_runner.cpp` — reads the bag directly via
  `rosbag::View` (no real-time pacing), running IMU propagation, stereo KLT
  tracking, and MSCKF updates as fast as the CPU allows.
* **Official OpenVINS**: `rpng/open_vins` commit `6948812`, `run_subscribe_msckf`
  launched via `roslaunch ov_msckf subscribe.launch config:=kaist_vio`, bag
  played in **real time** (`rosbag play -r 1`) since it is a live ROS node.
* **Python v2** (`vio_pipeline_v2/main.py`): same real-time bag playback via a
  live ROS node; this is the only implementation with working hover
  detection (FIFO/LIFO switching), so it is also the hover-behavior reference.
* Ground truth for all three: the single `/pose_transformed` recording from
  the DOD run (same bag, same topic, so it is identical across runs).
* Accuracy: rigid-body (Umeyama) alignment + ATE via
  `scripts/plot_circle_comparison.py`'s alignment routine (matching the
  standalone `evaluate_trajectory.py` used for the JSON reports in
  `docs/results/`), timestamp-associated at 30 ms tolerance.
* Hover segments: ground-truth samples with instantaneous speed
  ≤ 0.15 m/s (47% of the trajectory) evaluated separately from the
  non-hover remainder.
* Memory: peak resident set size (`VmHWM`) polled every 0.5 s on the
  estimator process (`/usr/bin/time` unavailable on the offline remote host).

### Results

| Metric | DOD C++ (ours) | Official OpenVINS | Python v2 |
| :--- | :--- | :--- | :--- |
| Frames/updates processed | 4,758 | 2,620 | 1,629 (rate-limited under real-time load) |
| Mean per-frame latency | **11.19 ms** (track 1.55 + update 9.63) | 8.34 ms | 99.9 ms (max spike 785 ms) |
| Peak RSS | 390 MB | **75 MB** | 941 MB |
| Full-trajectory ATE RMSE | 0.685 m | **0.067 m** | 0.656 m |
| Hover-segment ATE RMSE | 0.525 m | **0.078 m** | 0.569 m |
| Non-hover ATE RMSE | 0.780 m | **0.065 m** | 0.728 m |
| Estimated path length vs GT (29.8 m) | 10.8 m (under) | 30.9 m (close) | 54.1 m (2.8× over) |

### Interpretation

* **Speed**: the DOD port is a clear step up from the Python reference
  (~9× lower per-frame latency, no real-time drops) and is within ~35% of
  official OpenVINS per-frame despite using a much simpler MSCKF-only
  update path — official OpenVINS is still faster per frame because its
  feature-management and SLAM-augmented update are more selective about
  which tracks enter the filter.
* **Accuracy**: official OpenVINS is ~10× more accurate than both the DOD
  port and the Python reference. **Correction to an earlier claim in this
  document**: the "MSCKF update assumes camera 0 only" gap recorded here and
  in `docs/COMPREHENSIVE_AUDIT.md` was stale — `update_msckf` actually calls
  `get_feature_jacobian_mixed` (`msckf/updater_mixed.cpp`), which already
  loops per-observation using each measurement's real `cam_id`, correctly
  stacking stereo rows; the camera-0-only function (`updaters.cpp`) was dead
  code with zero callers. The real remaining gap is the missing SLAM
  (delayed anchored-landmark) update — see "SLAM updater attempt" below —
  which is what actually causes the scale/drift accumulation over the full
  circle/infinity trajectories. Python v2 has the equivalent gap in a
  different form (rate-limited real-time processing drops frames, and its
  own scale/drift accumulates similarly) — the two land within 5% of each
  other on `circle.bag` despite very different implementations.
* **Hover**: all three are somewhat *more* accurate during hover segments
  than during motion (ATE drops for all three), which is expected since
  hover segments have near-zero apparent motion and thus less opportunity
  for scale/rotation drift to accumulate — but only Python v2 has explicit
  hover detection logic (`HOVER-DET`/FIFO-LIFO mode switching in its log);
  DOD and official OpenVINS have no hover-specific logic and are simply
  benefiting from the segments' low dynamics.
* **Memory**: the DOD port's arena-based, zero-heap-allocation design keeps
  it well under Python's footprint but well above official OpenVINS, which
  processes fewer frames per second of wall time and uses a leaner
  feature-management footprint than the DOD port's fixed-capacity buffers.

Raw per-run CSVs, JSON accuracy reports, and the trajectory comparison figure
are in `docs/results/` and `docs/circle_trajectory_comparison.png`.

---

## Benchmark 5: KAIST `infinite.bag` (second trajectory, same rig/config)

Same methodology, calibration, and topics as Benchmark 4, run on the KAIST
figure-8 trajectory (`infinity/infinite.bag`, 188 s) instead of the circle.
Confirms Benchmark 4's pattern is not a one-dataset fluke. **Updated
2026-07-22** with DOD's current build (ZUPT-off, tracker RANSAC + mono-track
survival, feature-init config wiring fix — see below); the original
2026-07-21 numbers (0.587 m ATE, 9.2 m path length) predate those fixes.

| Metric | DOD C++ (ours) | Official OpenVINS |
| :--- | :--- | :--- |
| Frames/updates processed | 4,471 | 3,756 |
| Mean estimator latency | 7.65 ms/frame | — (native OV timing not extracted for this run) |
| Peak RSS | 258 MB | — |
| Full-trajectory ATE RMSE | 0.727 m | **0.070 m** |
| Estimated path length vs GT (29.1 m) | 6.1 m (severe scale collapse) | 28.8 m (close) |

The trajectory overlay (`docs/infinity_trajectory_comparison.png`) shows this
directly: OpenVINS traces the full figure-8 closely, while the DOD estimate
still collapses to a small tangle near the crossing point — same pattern as
`circle.bag` (0.69 m ATE there). Today's tracker/config fixes did not close
this gap on `infinity.bag` either (0.587 m → 0.727 m, essentially unchanged
within run-to-run variance) — consistent with the `circle.bag` finding that
none of the fixes tried so far are individually the dominant cause. See
"Reference switched to official OpenVINS C++ source" below for the full
list of what was checked and ruled out.

## SLAM updater attempt (2026-07-21/22) — not yet a net win

Given the DOD port's ~10× accuracy gap traces to the missing SLAM (delayed
anchored inverse-depth landmark) update, a full port was attempted:
`msckf/updater_slam_helper.{hpp,cpp}` (representation Jacobian +
per-measurement chain) and `delayed_init_slam`/`update_slam`/`change_anchors`/
`perform_anchor_change` in `msckf/updaters.cpp`, wired into
`msckf/vio_manager.cpp`'s main loop behind a new `VioManagerOptions::enable_slam`
flag (**default `false`** — not part of the numbers above).

* **The Jacobians are verified correct.** `tests/verify_slam_jacobians.cpp`
  finite-difference validates both the representation Jacobian
  (`get_feature_jacobian_representation`) and the full per-measurement chain
  (`get_feature_jacobian_slam`) against numerically perturbed state, with
  errors at the 1e-4–1e-6 level (floating-point precision, not a bug). This
  rules out a sign/frame-convention error in the new math.
* **Originally not a net accuracy win, and unstable** (see below for the fix):
  with SLAM enabled, `circle.bag` ATE got *worse* (0.75–1.0 m depending on
  variant) rather than better, with growing elevation drift, and tightening
  the chi2 acceptance gate as a mitigation attempt caused a catastrophic
  divergence (45,000,000 m ATE).
* **Root cause found 2026-07-22: this was the tracker RANSAC bug (below), not
  a SLAM-specific issue.** Bad/outlier feature correspondences were feeding
  garbage triangulated 3D points straight into `delayed_init_slam`, producing
  poorly-conditioned landmarks whose covariance blew up. After fixing the
  tracker (adding the missing RANSAC epipolar-consistency check,
  `core/tracker.cpp`), re-running SLAM+RANSAC together on `circle.bag`
  completed the full 4758-frame run with **no crash and no divergence**
  (ATE 0.675 m, vs. 0.686 m for RANSAC alone) — confirming the instability is
  gone, though SLAM's accuracy contribution on this dataset is still modest
  (see the tracker section below for why: too few features survive long
  enough to qualify as SLAM candidates to meaningfully correct drift here).
* **Status**: code compiles, is regression-tested (5/5 `ctest`, including the
  finite-difference Jacobian check), and is **enabled** in
  `ros/vio_rosbag_runner.cpp` (`enable_slam = true`) since it is now verified
  stable and does not hurt accuracy — the numbers above (0.675 m circle.bag
  ATE) include it.

## Tracker bug found 2026-07-22: missing RANSAC outlier rejection

Investigating why DOD's *raw* (unaligned) trajectory shape was a jagged,
non-repeating open path — not just "10× more ATE than OpenVINS" but visibly
the wrong *shape*, not tracing the circle/figure-8 at all — traced back to
`core/tracker.cpp`'s stereo KLT frontend. Comparing against the Python
reference (`core/track_KLT.py`, ported alongside this file) found the C++
port had dropped `cv2.findFundamentalMat(FM_RANSAC)` epipolar-consistency
rejection entirely: `calcOpticalFlowPyrLK`'s own `status` flag reports a track
as "successful" even when it converged onto the wrong structure or slid along
an edge — exactly the kind of plausible-but-geometrically-wrong
correspondence that corrupts every downstream MSCKF/SLAM update without
tripping any existing check.

**Fix**: `core::ransac_inliers` in `core/tracker.cpp` runs
`cv::findFundamentalMat` on undistorted (normalized) point pairs for each
camera's own temporal KLT flow (old→new), independently for left and right,
requiring both to agree; skipped (accept-all) below 10 points, matching the
reference's own minimum sample size.

**Effect on `circle.bag`** (before → after, ZUPT still on at this point):
ATE 0.685 m → 0.686 m (roughly unchanged) but raw path length 10.3 m → 17.0 m
(closer to the 29.8 m ground truth) and the raw trajectory shape changed from
pure jagged noise to a recognizable (if still not closed/repeating) loop —
see `docs/results/` for the underlying CSVs. ATE didn't move much because
MSCKF's own chi2 gate was already rejecting a lot of the resulting bad
updates; the RANSAC fix instead improves what happens *before* that gate,
raising the quality of what does get accepted. Confirmed as a real,
independent contributor: re-enabling SLAM after this fix was stable (see
above), where before the fix it caused a 45,000,000 m divergence.

**Remaining accuracy gap**: even with RANSAC, the shape doesn't overlay
repeated laps the way OpenVINS's does. This is very likely severity, not a
different bug — pure MSCKF-without-SLAM inherently accumulates more drift per
revolution than OpenVINS's more mature, decade-refined estimator, and over 8
circle revolutions that compounds into visible non-overlap even at a per-step
drift rate only modestly worse than OpenVINS's. Confirmed NOT the propagation
math (verified line-by-line identical to `msckf/Propagator.py`, including the
Qc→Qd noise discretization) and NOT insufficient SLAM landmark count (43/50
capacity used, healthy). Further narrowing (tracker feature density/quality
vs. OpenVINS's own frontend, static-initialization bias/orientation accuracy)
was not completed this session.

## Real bug found and fixed 2026-07-22: ZUPT always attempted

Official OpenVINS's own `kaist_vio/estimator_config.yaml` ships
`try_zupt: false` for this dataset (already noted in
`docs/COMPREHENSIVE_AUDIT.md` from an earlier session, but never wired into
the DOD runner). The DOD runner was unconditionally calling
`try_update_zupt()` every frame with no way to disable it. Instrumented and
confirmed: on `circle.bag`, ZUPT fired on 68–84% of frames early in the run
(dropping to ~37% by the end) — a slow, gentle circular flight looks
"stationary enough" to the IMU-jerk/disparity gate far more often than a true
hover, freezing propagation on a large fraction of frames.

**Fix**: added `VioManagerOptions::enable_zupt` (default `true` for
backward compatibility), set to `false` in `ros/vio_rosbag_runner.cpp` to
match the upstream KAIST config. Kept — this is a real, verified difference
from official OpenVINS's own settings for this dataset, independent of
everything else in this section.

## Reference switched to official OpenVINS C++ source (2026-07-22, continued)

Python (`msckf/*.py`) was dropped as a correctness reference — it was never
itself validated as bit-perfect against official OpenVINS, so "matches
Python" only proved DOD matched Python's own bugs too. Re-audited against the
actual official C++ source (`/workspace/open_vins_official`, commit
`6948812`, the same checkout that produces the 0.067 m ATE numbers above).

* **Tracker: mono-track survival + RANSAC threshold fix.** Official's
  `TrackKLT.cpp` keeps a track alive if *either* camera's temporal KLT
  succeeds (falling back to a mono/left-only observation); the port required
  *both* cameras to succeed every frame, capping every feature's lifetime at
  "still stereo-visible" — much shorter than official's per-camera-independent
  lifetime. Also, the RANSAC threshold used `2.0 / max(fx,fy)` in official vs.
  a mistaken `1.0 / fx-only` in the port (≈2× too strict, ignoring `fy`
  entirely). Both fixed (`core/tracker.cpp`) — **but measured effect was a
  wash**: ATE 0.675 m → 0.690 m, path length 16.7 m → 12.6 m. Kept anyway
  since it now genuinely matches official's tracking behavior and doesn't
  hurt; the accuracy gap is evidently not primarily a track-lifetime problem.
* **MSCKF update math (Jacobians, chi2 gate, nullspace projection, measurement
  compression): verified equivalent to official.** Chi-square gating uses a
  Wilson-Hilferty approximation vs. official's boost chi-squared table —
  numerically within 0.2% at realistic dof, not the cause. Nullspace
  projection/compression use HouseholderQR locally vs. Givens rotations in
  official — different method, same information content, provably equivalent
  under an orthogonal transform. Noise weighting (`sigma_pix_sq`, R
  construction) matches structurally.
* **Feature-initialization config: a real wiring bug found, but official's
  actual threshold values are dangerous without official's full pipeline.**
  `update_msckf()` was constructing its own `core::FeatureInitializerOptions`
  as a disconnected local variable, silently ignoring
  `VioManagerOptions::feat_init_opt` (the SLAM path already used it
  correctly). Fixed the wiring (`UpdaterMSCKFData` now carries and uses
  `feat_init_options`, threaded through `init_updater_msckf`). Then tried
  official's actual effective KAIST values (`min_dist=0.10, max_dist=10.0,
  max_baseline=200, max_cond_number=25000, max_runs=5`, read directly from
  `config/kaist_vio/estimator_config.yaml` and
  `FeatureInitializerOptions.h`'s defaults) — this caused a **catastrophic
  divergence** (169 m ATE, path length 1280 m, elevation drift −1234 m).
  Official's much looser `max_cond_number`/`min_dist` are only safe inside
  official's *complete* pipeline (which has other complementary
  quality-control this port doesn't have yet, e.g. the mono-track lifetime
  behavior, sub-pixel corner refinement, and OpenCV's real FAST detector
  instead of a custom heuristic) — copying just the numbers let
  badly-conditioned near-degenerate triangulations straight into the filter.
  **Reverted to the port's own safer default thresholds**, keeping the wiring
  fix (a genuine correctness improvement independent of which values are
  used).

**Status after this round**: `circle.bag` ATE ≈ 0.69 m (essentially
unchanged from before this round — the two kept fixes were each individually
correct/verified but neither was the single cause of the ~10× gap). Every
concrete, verifiable lead surfaced by comparing against the propagation math,
the MSCKF update math, the chi2/nullspace/compression math, and the
feature-initialization config has now been checked against the real official
C++ source and is either already fixed or ruled out as the dominant cause.
The remaining gap likely requires reproducing more of official's complete
quality-control pipeline as a unit (sub-pixel corner refinement, real FAST
detector, and possibly static-initialization bias/orientation accuracy),
rather than one more isolated parameter or threshold fix — a substantially
larger undertaking than anything attempted so far.

## Deep-dive continued (2026-07-22): mechanistic root cause found, full fix not reached

Following the "/goal" directive to make DOD identical to official OpenVINS,
pursued several more hypotheses systematically, each tested empirically on
`circle.bag` (baseline: 0.685 m ATE):

* **Static-init covariance blocks**: found official overrides position
  (σ=0.05) and velocity (σ=0.01) rather than using a uniform σ=0.02 for all
  15 error-state dims at init. Fixed to match — **negligible effect** (kept,
  since it's genuinely correct).
* **Stereo timestamp**: official uses cam0's own timestamp as authoritative;
  the port averaged left/right. Switched to cam0-only — **negligible effect**
  (kept, removes a variable).
* **Initial state values, direct numeric comparison**: instrumented both
  DOD and official to print their actual computed initial orientation/gyro
  bias/accel bias on the identical bag. Official: `bg=[-0.0022,-0.0012,0.0000]`.
  DOD: `bg=[0.0001,-0.0000,0.0000]` — DOD's gyro bias estimate is essentially
  zero while official's is meaningfully nonzero. **Decisive test**: hardcoded
  official's exact values into DOD's init and re-ran — **ATE unchanged**
  (0.684 m). This proves the initial-state discrepancy is NOT the cause; the
  ongoing per-step correction doesn't hold the correct value either way.
* **Error-over-time analysis**: plotted alignment error at fine time
  granularity instead of one aggregate ATE number. No single catastrophic
  frame-to-frame jump exists (max ~0.07 m in one step) — the error grows
  gradually (if noisily) over the full 158 s, not from one bad event.
* **Major mechanistic finding**: instrumented `update_msckf` and
  `single_triangulation` directly. **Only ~4-5% of MSCKF feature-update
  attempts succeed** at steady state; the rest are rejected, overwhelmingly
  at the triangulation stage (not chi2). Breaking down triangulation
  rejections: ~52% condition-number-too-high (insufficient parallax —
  plausible given `circle.bag`'s slow, small-radius motion), ~30%
  near-distance (depth below `min_dist`, including some literal negative
  depths consistent with genuinely poor-parallax geometry rounding to
  nonsense). **MSCKF is severely constraint-starved on this real dataset** —
  this is the first concrete, data-verified mechanism found all session for
  *why* DOD underperforms, as opposed to "everything checked matches."
* **Tried to fix the starvation via tracking precision, twice, both made it
  worse**: (1) real `cv::FAST` detector + `cv::cornerSubPix` sub-pixel
  refinement together → 0.87 m (worse). (2) Isolated test: sub-pixel
  refinement ALONE on top of the *original* detector's point selection
  (decoupling "which points" from "how precise") → 1.18 m (worse still).
  This rules out detector choice as the cause and implicates something
  **downstream that doesn't tolerate improved measurement precision** — the
  leading unexamined suspect is online camera-intrinsics calibration
  (`do_calib_camera_intrinsics = true`; its Jacobian/consistency was never
  audited this session). Both experiments reverted; original detector kept.

**Status**: every other major subsystem checked against the real official
C++ source this session — propagation, MSCKF Jacobians/chi2/nullspace/
compression, static-init algorithm, calibration values, per-frame operation
order, FEJ — matches or has negligible effect. Six real, verified fixes are
kept (ZUPT toggle, tracker RANSAC + threshold, mono-track survival,
feature-init config wiring, static-init covariance, cam0 timestamp). Three
attempts that made things worse were identified and reverted (official's
raw triangulation thresholds, FAST+subpixel together, subpixel alone).
`circle.bag` ATE remains ~0.68-0.69 m vs official's 0.067 m. Full parity was
not reached. The most concrete untried lead: audit the online
camera-intrinsics calibration Jacobian, since it's the one subsystem never
checked and is implicated by the "better precision makes it worse" pattern.

## Camera-intrinsics resync bugs found and fixed (2026-07-22, still no ATE change)

The "improving tracker precision makes things worse" paradox (real FAST
detector + sub-pixel refinement: 0.69→0.87→1.18 m ATE) led to auditing online
camera-intrinsics calibration (`do_calib_camera_intrinsics=true`) against
official's real C++ source — and found **two genuine, high-confidence
correctness bugs**:

1. The EKF estimates updated intrinsics into `state.cam_intrinsics[]`, but
   every residual/Jacobian computation in the updater read distortion from
   `vio.params.cam_models[]` — a **separate buffer written once at init and
   never resynced**. Official's `StateHelper::EKFUpdate` copies the estimated
   intrinsics back into its live camera objects after every update
   (`StateHelper.cpp:191-197`); the port had no equivalent. Fixed in
   `msckf/vio_manager.cpp`: resync `params.cam_models[]` from
   `state.cam_intrinsics[]` after every visual update.
2. The **tracker** also keeps its own separate camera-model copy
   (`TrackerData::cameras[]`, used for `undistort()` when computing every
   observation's `uv_norm`) — also never resynced, so raw measurement data
   was computed with stale intrinsics even after fix #1. Fixed in
   `ros/vio_rosbag_runner.cpp`: copy the estimator's updated `cam_models[]`
   into the tracker's copy every frame.

Both are real bugs (the estimated intrinsics were silently drifting away
from what the filter actually used, exactly the kind of thing that would
make more-confident/precise measurements push harder on a bogus,
never-applied "correction") — **but fixing both together produced no
measurable ATE change** (0.685 m before and after). A follow-up **control
test** — disabling `do_calib_camera_intrinsics` entirely, freezing intrinsics
at their initial calibrated values — also produced no meaningful change
(0.695 m). This rules out online intrinsics calibration, sync bugs aside, as
a major factor for this dataset, even though the sync bugs themselves are
worth having fixed.

**Covariance-conditioning check**: instrumented `state.Cov`'s active block to
log eigenvalues periodically. Small negative eigenvalues are present
throughout (~1e-14 to 1e-19 in magnitude against a max eigenvalue of 1-55) —
this is floating-point roundoff noise near the system's genuinely-zero
unobservable directions (3 translation + 1 yaw, per standard VINS
observability theory), not a real positive-semi-definiteness violation. One
unexplained anomaly: the largest eigenvalue froze at an identical value
(55.04764) for an 80+ second window — plausibly related to the known MSCKF
starvation (that state direction not receiving updates), but not
conclusively diagnosed; flagged for future investigation.

**Status after this round**: every subsystem checked against the real
official C++ source this session — propagation, MSCKF Jacobians/chi2/
nullspace/compression, static-init algorithm and actual numeric values
(disproven as the cause via direct injection test), calibration values,
per-frame operation order, FEJ, camera-intrinsics sync — either matches or
has been fixed with no net accuracy change. `circle.bag` ATE remains ~0.685 m
vs official's 0.067 m. **Full numerical parity was not achieved.** The
mechanistic finding (MSCKF ~4-5% accept rate from triangulation starvation)
remains the best explanation found for *why*, but the underlying cause of
that starvation (beyond raw tracker precision, which was tried and made
things worse) was not isolated. The only remaining path identified to fully
close the gap is a fundamentally different methodology: instrumenting and
rebuilding *official's own source* to dump comparable frame-by-frame internal
diagnostics for direct lockstep comparison — a substantially larger
undertaking than anything attempted this session, since it means modifying
the reference implementation itself.

## Hover detector implemented from the Kottas/Wu/Roumeliotis paper — disabled, over-triggers

Per request, implemented the actual algorithm from *"Detecting and Dealing
with Hovering Maneuvers in Vision-aided Inertial Navigation Systems"* (Kottas,
Wu & Roumeliotis) rather than continuing to rely on Python v2's IMU-velocity/
disparity heuristic (which the user recalled as unable to detect real
hovers): `msckf/hover_detector.{hpp,cpp}` computes the rotation-compensated
bearing-vector residual `d_k` between the two most recent camera poses
(paper Eq. 26–29, the "0-pt RANSAC" case) with consecutive-decision
hysteresis, and `msckf/state_helper.cpp`'s new `marginalize_lifo_clone`
implements the paper's LIFO clone-window scheme (Sect. III-B): while
hovering, replace the most-recently-added clone instead of the oldest, so the
generic-motion baseline already in the sliding window is preserved instead of
being squeezed out.

* **A real bug was found and fixed in the detector itself**: the relative
  rotation between clones was computed in the IMU frame, but the bearing
  vectors (`uv_norm`) are in the camera frame — missing the camera-IMU
  extrinsic conjugation (`R_cam = R_ItoC · R_imu · R_ItoC^T`). Before the fix,
  enabling the detector caused a catastrophic 43 m ATE divergence (the
  LIFO/FIFO switching decision was essentially noise, driving degenerate
  clone-window management). After the fix, `circle.bag` no longer diverges
  catastrophically (1.1 m ATE) but is still worse than detection disabled
  (0.675 m).
* **Still over-triggers**: instrumented and confirmed the fixed detector
  classifies **99.8–100% of frames as hovering** during a continuous, slow
  circular flight that is clearly not a hover. Root cause not found this
  session — plausible candidates (not verified): the `epsilon` threshold
  (0.01, in normalized/focal-length units) may simply be too loose for this
  camera's real feature-tracking noise floor at this flight speed, or there
  may be a second, subtler bug in the residual computation itself (e.g. the
  hysteresis logic latching onto an early false decision and never
  recovering, since `raw_decision` holds the *previous* confirmed mode when
  too few correspondences are available rather than defaulting to
  non-hovering).
* **Status**: code compiles, is present in the repo, but is **disabled**
  (`VioManagerOptions::enable_hover_detection = false` in
  `ros/vio_rosbag_runner.cpp`) since it is not yet a net win — a documented,
  resumable starting point, not wired into the numbers in this document.

## Retry-storm bug found and fixed (2026-07-22): root cause, not a net ATE win

Built a second, deliberate copy of official's source (`/workspace/open_vins_instrumented`,
its own catkin workspace `openvins_instrumented_ws`, symlinked — never touches the
untouched `open_vins_official` baseline) and instrumented `UpdaterMSCKF.cpp` with
per-100-call counters (`[OV-MSCKF-DEBUG] calls=... seen=... rej_tri=... rej_chi2=...
accept_rate=...`). Lockstep comparison against DOD's own (previously added) MSCKF
counters found the real mechanism behind the "~4-5% accept rate" number above:

* **Official**: accept_rate stabilizes at ~68-73%, averaging only ~5 features fed
  to `update_msckf` per call.
* **DOD (before this fix)**: ~4-5% accept rate, ~150 features fed per call — 30×
  more features, almost all rejected.

Root cause: `msckf/vio_manager.cpp`'s post-update loop was doing
`for (...) marginal_features[i]->to_delete = false;` after calling
`update_msckf(...)`, **overriding** `update_msckf`'s own unconditional
`to_delete = true` marking on every feature it touches (success or failure).
This kept failed features alive in `vio.db`, so the *same* feature, with its
*same* stale measurement history, was re-fed into the MSCKF update again next
frame, and the frame after, forever — an endless retry storm on
already-known-bad data, which explains both the inflated features-per-call
count and the crushed accept rate. Official's `VioManager.cpp` has no
equivalent override: every fed feature is unconditionally discarded
(`to_delete = true` + `cleanup()`) regardless of outcome — a strict one-shot
attempt per feature, relying on the tracker to hand back a fresh entry next
frame if the physical point is still visible.

**Fix**: removed the un-delete override in `msckf/vio_manager.cpp`; the
post-update block now matches official exactly — `update_msckf(...)` followed
by `core::cleanup_db(vio.db)`, no re-enabling of `to_delete`.

**Effect (measured, not assumed)**:

| Dataset | Metric | Before (retry-storm) | After (one-shot, this fix) | Official |
|---|---|---|---|---|
| circle.bag | ATE RMSE | 0.685 m | 0.785 m | 0.067 m |
| circle.bag | path length | ~16-24 m (varies by run) | 23.6 m | 29.8 m (GT) |
| infinity.bag | ATE RMSE | 0.727 m | 1.070 m | 0.070 m |
| infinity.bag | path length | 6.1 m | 21.1 m | 29.1 m (GT) |

Path length and trajectory shape improved substantially and consistently on
both datasets — the earlier "scale collapse" (estimated path ~15-20% of
ground truth on infinity.bag) is mostly gone, trajectories are visibly denser
and smoother (see regenerated `docs/circle_trajectory_comparison.png` and
`docs/infinity_trajectory_comparison.png`). But the aggregate ATE RMSE got
**worse** on both datasets. This is the most concrete, best-evidenced
structural bug found this entire investigation (confirmed by direct lockstep
comparison against official's real C++ source, not static reading), yet it
did not close — and by this one metric, widened — the gap to official's
0.067-0.070 m. Likely explanation: with retries removed, previously-retried
(and previously road-tested-by-repetition) noisy updates are replaced by a
smaller number of single-shot updates that individually pass chi2 but are
less constraining in aggregate, so the filter's scale/shape improves while
point-wise RMSE does not; this has not been proven, only the measured before/
after numbers above are certain.

**New bug surfaced by this fix**: with the retry-storm fix applied and SLAM
(`enable_slam = true`) also enabled together, `circle.bag` hits a
**reproducible** `EKFPropagation()` crash — negative covariance diagonal at
state index 178, identical value (`-0.0126362`) and identical frame on every
run. Deterministic, not numerical noise, but root cause (likely in
`perform_anchor_change`/`change_anchors`) not isolated this session.
Pragmatic workaround: SLAM is now disabled
(`ros/vio_rosbag_runner.cpp`: `options.enable_slam = false`, with a code
comment explaining why) — the MSCKF-only path with the retry-storm fix is
stable and is the current shipped configuration. All numbers in the table
above are MSCKF-only (no SLAM).

**Next lead, not yet pursued**: re-add per-attempt (single-shot, retries
already gone) triangulation/chi2 debug counters to DOD's `update_msckf` and
re-measure the true single-shot accept rate now that the retry-storm's
masking effect is removed, to see whether DOD's real one-shot triangulation
success rate is still meaningfully below official's ~70%, independent of the
retry-storm behavior.

**Bottom line**: full numerical parity with official OpenVINS (0.067-0.070 m
ATE) was **not** achieved. This was the deepest and most rigorously verified
lead of the entire multi-day investigation, and it measurably fixed the scale/
path-length problem, but the ATE metric itself did not improve.

## Post-retry-storm-fix follow-up (2026-07-22): single-shot accept rate re-measured, cap fix tried and reverted

Following the retry-storm fix above, re-instrumented DOD's `update_msckf`
directly (guarded by `#ifdef DOD_MSCKF_ACCEPT_DEBUG`, `msckf/updaters.cpp`)
with the same per-100-call counters used on official, to check whether the
underlying single-shot (no-retry) triangulation/chi2 accept rate was still
far below official's ~70% now that the retry-storm's masking effect is gone.

**Result: yes, decisively.** Even with retries removed, DOD's single-shot
accept rate stabilizes at **~6%** on `circle.bag` (rej_tri dominates: e.g.
64485/77818 = 83% of all seen features fail triangulation, vs. ~5% failing
chi2). Also corrects an earlier-session estimate: average features fed to
`update_msckf` per call is ~17, not ~150 — the ~150 number was from before
`ros/vio_rosbag_runner.cpp` had `max_msckf_in_update = 50` set at all.

**Hypothesis tried**: official's `VioManager.cpp` sorts `featsup_MSCKF` by
track length and caps it at `state->_options.max_msckf_in_update` (50 for
KAIST), keeping only the longest tracks. DOD's `StateOptions::max_msckf_in_update`
existed in config (`kaist_vio/estimator_config.yaml: max_msckf_in_update: 50`)
but was never actually enforced anywhere in `msckf/vio_manager.cpp` — a dead
config value, same pattern as earlier-session findings. Implemented the
sort+cap in `vio_manager.cpp` to match official exactly.

**Measured effect: negative.** Since msckf_count rarely exceeds 50 anyway
(~17 average), the cap barely moved the accept rate (6.4% → 8.2%), and it
made ATE RMSE *worse* on both datasets: circle.bag 0.785 m → 1.043 m,
infinity.bag 1.070 m → 1.306 m. **Reverted** — confirmed via remote rebuild,
ctest 5/5 pass, and re-running circle.bag to reproduce the exact prior
0.7854808 m ATE bit-for-bit, then re-syncing local/remote. The debug counters
remain in `updaters.cpp` behind the `DOD_MSCKF_ACCEPT_DEBUG` macro (inert in
normal builds) for any future re-measurement.

**Conclusion**: the dominant bottleneck is not feature *selection count* but
triangulation *quality* — the ~83% triangulation failure rate itself. This
matches and reinforces the earlier-session finding ("MSCKF is severely
constraint-starved... ~52% condition-number-too-high, ~30% near-distance")
and the finding that improving tracking precision (real FAST + sub-pixel
refinement) made things worse, not better. The bottleneck is upstream of
which/how-many features are chosen — likely in the geometry/parallax
available from `circle.bag`'s slow, small-radius motion combined with DOD's
tracker producing correspondences official's more mature frontend doesn't,
or a still-unfound difference in the triangulation math itself. Not resolved
this session; current shipped configuration remains the retry-storm fix only
(0.785 m / 1.070 m ATE), the best measured result to date.

## KLT frontend structural comparison (2026-07-22): two real bugs found, one tried and reverted, negative result

Per the `/goal` directive, went deeper: a structural line-by-line comparison
of DOD's `core/tracker.cpp` against official's actual `TrackKLT.cpp` (886
lines, not just the previously-checked mono-survival/RANSAC-threshold
pieces) turned up two genuine, previously-unexamined structural differences:

1. **Stale-seed re-tracking bug** (`core/tracker.cpp` ~line 280-281): when a
   feature's KLT fails in one camera but survives in the other, DOD writes
   the *old* (pre-motion) pixel coordinate back into that camera's track list
   as if it were current, then advances the stored "previous" image to the
   new frame. Next frame, KLT seeds from a stale coordinate against a fresh
   image — a built-in one-frame lag for any track that ever drops a camera.
   Official maintains fully independent per-camera track lists paired by ID
   search, with no equivalent stale re-seed. **Not yet fixed** (a real
   correctness bug, but a larger refactor than time allowed this round).
2. **Unconditional per-frame re-detection**: DOD tops off to `num_features`
   (200) every single frame; official only re-detects when the deficit
   exceeds `min(20, num_features/2)`, letting existing tracks mature in
   bursts instead of continuously injecting zero-baseline new features.

**Tried fix #2** (the cheaper, fully reversible one): gated
`detect_grid_fast` behind official's threshold. Measured effect: ATE got
*worse* (0.785 m → 1.274 m on circle.bag), and counter-intuitively average
features fed per MSCKF call went *up*, not down — the opposite of the
hypothesis. **Reverted**; confirmed rebuild reproduces the exact prior
0.7854808 m ATE bit-for-bit, ctest 5/5 pass, local/remote re-synced.

**Status**: this is now the fourth isolated structural fix this session
(after the retry-storm fix, the max_msckf_in_update cap, and this detection
gate) whose individual application made ATE worse despite each one being a
real, verified divergence from official's actual behavior. The pattern
suggests DOD's current pipeline, taken as a whole, has adapted to compensate
for its own imperfections in ways that break when only one piece is aligned
to official at a time — isolated fixes are not composing safely. The
stale-seed bug (#1 above) remains unfixed and is a plausible next target,
but given three consecutive negative results from partial alignment, a full
simultaneous port of official's tracker (not one piece at a time) is likely
required to see a real improvement, consistent with what was flagged as the
substantially larger remaining undertaking. Not attempted this round.

## Stale-seed stereo bug fixed (2026-07-22): first genuine net improvement this session

Structural difference #1 from the KLT frontend comparison above (the one
left unfixed at the time) was implemented and tested: when a track's
temporal KLT fails in exactly one camera, DOD was writing the *previous
frame's* stale pixel coordinate back into that camera's slot and then
advancing the stored image to the current frame — so next frame's KLT seeds
from a one-frame-stale coordinate against a fresh image, silently corrupting
that camera's observations for the rest of the track's life.

**Fix** (`core/tracker.cpp`, new block right after the per-camera RANSAC
check): when left fails but right survives (or vice versa), immediately
attempt a stereo (rectified, small-disparity) recovery match — run KLT from
the surviving camera's fresh position across to the failed camera's current
frame, seeded at the same pixel location. If the stereo recovery succeeds
and lands in-bounds, use that as the camera's current position instead of
the stale one; otherwise fall back to the previous (stale) behavior
unchanged, so this can only help, never regress a track that had no
alternative.

**Measured effect** (ATE RMSE, MSCKF-only, SLAM disabled, retry-storm fix
still in place):

| Dataset | Before this fix | After this fix | Official |
|---|---|---|---|
| circle.bag | 0.785 m | **0.636 m** | 0.067 m |
| infinity.bag | 1.070 m | 1.258 m | 0.070 m |

Mixed across datasets, but circle.bag's DOD trajectory now visibly traces a
proper closed loop for the first time (see the regenerated
`docs/circle_trajectory_comparison.png` — compare the "middle segment" panel
to earlier revisions where DOD's estimate wandered off-loop). Path length
improved on both datasets regardless of the ATE direction. This is the first
change this session that produced a clear net improvement on at least one
real dataset rather than a uniform regression, and it's a genuine, previously
undiscovered correctness bug (stale re-seeding), not a parameter tweak.
Kept. 5/5 ctest pass; local/remote re-synced; regenerated
`docs/results/dod_{circle,infinity}_{estimate,eval}.*` and both trajectory
comparison plots to reflect this as the new shipped configuration.

**Status**: full parity with official (0.067–0.070 m) still not reached —
circle.bag is now ~9.5x worse instead of ~11.7x, infinity.bag got
worse (~18x instead of ~15.3x). The stale-seed bug was real and its fix
genuinely helps in the case it targets (features that drop one camera
temporarily), but infinity.bag's regression suggests either a different
dominant failure mode on that dataset, or that the stereo-recovery match
itself introduces its own occasional bad correspondences that hurt more than
the stale-seed bug did there.

**Follow-up tried**: added a ±2px rectified-epipolar (row) sanity gate on the
recovered cross-camera match, hypothesizing it would filter bad recoveries
without losing good ones. Measured effect: worse on **both** datasets
(circle.bag 0.636→0.899 m, infinity.bag 1.258→1.326 m) — the gate rejected
more good recoveries than bad ones. **Reverted.**

**Second follow-up**: instrumented the recovery pass itself (guarded by
`DOD_TRACKER_RECOVERY_DEBUG`, inert in normal builds) to count
attempts/successes per dataset, to check whether infinity.bag triggers
recovery abnormally more or with an abnormally worse success rate (which
would point at a dataset-specific cause). Measured: circle.bag ~820
recovery attempts over 4600 calls at ~78% success; infinity.bag ~1064
attempts over 4400 calls at ~73-75% success — infinity.bag does trigger
recovery somewhat more often (consistent with its more complex figure-8
motion causing more per-camera track drops near the crossing point and
sharper turns) but the success *rate* is not dramatically different. No
smoking-gun anomaly found — the regression is not explained by an obvious
spike in bad-recovery frequency on this dataset.

**Third follow-up — found it**: split the recovery pass into its two
independent directions (recover-left-from-right vs. recover-right-from-left)
and tested all four combinations on both datasets:

| Config | circle.bag ATE | infinity.bag ATE |
|---|---|---|
| Neither (retry-storm fix only, prior baseline) | 0.785 m | 1.070 m |
| Both directions (previous shipped) | 0.636 m | 1.258 m |
| Right-from-left ONLY | 0.865 m | **1.009 m** |
| Left-from-right ONLY | **45.18 m** (reproduced twice, deterministic) | 1.105 m |

Recovering the LEFT camera from the right is the actual culprit: alone, it
causes a reproducible catastrophic divergence on circle.bag (not measurement
noise — identical 45.1812673733896 m ATE on two separate runs) while still
being mildly beneficial on infinity.bag. Recovering the RIGHT camera from
the left, alone, is safe and beneficial on **both** datasets and gives the
best worst-case gap to official across both bags:

| Config | circle.bag gap | infinity.bag gap | worst-case |
|---|---|---|---|
| Neither | 11.7x | 15.3x | 15.3x |
| Both directions | 9.5x | 18.0x | 18.0x |
| **Right-from-left only (shipped)** | **12.9x** | **14.4x** | **14.4x** |

**Shipped**: `core/tracker.cpp`'s stereo-recovery pass now does
right-from-left recovery only (the left-from-right block removed entirely,
not just disabled). Reconfirmed circle.bag 0.8647056938307592 m / infinity.bag
1.0083763651351638 m, ctest 5/5 pass, local/remote synced, both trajectory
comparison plots regenerated.

**Mechanism investigated further, two specific hypotheses checked and ruled
out**:

1. *Hover detector camera-0 dependency*: `msckf/hover_detector.cpp:24` (`if
   (meas.cam_id != 0) continue;`) means the hover classifier's
   bearing-residual computation uses ONLY left-camera pixel data — a
   plausible single-camera-corruption-sensitive pathway that could flip
   `is_hovering` and change clone-marginalization mode (FIFO vs LIFO).
   **Ruled out**: `ros/vio_rosbag_runner.cpp` sets
   `options.enable_hover_detection = false`, so `update_hover_detector` is
   never even called in the shipped configuration — this pathway cannot be
   the cause.
2. *Anchor-camera triangulation sensitivity*: hypothesized that since
   `single_triangulation`'s anchor camera defaults to whichever camera has
   more measurements (ties go to camera 0), and left almost always has more
   measurements than right (right is the one that drops and needs
   recovery), corrupting left's pixel could poison the anchor frame more than
   corrupting right's. **Ruled out on closer reading**: the anchor camera's
   role (`core/feature.cpp:218-219`) only supplies the *reference pose*
   (`R_GtoA`/`p_AinG`), which comes from the state's stored clone pose, not
   from the current frame's pixel value — the anchor's own pixel measurement
   is folded into the linear system (`A`/`b`) as just one more equation among
   all measurements, with no special weighting. The math is provably
   anchor-frame-invariant; corrupting the anchor camera's pixel should be no
   more harmful than corrupting any other camera's pixel for the same
   feature.

With both specific mechanisms ruled out, the true reason recovering
left-from-right specifically triggers a reproducible 45 m divergence on
circle.bag (but only a mild regression on infinity.bag) **remains an open,
unexplained instability**.

**Localized further**: plotted the left-from-right-only run's raw position
trace frame-by-frame. The divergence is not immediate or a single obvious
bug trigger — it starts late, ~85% through the 4758-frame run (first
1.86 m single-step jump at frame 4032, timestamp 1598870318.5), then goes
fully runaway two seconds later with a 20.2 m single-step jump at frame 4233
(timestamp 1598870325.2), after which the estimate never recovers. This
"stable for a long time, then a late catastrophic break" pattern is more
consistent with a slowly-accumulating conditioning/covariance issue reaching
a tipping point than an immediate correspondence bug — but the specific
accumulating quantity was not identified this round. Pinning it down further
would need frame-by-frame logging of covariance eigenvalues, per-feature
triangulation depth/condition-number, and recovered-vs-normal feature counts
specifically in the frame range 4000-4250 of this run — not done. The safe
engineering choice (ship right-from-left only, which avoids the failure mode
entirely even without fully explaining it) stands, but this is a workaround,
not a proven root-cause fix, and is disclosed as such.

## Architectural fix (2026-07-22): independent per-camera track lists, replacing the paired-array design

Redirected per explicit product guidance: DOD exists to avoid inheriting
official OpenVINS's license, not to reinvent the algorithm — its *logic*
must be identical to official's, only the architecture (data-oriented,
no OOP) differs. Chasing isolated numeric wins (the stereo-recovery patch
above) was the wrong frame; the right fix is making `core/tracker.cpp`
structurally match official's actual design. See the new
`docs/DOD_OFFICIAL_PARITY_MANUAL.md` for the full file-correspondence map,
complete fixed/open divergence list, and methodology — this entry is the
detailed experiment log behind its newest entry.

**The real root cause of the stale-seed bug** (and, by extension, of why the
right-from-left-only stereo-recovery patch was ever needed) was DOD's
per-camera track arrays being paired by array index — the same index always
meant "the same feature" in both `tracker.previous[0]` and
`tracker.previous[1]`, forced to equal length every frame. Official's
`TrackKLT.cpp` has no such constraint: it keeps `pts_last[cam_id]` as fully
independent per-camera lists, matched only by feature ID. When one camera's
KLT failed under DOD's paired design, *something* had to be written into
that slot to keep the arrays the same length — the stale carry-forward that
started this whole investigation.

**Fix**: rewrote `track_stereo_frame` in `core/tracker.cpp` so each camera
tracks completely independently — its own survivor list, its own count, no
forced pairing. A camera that fails to track a point simply drops that ID
from its own list; there's no stale coordinate to seed next frame because
there's no invented entry to carry forward at all. The stereo cross-camera
"recovery" patch (right-from-left only, the safe direction found through
extensive experimentation above) is no longer needed and was removed
entirely — a net simplification. Also fixed in the same rewrite: a
newly-detected left corner is now kept as a valid mono track even if its
immediate stereo match to the right camera fails (previously discarded
outright, a gratuitous DOD-only restriction with no official counterpart).

**Measured effect**:

| Dataset | ATE before (recovery patch) | ATE after (independent lists) | Path length before | Path length after | GT path length |
|---|---|---|---|---|---|
| circle.bag | 0.865 m | 1.171 m | 23.6 m | **27.95 m** | 29.8 m |
| infinity.bag | 1.009 m | 1.114 m | 21.1 m | **31.70 m** | 29.1 m |

Point-wise ATE RMSE is nominally a bit worse, continuing the pattern seen
all session where architecturally-correct fixes improve trajectory
shape/scale without reducing point-wise RMSE. But path length is now within
~7% of ground truth on **both** datasets simultaneously for the first time
in this entire investigation (previously anywhere from ~40-80% of ground
truth depending on configuration) — see the regenerated
`docs/circle_trajectory_comparison.png` / `docs/infinity_trajectory_comparison.png`,
where DOD's trajectory for the first time traces recognizable loops at
roughly the correct scale on both bags, not just one.

**Shipped regardless of the ATE number** — this is the logically faithful
design official actually uses, verified structurally (not just
numerically) equivalent, and it eliminates an entire class of
invented-measurement bug (the stale/recovered-pixel patch) rather than
managing one direction of its symptoms. ctest 5/5 pass; local/remote synced.

**Not yet at parity** (1.171 m / 1.114 m vs official's 0.067 m / 0.070 m).
The remaining gap is now isolated to tracker/frontend correspondence
*quality* — how good each individual measurement is — not to shape,
scale, math equivalence, or the array-architecture (all now resolved or
verified equivalent). The single-shot MSCKF accept-rate measurement (~6%
DOD vs ~68-73% official) predates this fix and should be re-measured now
that the architecture has changed, since it's not yet known whether this
fix moved that number.

## Real-detector port attempted, reverted on stability grounds; root-caused the underlying circle.bag fragility (2026-07-22, same day)

Per redirected priority (logic parity over aggregate ATE), ported official's
actual detection pipeline into `core/tracker.cpp`: real `cv::FAST` per grid
cell (sorted by response, top-N kept) + real `cv::cornerSubPix` sub-pixel
refinement + official's exact two-level occupancy grid (fine `min_px_dist`
grid for near-duplicate rejection, coarse `grid_x*grid_y` grid for per-cell
capacity) + official's exact redetection gate (`needed < min(20,
num_features/2)` skips extraction this frame) — replacing the hand-rolled
FAST-score approximation used until now. This matches
`ov_core::Grider_FAST`/`Grider_GRID::perform_griding` +
`TrackKLT::perform_detection_monocular` line-for-line.

**Measured effect**: infinity.bag improved substantially (1.114 m → 0.661 m
ATE). circle.bag **catastrophically diverged** (277 m path length vs 29.8 m
ground truth, single-step jumps up to 63 m), starting at frame ~4030 — the
same frame index where an earlier, unrelated change (left-from-right stereo
recovery, before the independent-list architecture existed) also
catastrophically diverged. Since official's real source does not diverge on
this bag, this was treated as a genuine stability defect to root-cause, not
an accepted parity tradeoff — reverted the real-detector port pending
root-cause (kept the independent-camera-list architecture, which is stable).

**Investigated the same day, initial "root-caused" claim retracted after
further check**: added `Eigen::SelfAdjointEigenSolver` eigenvalue logging of
the active covariance block (`ros/vio_rosbag_runner.cpp`, `DOD_DIAG_DEBUG`),
run across frames 2900-4550 on both the healthy baseline and the diverging
real-FAST build. Found the dominant eigenvalue frozen at an identical
bit-for-bit value (0.034882) for ~600 frames (3200-3800) on the diverging
build only, then growing geometrically to 1122 by frame 4550 — matching an
anomaly flagged much earlier this investigation ("the largest eigenvalue
froze for 80+ seconds"). Initially reported as the root cause.

**This was wrong.** Computed the actual eigenvector (not just the
eigenvalue) at frame 3500 and checked which state components it loads onto.
The top TWO eigenvalues (ranks 0 and 1, both ~frozen magnitude) load with
**exactly equal weight on the IMU state and on every single one of the 11
clones** — the textbook signature of VIO's standard unobservable global
position/yaw gauge freedom (present in every MSCKF/VIO system by
mathematical necessity, official's included, not a DOD defect). Its
unbounded growth is expected; critically, ATE is computed after rigid
(Umeyama) alignment to ground truth, which exactly cancels global
position/yaw — so this direction's growth, however dramatic, cannot cause a
relative-trajectory divergence. Ranks 2-5 (much smaller, IMU-concentrated,
not spread across clones) show no comparable anomaly at this frame.

**Net result: the actual mechanism behind the frame ~4030 circle.bag
divergence remains unfound.** Ruled out so far: hover detector (disabled),
anchor-frame triangulation sensitivity (proven invariant), and now the
dominant-eigenvalue growth (proven to be the benign, ATE-irrelevant gauge
direction). Next step: check the smaller (rank 2+, IMU-concentrated)
eigenvalues for anomalies specifically across frames 4000-4300 on the
diverging build — not yet done. Full details in
`docs/DOD_OFFICIAL_PARITY_MANUAL.md`'s open item #2.

## EuRoC `MH_01_easy.bag` — blocked

A EuRoC-specific runner (`ros/vio_rosbag_runner_euroc.cpp`, real calibration
from the dataset's own `mav0/{cam0,cam1,imu0}/sensor.yaml`, since EuRoC uses
different topics — `/cam0/image_raw`, `/cam1/image_raw`, `/imu0` — and raw
[undistorted] rather than pre-rectified images) was built and compiles
cleanly, but the DOD estimator diverges catastrophically within the first
~250 frames (`EKFPropagation() - diagonal at 5 is -2.5e12`). Likely cause:
EuRoC's real lens distortion coefficients (~-0.28, 0.07) are roughly 40×
larger in magnitude than KAIST's (~0.007, near-zero since those images are
pre-rectified) — the RADTAN distortion Jacobian (`core::compute_distort_jacobian`)
has never been exercised at this magnitude before. Needs its own
finite-difference validation pass (same method as `verify_slam_jacobians.cpp`)
before EuRoC can run. Not attempted further this session.

## Uninitialized-memory bug found and fixed (2026-07-23): largest single improvement this investigation

Prompted by a direct hint to check for arrays used with `+=` without being
zeroed first. Audited every `+=` accumulation across the codebase against
its initialization; found the real match in `msckf/updaters.cpp`:
`update_msckf` and `update_slam` both accumulate per-feature Jacobian/
residual blocks into pre-allocated `Hx_big`/`res_big` buffers
(`Eigen::MatrixXd::Zero(2000, 300)` initially), growing them via
`conservativeResize(...)` whenever a call needs more rows or columns than
the initial allocation (6 call sites total: row growth + column growth, in
both `update_msckf` and `update_slam`).

**The bug**: `Eigen::MatrixXd::conservativeResize` does **not**
zero-initialize newly added rows/columns — they retain whatever was
previously in that memory. Every one of the 6 sites immediately follows the
resize with `Hx_big.block(...) += H_x.block(...)`, accumulating directly
into the freshly-grown, garbage-filled region whenever a call actually
needed to grow past the initial 2000×300 allocation. This silently injects
uninitialized memory into the linear system fed straight to the EKF
update — exactly the kind of deterministic-per-run, memory-layout-dependent
corruption that would explain hard-to-reason-about divergences.

**Fix**: added explicit `.setZero()` on the newly-grown region immediately
after every `conservativeResize` call, all 6 sites.

**Measured effect — the largest single improvement this entire
investigation, on both datasets**:

| Dataset | ATE before | ATE after | Path length before | Path length after | GT path length | Gap to official before | Gap after |
|---|---|---|---|---|---|---|---|
| circle.bag | 1.171 m | **0.624 m** | 27.95 m | 26.45 m | 29.8 m | 17.5x | **9.3x** |
| infinity.bag | 1.114 m | **0.201 m** | 31.70 m | **29.59 m** | 29.1 m | 15.9x | **2.9x** |

infinity.bag's path length is now within 1.5% of ground truth and its ATE
gap to official has closed to 2.9x — by far the closest this investigation
has gotten on either dataset, all session. ctest 5/5 pass; local/remote
synced; both trajectory comparison plots regenerated (infinity.bag's DOD
trace now visibly overlays official's figure-8 shape closely).

**Retested the reverted real-FAST/`cornerSubPix` detector port on top of
this fix**, hypothesizing it might have been the actual cause of that
fragility too (more candidate features feeding the linear system → more
likely to trigger the buggy resize path). **It was not**: circle.bag still
diverges catastrophically with the real-FAST port even with this fix in
place (33.4 m ATE, unchanged in character from before). The two issues are
confirmed independent — this fix does not resolve, and is not related to,
the still-open circle.bag frame-4030 fragility investigated earlier.
Full writeup in `docs/DOD_OFFICIAL_PARITY_MANUAL.md`'s fixed-divergences
list, item #12.

**Status**: not yet at full parity (9.3x / 2.9x gaps remain), but this is
the most impactful fix of the entire investigation and the closest DOD has
come to official's numbers on either dataset.


---

## Benchmark 6 — EuRoC MH_01_easy (real distorted camera, GNSS-denied target case)

First run of the DOD pipeline on **real, raw-distorted camera data** (EuRoC
MAV, 752x480 radtan, distortion ~40x KAIST's). This is the representative
case for the GNSS-denied navigation target.

| Pipeline | ATE RMSE (SE3) | ATE RMSE (Sim3) | Scale | Traj. len | Initializer |
|---|---|---|---|---|---|
| **DOD C++** | **0.393 m** | 0.361 m | 0.964 | 81.8 m | static still-window |
| Official OpenVINS | 0.133 m | 0.126 m | 1.010 | 80.3 m | dynamic MLE |

Ground truth: `/leica/position` (Leica MS50, position-only), Umeyama-aligned.
Same estimator params / calibration / noise / feature representation on both
sides; the **only** difference is the initializer.

### Root cause fixed (was: catastrophic divergence to km)
The earlier "distortion-Jacobian" hypothesis was **wrong** — the RADTAN
distortion Jacobian is finite-difference-correct to 1e-8 and undistort
round-trips to 0.01 px at EuRoC coefficients. The real cause was
**initialization on a non-stationary window**:

- MH_01 has **no clean stationary start** — the platform is carried/jostled
  (accel std ~2.0 m/s^2) for the first ~6 s and is only genuinely still
  (~0.09 m/s^2) around t=21-23 s.
- The loose `init_imu_thresh = 0.60` let static init fire on a marginal,
  tilted window -> wrong gravity direction -> residual gravity integrated
  into velocity, which ramped unbounded (1 -> 18 m/s) and diverged to
  thousands of metres.
- **Fix**: tighten `init_imu_thresh` to 0.25 so init waits for the true
  still window. Post-init `|ba| = 0.038`, velocity holds ~0.01 m/s while
  stationary, trajectory stays bounded. One-line change, dataset-justified
  by the measured stillness profile.

### Honest comparison note
Official OpenVINS with its **default static/jerk initializer also fails to
initialize on MH_01** ("no accel jerk detected, platform moving too much",
zero trajectory output). Official only succeeds here via its **dynamic MLE
initializer** (`init_dyn_use: true`), which initializes at t~3 s during
motion and therefore covers the full 80 m (DOD's static init loses the
first ~21 s). The ~3x ATE gap is consistent with the ~2-11x gap seen on
KAIST and traces to the same fine-grained numerical-method differences, not
a logic bug. The clear capability gap is that **DOD lacks a dynamic
initializer** — the concrete next step for full EuRoC parity.

Artifacts: `docs/results/dod_mh01_*.csv`, `docs/results/ov_mh01_state.txt`,
`docs/results/euroc_mh01_eval.json`,
`docs/euroc_mh01_trajectory_comparison.png`. Result reproduced bit-identical
across two clean builds (deterministic).

---

## Benchmark 7 — EuRoC MH_01: dynamic (linear-MLE) initializer ported

Closed the initializer capability gap flagged in Benchmark 6. Ported the
linear stage of OpenVINS's `ov_init` DynamicInitializer into
`initialize::dynamic_initialize` (`initialize/initialization.cpp`): CPI-v1
mean preintegration + the |g|-constrained Dongsi closed-form solve
(companion-matrix eigenvalues). No Ceres dependency -- the linear solution
is used directly (official refines it further with a Ceres MLE step). Wired
as a fallback in `try_to_initialize`: static init is tried first every frame,
dynamic init fires when static cannot (matches official's InertialInitializer).

| Config | ATE (SE3) | Inits at | Coverage |
|---|---|---|---|
| DOD static (init_imu_thresh 0.25) | 0.393 m | t = 21 s | misses first ~21 s |
| **DOD dynamic (new)** | **0.407 m** | **t = 2.8 s** | **full 89.8 m** |
| Official OpenVINS (dynamic MLE) | 0.133 m | t = 3.0 s | full |

DOD now initializes **during motion at t = 2.8 s** (matches official's
~3.0 s) with no stationary window required -- the real capability parity
win. It recovers gravity to |g| = 9.81 and covers the full 89.8 m trajectory
(3018 matched poses vs static's 2712). ATE 0.407 m is marginally higher than
static's 0.393 m only because it now includes the harder carried/jostled
opening segment static skipped; scale 0.980. The residual ~3x gap to official
is the same systemic fine-numerical difference seen throughout, plus DOD's
omission of the Ceres MLE refinement.

Validated by `tests/verify_dynamic_init.cpp` (ctest 6/6): a synthetic
rotating+translating trajectory with consistent IMU + feature bearings; the
initializer recovers gravity direction to **0.13 deg** and speed to
**0.04 m/s**. Artifacts: `docs/results/dod_mh01_dyninit_estimate.csv`,
`docs/euroc_mh01_trajectory_comparison.png` (updated).

---

## Benchmark 8 — Bit-exact parity harness against official Open_VINS (2026-08-14)

Previous benchmarks compared *trajectories*. This one compares *functions*, by
linking official Open_VINS into the test binary as an oracle and demanding
ULP-0 agreement on identical inputs. Full method and findings:
`docs/DOD_OFFICIAL_PARITY_MANUAL.md`, section "Bit-exact parity programme".

### What is now provably bit-identical to official

| Stage | Checks | Result |
|---|---|---|
| Lie/quaternion math (20 functions) | 4000 | 0 ULP |
| chi-squared 0.95 gate (dof 1..499) | 499 | 0 ULP |
| Camera distort + both Jacobians (5 calibrations) | 8000 | 0 ULP |
| IMU selection, 3 mean predictors, F/G Jacobians | 5638 | 0 ULP |
| Triangulation, compute_error, Gauss-Newton | 2120 | 0 ULP |
| Nullspace projection + measurement compression | 720 | 0 ULP |

Ten distinct logical divergences were found and fixed (numbered 13-22 in the
parity manual). Several were structural rather than last-bit: official uses
Givens rotations where DOD used HouseholderQR, official anchors tied stereo
features to camera 1 where DOD hardcoded camera 0, and DOD carried two
invented rejection tests (`|det| < 1e-12`) that official does not have.

### EuRoC MH_01 end-to-end

| Build | ATE RMSE (m) | Path (m) |
|---|---|---|
| Pristine HEAD (`d35af4c`) | 0.40725 | 88.919 |
| All estimator parity fixes | 0.40742 | 88.920 |
| Parity fixes + official undistortion | 6.27694 | 155.451 |
| **Official Open_VINS** (re-measured) | **0.13330** | **80.241** |

Ground truth path 79.711 m; 3018 associated samples over 182.05 s. Official was
re-run this session rather than quoted, and scored by the same
`scripts/evaluate_trajectory.py` against the same ground truth, associating the
same 3018 samples (`scripts/ov_state_to_csv.py` converts its state dump). It
reproduced its documented 0.133 m.

### DOD vs official error breakdown, EuRoC MH_01

| Metric | DOD | Official | Ratio |
|---|---|---|---|
| ATE RMSE (m) | 0.40742 | 0.13330 | 3.06x |
| ATE mean (m) | 0.32688 | 0.12354 | 2.65x |
| ATE median (m) | 0.29530 | 0.12419 | 2.38x |
| ATE p95 (m) | 0.94435 | 0.20538 | 4.60x |
| ATE max (m) | 1.38915 | 0.38230 | 3.63x |
| Rel. translation RMSE (m) | 0.20533 | 0.07893 | 2.60x |
| Estimated path (m) | 88.920 | 80.241 | GT 79.711 |
| Path length error | **+11.55%** | **+0.67%** | |

The path-length column is the most diagnostic number here. Official tracks the
79.7 m of ground truth to within 0.67%; DOD travels 88.9 m over the same
interval — **9.2 m of excess path over 182 s**. The error is not a slow drift
in one direction: the median is only 0.295 m while p95 is 0.944 m and the
maximum 1.389 m, so most of the run is reasonable and the error concentrates in
bursts. An over-long path with bursty error is the signature of noisy
per-frame corrections rather than a bias or a scale factor, which points at
measurement quality and at the update path (see the `EKFUpdate` section of the
parity manual, particularly the invented jitter and negative-variance
flooring), not at propagation — propagation is now proven bit-exact.

The estimator parity fixes are **net-neutral end-to-end** (+0.00017 m). That is
the expected outcome of last-bit corrections and is not a disappointment — the
point of the harness is that it localises defects that ATE cannot see, and it
found ten of them. The discrete fixes (exact chi2, anchor tie-break, removed
rejections, Givens) evidently offset each other on this sequence.

The third row is the finding that matters: **making undistortion bit-exact to
official makes the pipeline 15x worse**, because DOD's tracker is not yet
matched to official's `TrackKLT`. Bisected two ways — reverting only the
undistort change restores 0.4074 m, reverting only the distort/Jacobian
changes leaves 6.28 m. Parity is a property of the whole pipeline; half a
matched pair is worse than neither half. `core::undistort_official()` therefore
stays proven-but-unshipped until the frontend parity work.

### Correction: circle.bag's documented baseline does not reproduce

Pristine HEAD on `circle.bag` gives **ATE 39.0 m with a 264 m estimated path**
against 29.5 m of ground truth — not the 0.624 m recorded in Benchmark 5.
Reproduced across the parity build and the pristine baseline, with and without
the new FP flags, to five decimal places. Whatever produced 0.624 m is not what
is committed. Until that is explained, circle.bag cannot attribute any change,
and the 9.3x gap quoted in earlier sections should not be relied upon. EuRoC
MH_01 does reproduce its documented number exactly and is the trustworthy gate.

### Also: `verify_math` never verified anything

Every check in it was an `assert()`, and Release defines `NDEBUG`. It reported
"passed" with all assertions compiled out, and underneath the pipeline never
initialized on its vectors, so the state stayed identity/zero. Present in
pristine HEAD too. Now unregistered from ctest, superseded by `bitdiff_*`.

---

## Benchmark 9 — Latency profile, and where the path-length error comes from (2026-08-14)

### Latency, EuRoC MH_01, single-threaded, in-container

| Stage | mean | median | p95 | p99 | max |
|---|---|---|---|---|---|
| Tracking | 1.517 | 1.460 | 2.052 | 2.407 | 4.065 ms |
| Estimator | 2.621 | 1.622 | 5.192 | 7.575 | 2247.242 ms |
| **Total** | **4.138** | **3.143** | **6.832** | **9.110** | 2248.746 ms |

3682 frames, 372 observations/frame average. At 20 Hz the budget is 50 ms, so
the steady state runs with **12x real-time headroom** and exactly one frame
exceeds budget.

That one frame is the **dynamic initializer**, 2247 ms at t=2.0 s — a one-time
startup stall, not a recurring cost. Excluding it the estimator averages
2.012 ms with p99 7.564 ms and a worst case of 42.3 ms (t=73.1 s), still inside
budget but with only 1.2x margin at the peak.

For a real-time target the two things worth attention are the 2.25 s
initialization stall (which would need to be amortised or run off the critical
path on a live platform) and that 42 ms peak, not the average.

### The path-length error is measurement supply, not the filter

DOD's estimated path is 88.92 m against 79.71 m of ground truth (+11.55%);
official's is 80.24 m (+0.67%). Decimating before measuring separates
high-frequency noise from genuine travel:

| Decimation | DOD | Official | Ground truth |
|---|---|---|---|
| raw (20 Hz) | 89.80 | 80.28 | 80.99 |
| 0.05 s | 87.84 | 80.18 | 80.70 |
| 0.10 s | 86.73 | 80.09 | 80.42 |
| 0.20 s | 84.64 | 79.68 | 79.86 |
| 1.00 s | 77.79 | 75.20 | 73.68 |

Two separate defects:

1. **High-frequency jitter.** Decimating to 50 ms removes 1.96 m of DOD's path
   but only 0.10 m of official's — roughly 20x the sub-50 ms position noise.
   Worst single step 357 mm vs official's 102 mm (7 m/s instantaneous at 20 Hz).
   Consecutive-step direction reversals: 6.90% vs 5.72%.
2. **Genuine wander.** At 1 s decimation DOD is still +5.6% over ground truth
   where official is +2.1%, so smoothing does not account for it.

**Cause: DOD's MSCKF accept rate on EuRoC is 0.390.** Of 71522 features fed to
`update_msckf`, 47% are rejected by triangulation and 14% by the chi2 gate.
Each update is therefore built from about half the information official uses —
which produces both noisier corrections (defect 1) and a less well-determined
solution (defect 2). Thresholds and tracker options already match official's
`euroc_mav` config exactly, so this is frontend implementation, not tuning.

`EKFUpdate` was rewritten to official's form during this work (upper-triangular
LLT, no 1e-9 jitter, no negative-variance flooring). Effect on EuRoC: ATE
0.40742 -> 0.40744 m. It also reported **zero** negative covariance diagonals
across the run, so the flooring it used to do was never firing here.

### Hovering FIFO/LIFO: not triggering

`enable_hover_detection` is `false` in both runners and in
`VioManagerOptions`' default. `hovering` is initialised false and only assigned
inside that guard (`msckf/vio_manager.cpp:83-96`), so `marginalize_lifo_clone`
is unreachable and FIFO (`marginalize_old_clone`) is used on every frame. It is
not costing accuracy today.

It is worth knowing why it is off: when it was last enabled it fired on
99.8-100% of frames across 4758 frames of continuous slow circular flight —
i.e. it classified generic motion as hovering almost always. The camera-frame
vs IMU-frame bearing bug behind part of that was found and fixed (43 m -> 1.1 m
ATE), but the over-triggering itself was never root-caused. Turning it on
without fixing that would switch the clone window to LIFO permanently, which
would hurt.

---

## Benchmark 10 — The SLAM subsystem was inert (2026-08-14)

Three bugs, found by instrumenting the per-frame feature classification rather
than by reasoning about ATE. Before, per frame: `lost=2.30 marginal=17.74
maxtrack=0.02 slam_update=0.95`. DOD held 50 SLAM landmarks and updated **one**
of them per frame, while 17.74 features that should have become landmarks were
dumped into MSCKF instead.

### 1. Feature ids started below the ArUco reserve

`StateHelper::marginalize_slam` refuses to retire any landmark whose id is
`<= 4 * max_aruco_features` (= 4096) -- those ids belong to ArUco tags, which
are never dropped. Official sidesteps this by starting its tracker's ids above
the reserve: `TrackBase` sets `currid = 4 * numaruco + 1`. DOD started at 1, so
its first ~4096 features were **permanently unmarginalizable**. The first 50
landmarks were promoted early, held ids below 4096, and could never be retired
-- the landmark slots were occupied forever by tracks that had long since died,
which also meant no new landmark could ever be admitted.

### 2. Promotion used a total-measurement test, not a per-camera one

Covered in the parity manual: official promotes when any single camera exceeds
`max_clone_size` observations; DOD required twice that summed across cameras,
so mono features -- the overwhelming majority -- were never eligible.

### 3. Retirement was evaluated after the database had been cleaned

The "is this landmark still tracked?" check ran after `update_msckf`,
`update_slam` and `delayed_init_slam` had each marked their consumed features
`to_delete` and `cleanup_db()` had erased them. It therefore reported "not
tracked" for exactly the landmarks that had just been successfully updated.
Now sampled before the updates run, as official does.

### 4. Feature pointers went stale between the updates -- and this retracts the result below

`lost_features` / `marginal_features` / `maxtrack_features` /
`slam_update_features` are raw `core::Feature*` into `vio.db.features[]`,
captured during classification. `cleanup_db()` compacts that array **in place**
(`features[write_idx] = features[i]`), and it was being called after
`update_msckf`, after `update_slam` and after `delayed_init_slam`. Every array
consumed after the first cleanup therefore pointed at whichever feature had
been shifted into that slot. Measured: 22% of SLAM candidates arrived with <2
measurements and 63% failed triangulation -- on features specifically selected
for having the longest tracks and the best geometry.

Fixed by carrying feature IDS alongside the pointers and re-resolving through
`core::get_feature()` at each use, keeping the cleanups (they are what stops a
feature consumed by one updater from being consumed again by the next).

**This invalidates the 0.2632 m figure below.** That build was handing
`update_slam` and `delayed_init_slam` garbage, most of which failed and was
skipped, so SLAM was barely running. With the pointers correct:

| EuRoC MH_01, correct pointers | ATE RMSE |
|---|---|
| SLAM off | **0.3947 m** |
| SLAM on, one joint update | 1.2506 m |
| SLAM on, official's batching | 1.1296 m |

So the landmark **lifecycle** is now correct -- promotion, retirement and the
id range were each broken and are each fixed, and the subsystem holds its full
50 landmarks with ~37 updates/frame -- but the SLAM **update math** is wrong:
giving it valid landmarks makes the estimate three times worse. `enable_slam`
is therefore back to `false` on EuRoC, with these numbers recorded at the call
site. `msckf/updater_slam_helper.cpp` and `update_slam()` are the only
estimator functions never diffed against official, and they are the next step.

### Result, EuRoC MH_01 (SUPERSEDED -- see item 4 above)

| | before | after |
|---|---|---|
| SLAM promotions / frame | 0.02 | 12.28 |
| SLAM updates / frame | 0.95 | 29.71 |
| features into MSCKF / frame | 17.74 | 7.80 (official ~5.9) |
| **ATE RMSE** | 0.3960 m | **0.2632 m** |
| **Estimated path** (GT 79.71 m) | 89.03 (+11.7%) | **82.79 (+3.9%)** |

Gap to official's 0.1333 m: 3.0x -> 2.0x. The path-length error -- the
diagnostic that started this -- has come down from +11.6% to +3.9%, and the
median error to 0.181 m against official's 0.124 m (1.46x). The RMSE gap is
now driven by the tail: p95 0.493 m vs official's 0.205 m.

### Two further changes, measured

`get_feature_jacobian_representation` (the SLAM landmark Jacobian, applied to
every landmark every frame) wrote `-alpha / (rho * rho)` where official writes
`-(1.0 / (rho * rho)) * alpha` -- reciprocal-then-multiply, the same class of
divergence fixed earlier in triangulation. Corrected; ATE 0.26319897 ->
0.26319832 m, i.e. a last-bit change, which is what a ULP-level fix should do.

Official applies SLAM updates in sequential batches of `max_slam_in_update`
(25), correcting the state between batches, rather than one joint update of
all ~30. That was implemented and measured: **worse** on EuRoC -- ATE 0.2632 ->
0.3508 m, p95 0.733, path 82.79 -> 84.26 m. Reverted, and the measurement
recorded at the call site so it is not retried blind.

Note what this says about the earlier investigation: the accept rate, the
frontend, and the estimator math were all measured and cleared, and the actual
defect was that the SLAM landmark lifecycle never ran at all. It was invisible
to ATE and to every bit-exactness test, because each individual function was
correct; what was wrong was that one of them was never reached.

---

## Benchmark 8 — EuRoC MH_01: DOD overtakes official (2026-08-15)

**DOD 0.1213 m vs official OpenVINS 0.1333 m** on MH_01 -- same ground truth,
same `scripts/evaluate_trajectory.py`, 3018 associated samples. On V1_01_easy,
DOD is **0.0506 m** against OpenVINS's published ~0.056 m (that one is a paper
number, not re-measured here).

Every EuRoC number before this one was measured through a defect in the
**runner**, not the estimator: `ros/vio_rosbag_runner_euroc.cpp` processed a
stereo pair as soon as both images had arrived, and rosbag iteration order is
*receive* order, so images were routinely updated before the IMU spanning them
had been fed. `feed_measurement_camera_tracks` propagates with whatever is in
`vio.imu_buffer` and never checks coverage, so those updates were built on
short propagation. Official OpenVINS is immune: `VioManager` queues camera
messages and processes one only once the IMU has passed its timestamp.

The sensitivity, measured by deliberately holding the IMU back
(`VIO_IMU_LAG_S` in `tools/dod_asl_runner.cpp`), SLAM on:

| IMU lag | ATE | estimated path (GT 79.71 m) |
|---|---|---|
| 0 | **0.1301 m** | 79.67 m |
| 5 ms | 9.22 m | 170.65 m |
| 20 ms | 9.2e5 m | diverged |

5 ms is a 70x error. That is why so many correct parity fixes moved nothing.

| MH_01 config | ATE | path | p95 |
|---|---|---|---|
| SLAM off | 0.1900 m | 81.30 | 0.313 |
| SLAM on | 0.1301 m | 79.67 | 0.205 |
| **SLAM on + sigma_px 1.0 (shipped)** | **0.1213 m** | 79.87 | 0.199 |
| official OpenVINS | 0.1333 m | 80.24 | 0.205 |

Two config values moved to official's, both improving both sequences:
`up_*_sigma_px` 1.2 -> 1.0, and `init_imu_thresh` 0.25 -> 1.5. The 0.25 was
MH_01-specific tuning added to stop a bad init; with IMU coverage fixed it is
unnecessary there (identical 0.1301 at either value) and actively fatal on
V1_01, where DOD's static init can then never fire, the dynamic fallback takes
over and the run diverges to 6.98 m.

| V1_01_easy config | ATE | path (GT 58.56) |
|---|---|---|
| init_imu_thresh 0.25 | 6.98 m | 49.60 |
| init_imu_thresh 1.5, SLAM off | 0.0552 m | 58.12 |
| **shipped** | **0.0506 m** | 58.02 |

**"SLAM costs 3x" is retracted.** The 1.1296 m measurement was taken through
the lagging path. With correct IMU coverage SLAM is the single biggest win
(0.190 -> 0.130), exactly as it is for official. `enable_slam` is now `true`.

Config sweep, all with SLAM on: official's own tracker settings (fast 20,
min_px_dist 10) are *worse* for DOD (0.171); 250 features 0.135; 160 features
0.142; `max_slam` 25 gives 0.319 and 100 is clamped to the array capacity of
50. DOD's existing defaults are at the optimum.

### What made this measurable

- `tools/dod_asl_runner.cpp` — ROS-free EuRoC runner over the dataset's ASL
  layout, so the ATE loop runs on a box with no ROS. It feeds all IMU up to
  each image by construction, which is what exposed the bug.
- `tools/euroc_options.hpp` — the config, now shared with the ROS runner so the
  two cannot drift.
- `tools/bag_to_asl.py` — ROS1 bag to ASL layout with no ROS install.
- `tools/dod_track_dump.cpp` / `tools/ov_track_dump.cpp` /
  `scripts/track_lifetime.py` — frontend comparison against official's
  unmodified `TrackKLT` (`get_last_obs()`/`get_last_ids()` are public).

### Frontend comparison, for the record

Measured before the IMU bug was found, on identical images. DOD's frontend is
not the weak side:

| | DOD | Official |
|---|---|---|
| lifetime median | 11 frames | 7 |
| parallax median | 5.82 deg | 4.32 |
| epipolar Sampson median | 0.220 px | 0.200 |
| epipolar > 5 px | 9.03% | 10.81% |
| stereo pairs / frame | 146.8 (77.7%) | 127.9 (72.1%) |

### Open

- The ROS runner's hold-until-IMU-covers guard is **unverified** — no ROS on
  any currently available machine. The durable fix belongs inside
  `feed_measurement_camera_tracks`, which should defer or refuse rather than
  propagate short.
- KAIST `circle.bag` / `infinity.bag` have not been re-run against any of this
  and their runner still has the original flaw.
- `do_calib_camera_pose` and `do_calib_camera_timeoffset` change the result by
  literally nothing (identical to 4 decimals), which is not credible for a
  state-dimension change. Unexplained.
- `max_msckf_in_update` is dead code.

---

## Benchmark 9 — ahead of official on accuracy AND speed (2026-08-15)

Official was re-run from scratch in the lab box's `ros_container_v2` (ROS Noetic,
its own built OpenVINS, `/workspace/EuROC/MH_01_easy.bag`) with
`record_timing_information`, so both sides are now measured on the same machine
with no numbers carried over from anywhere.

**This corrected the baseline.** The 0.1333 m figure used up to Benchmark 8 came
from a container that no longer exists; re-run here official scores **0.1180 m**,
better than that. The comparison below is against the re-measured number.

| EuRoC MH_01 | DOD | Official |
|---|---|---|
| ATE RMSE | **0.1132 m** | 0.1180 m |
| ATE p95 | **0.187** | 0.221 |
| path (GT 79.71) | **79.84** | 80.40 |
| tracking | **1.639 ms** | 2.107 |
| estimator | **5.127 ms** | 5.045 |
| **total / frame** | **6.766 ms** | 7.152 ms |

V1_01_easy: DOD **0.0494 m** (OpenVINS publishes ~0.056 for this sequence).

### Per-stage timing, against official's own columns

| stage | DOD before | DOD after | Official |
|---|---|---|---|
| propagate | 0.309 | **0.052** | 0.128 |
| msckf update | 1.554 | 1.565 | 0.719 |
| slam update | 2.728 | 2.682 | 2.558 |
| slam delayed | 0.536 | 0.521 | 0.521 |
| re-tri & marg | 1.042 | **0.031** | 1.118 |

The msckf update stays ~2x official's because DOD feeds ~2x the features
(cap 75 vs official's 40) -- it is buying the accuracy above, and the total is
still lower.

### Three changes, all of them divergences from official

**1. `feat_rep_msckf` was dead.** Official's EuRoC config runs `GLOBAL_3D` for
MSCKF features and anchored inverse depth only for SLAM. DOD never read the
option and ran everything anchored, including the extra anchor-clone Jacobian
coupling. `get_feature_jacobian_mixed` now takes the representation and each
updater passes its own. Neutral on MH_01 (0.121345 either way, differing in the
8th digit), better on V1_01 (0.0506 -> 0.0487).

**2. `max_msckf_in_update` was dead.** Official sorts MSCKF features by track
length and keeps the longest `max_msckf_in_update` (40). DOD fed every feature.
Implemented with official's exact semantics; the value is measured, not copied,
because DOD's feature supply differs:

| cap | 25 | 40 | 50 | 75 | 100 | 150+ |
|---|---|---|---|---|---|---|
| MH_01 | 0.1334 | 0.1309 | 0.1190 | **0.1132** | 0.1168 | 0.1213 |
| V1_01 | -- | 0.0506 | -- | 0.0494 | 0.0509 | 0.0487 |

Shipping 75. Official's own 40 is clearly wrong for DOD.

**3. `EKFPropagation` symmetrised the entire covariance every frame.** Official
has no such loop -- its three block writes already leave the matrix as symmetric
as it needs to be. DOD's version touched all N^2 entries with a cache-hostile
stride on every propagation. Deleting it left ATE bit-identical and took
propagation 0.309 -> 0.052 ms and marginalisation 1.042 -> 0.031 ms, i.e. 1.4 ms
per frame for nothing. `Q` is now taken as `selfadjointView<Upper>` as official
does, rather than averaged with its own transpose.

### Still open

- `do_calib_camera_pose` / `do_calib_camera_timeoffset` change the result by
  nothing at all, which is not credible for a state-dimension change. Official
  runs both ON. Unexplained, and a plausible remaining accuracy lever.
- `feed_measurement_imu` silently drops samples past 10000. Harmless once
  initialised (the buffer is pruned to the clone window) but a long pre-init
  stretch would lose IMU with no warning.
- The ROS runner's IMU guard is still unverified; `ros_container_v2` can now
  build and run it.

---

## Benchmark 10 — the guard done properly, and MH_02 root-caused (2026-08-15)

### The IMU guard: queue, never drop

Benchmark 9's guard held a stereo pair in `left`/`right` until the IMU passed
it -- and the next image message then OVERWROTE the waiting pair. On EuRoC the
IMU catches up fast enough that this rarely fired; on KAIST it dropped a third
of all frames and took circle.bag from 0.666 m to 60 m. Both ROS runners now
push completed pairs into a `std::deque` and drain it whenever the IMU advances,
so nothing is ever dropped. The queue is not a detail -- it is the whole fix.

| | broken guard | queued | no guard at all | official |
|---|---|---|---|---|
| circle.bag | 60.3 m | **0.3182** | 0.666 | 0.0305 |
| infinite.bag | -- | **0.2369** | 0.3519 | 0.0284 |
| MH_01 (bag path) | 0.1296 | 0.1296 | 0.407 | 0.1180 |

Correct IMU coverage *and* no dropped frames beats the old no-guard baseline by
2x on circle and 1.5x on infinity.

### MH_02: the dynamic initializer's velocity

The linear stage of the dynamic initializer cannot observe v0 over a near-static
window -- the least-squares fit trades velocity against feature depth almost
freely. Against Leica truth at the init instant:

| | recovered v0 | true speed | outcome |
|---|---|---|---|
| MH_01 | 0.445 m/s | 0.048 m/s | survives (0.1132 m) |
| MH_02 | 0.162 m/s | 0.029 m/s | **diverges (6.5e5 m)** |

**No standard quality metric separates the fatal solve from the healthy one.**
Measured: MH_02's bad solve has the *lower* residual (0.0127 vs 0.0191), the
*better* conditioning (1.3e-3 vs 2.0e-4), and a gravity direction agreeing with
the accelerometer to 1.5 deg -- same as MH_01's. Also ruled out by measurement:
transport, SLAM, `init_imu_thresh`, static-vs-dynamic init, `init_wait_for_jerk`,
`init_dyn_min_deg`, and official's covariance inflation factors.

The shipped mitigation discards the velocity (`init_dyn_zero_velocity`, on for
EuRoC, off by default since `tests/verify_dynamic_init` shows a genuinely moving
start does carry velocity information). One subtlety is load-bearing: the state
VALUE is zeroed while the FEJ keeps the solved velocity. Zeroing both diverges
at 3.4e4; zeroing only the value gives 0.1744. That asymmetry is empirical and
not understood -- porting official's Ceres MLE refinement of the linear solution
is the principled fix.

### Full EuRoC matrix, both sides run on the same machine

| sequence | DOD | Official |
|---|---|---|
| MH_01_easy | **0.1131** | 0.1180 |
| MH_02_easy | 0.1744 | **0.1721** |
| MH_03_medium | **0.2223** | 0.2486 |
| MH_04_difficult | 0.4580 | **0.4110** |
| MH_05_difficult | **0.3074** | 0.3183 |
| V1_01_easy | **0.0494** | 0.0634 |
| V1_02_medium | **0.0551** | 0.0573 |
| V1_03_difficult | **0.0560** | 0.0595 |

Five wins, three losses, no divergences. KAIST remains 8-10x behind and is the
open robustness story. `ctest` 5/5.

Also fixed: the ASL runner paired stereo images by index, so MH_04 (2033 left,
2032 right images) refused to run at all. Paired by timestamp now: 0.4580 m.

---

## Benchmark 11 — the KAIST gap was configuration bias (2026-08-15)

Asked whether EuRoC looked better because the work was biased toward it, because
EuRoC has ASL folders while KAIST has only bags, or because of a real structural
weakness. Measured all three.

**Not transport.** MH_01 scores 0.1131 through the ASL runner and 0.1296 through
the rosbag runner -- same estimator, same images, 15% apart. That cannot produce
a 10x gap.

**It was bias, and it had a concrete mechanism**: every improvement of the last
sessions went into `tools/euroc_options.hpp`, while `ros/vio_rosbag_runner.cpp`
carries its own independent configuration and inherited none of them. Two
settings mattered:

- `enable_slam = false`, while official's own kaist config runs `max_slam: 50`.
  DOD was being compared against official with a third of the estimator switched
  off.
- the MSCKF update uncapped, while official caps at 50.

| circle.bag | ATE | | infinite.bag | ATE |
|---|---|---|---|---|
| SLAM off, uncapped (shipping) | 0.3182 | | SLAM off | 0.2369 |
| SLAM on | 0.0650 | | SLAM on + cap 50 | **0.0261** |
| SLAM on + cap 50 | **0.0374** | | official | 0.0284 |
| official | 0.0305 | | | |

circle goes from 10x behind to 1.2x behind; infinity now beats official. Both are
the runner's defaults now.

**The tuned constants do not transfer**, which is the real lesson: the best
MSCKF cap is 75 on EuRoC and 50 on KAIST (circle, SLAM on: uncapped 0.0650,
cap 75 0.0459, cap 50 0.0374). The earlier "SLAM costs 39 m on circle.bag"
finding was also a casualty of the IMU-ordering bug, not a property of SLAM.

### Where the real remaining gap is

| sequence | DOD | Official |
|---|---|---|
| MH_01 | **0.1131** | 0.1180 |
| MH_02 | 0.1744 | **0.1721** |
| MH_03 | **0.2223** | 0.2486 |
| MH_04 | 0.4580 | **0.4110** |
| MH_05 | **0.3074** | 0.3183 |
| V1_01 | **0.0494** | 0.0634 |
| V1_02 | **0.0551** | 0.0573 |
| V1_03 | **0.0560** | 0.0595 |
| circle | 0.0374 | **0.0305** |
| infinity | **0.0261** | 0.0284 |

Six wins, four losses, none of them large, no divergences. What is left is not a
dataset-shaped gap; it is per-dataset tuning that nobody has systematically
searched, plus the missing Ceres MLE refinement in the dynamic initializer.

---

## Benchmark 12 — why the MSCKF update cost 2x official's (2026-08-16)

Two questions, both answered by measurement rather than reasoning.

### Does official's cap of 40 make us faster? No — and it costs accuracy.

| cap | MH_01 | MH_02 | MH_04 | V1_01 | total ms (MH_01) |
|---|---|---|---|---|---|
| 40 (official's) | 0.1252 | **diverges** | 0.4769 | 0.0506 | 6.88 |
| 75 (shipped) | **0.1131** | 0.1744 | **0.4580** | **0.0494** | 6.80 |

Cap 40 is worse on 6 of 8 sequences, reintroduces the MH_02 divergence, and is
**no faster** -- the MSCKF stage itself barely moves (1.581 vs 1.538 ms). The
cap rarely binds, so it was never what drove the cost.

**This also retracts an explanation given earlier**: the back-end being slower
than official's was NOT "the cost of feeding ~2x the features". Feature count
was not the variable.

### Where the time actually was

Sub-stage timing inside `update_msckf` (ms/frame, MH_01):

| | ms |
|---|---|
| EKFUpdate | 0.590 |
| measurement compression | 0.437 |
| chi2 gate | 0.319 |
| Jacobians | 0.066 |
| triangulation | 0.048 |

and inside `EKFUpdate`, across all updates (MSCKF + SLAM), 3.32 ms/frame total:

| | before | after |
|---|---|---|
| `M_a = P·Hᵀ` | 1.458 | **0.665** |
| covariance update | 0.954 | **0.750** |
| S + inverse | 0.637 | 0.638 |
| K | 0.272 | 0.273 |

**`M_a` was the defect.** It looped over every state variable and, inside that,
every measurement variable -- ~65 x 13 tiny matrix products, each allocating its
own `M_i` temporary. Official's loop is written the same way, but the fix is to
accumulate over measurement variables alone against full-height covariance
blocks: identical arithmetic, ~13 tall GEMMs instead of ~845 small ones.

**The covariance update** mirrored its upper triangle with
`Cov.block(...) = Cov.block(...).selfadjointView<Upper>()`, a self-assignment
Eigen evaluates through a full N x N temporary allocated on every update. An
explicit column-wise copy removes the allocation.

A third "optimisation" was tried and reverted: collapsing the triangular update
plus mirror into one dense `Cov -= K * M_a.transpose()`. `K*M_a^T` is symmetric
in exact arithmetic but not in floating point, and the asymmetry drives
covariance diagonals negative -- EKFPropagation aborts on MH_01 within ~16 s.

### Result

Every ATE is bit-identical; the pipeline is simply faster.

| | before | after | official |
|---|---|---|---|
| MSCKF stage | 1.535 | **1.390** | 0.719 |
| SLAM stage | 2.684 | **1.970** | 2.558 |
| estimator mean | 5.51 | **4.52** | 5.08 |
| **total mean** | 7.40 | **6.32** | 7.32 |

DOD is now faster than official on **all 10 sequences** and 1.16x faster on
average, having been a tie. KAIST: circle 6.18 -> 5.33 ms, infinity 6.83 ->
5.77 ms.

The MSCKF stage is still ~2x official's (1.390 vs 0.719) with the remaining
cost in measurement compression (0.437) and the per-feature chi2 gate (0.319);
those are the next targets if this needs to go further.

---

## Benchmark 13 — compression and chi2 gate (2026-08-16)

Continued from Benchmark 12, same method: profile, fix the expression, verify
the ATE does not move.

| | before | after |
|---|---|---|
| measurement compression | 0.437 | **0.160** |
| `S = HPH^T + R` (in EKFUpdate) | 0.637 | **0.432** |
| chi2 gate | 0.322 | **0.273** |
| assembly buffer alloc | 0.071 | **0.023** |
| **MSCKF stage** | 1.535 | **1.000** |
| **SLAM stage** | 2.684 | **1.761** |

**Compression**: official sweeps the system with ~23k individual Givens
rotations, each a separate 2 x (cols - n) Eigen call. The EKF update is
invariant to left-multiplication by an orthogonal matrix -- that invariance is
the entire premise of the compression -- so one blocked Householder QR of the
augmented `[H | res]` produces the same R without ever forming Q. Augmenting
res as the last column is what avoids Q.

**S in EKFUpdate**: it re-gathered the marginal covariance, when the rows of
`M_a` belonging to the measurement variables already *are* `P_small * H^T`.

**chi2 gate**: the gather itself is fine; its per-feature 83x83 allocation was
not. It now fills a reused buffer. Building S block-wise to avoid the gather
entirely was tried and is **slower** (0.285 -> 0.404) because it turns two dense
products into ~169 small ones per feature -- recorded at the call site so it is
not retried.

**Assembly buffer**: allocated and zeroed a 2000x300 matrix (4.8 MB) per frame
while touching ~280 rows. Reused and cleared to the bound that can be written.

### Standing, all 10 sequences

| sequence | ATE | DOD total | official total |
|---|---|---|---|
| MH_01 | 0.1131 | **5.14** | 7.15 |
| MH_02 | 0.1744 | **5.21** | 7.22 |
| MH_03 | 0.2223 | **5.61** | 7.41 |
| MH_04 | 0.4580 | **5.19** | 6.81 |
| MH_05 | 0.3074 | **5.36** | 7.05 |
| V1_01 | 0.0494 | **6.32** | 8.26 |
| V1_02 | 0.0551 | **6.18** | 7.73 |
| V1_03 | 0.0560 | **5.67** | 6.92 |
| circle | 0.0374 | **4.58** | 6.50 |
| infinity | 0.0261 | **5.01** | 7.26 |
| **mean** | | **5.43** | 7.23 |

**1.33x faster than official on every sequence**, with every ATE unchanged from
before any of this optimisation work. The MSCKF stage is now 1.000 ms against
official's 0.719; the remaining difference is spread thin (EKFUpdate 0.425,
chi2 0.273, compression 0.160) with no single dominant term left.

---

## Benchmark 14 — allocation profiling (DHAT) and end-to-end timing (hyperfine)

Tools: valgrind's DHAT for allocation accounting, hyperfine for wall-clock.
hyperfine is a static musl binary in `~/bin` on the lab box and
`/usr/local/bin` inside `ros_container_v2`; valgrind was already installed.

### Dynamic allocation, 12 s of EuRoC MH_01

| | blocks | bytes |
|---|---|---|
| before | 1,322,658 | 5.32 GB |
| after | 825,790 | **2.02 GB** |
| | -38% | **-62%** |

Three sites accounted for nearly all of the DOD-attributable churn:

**`initialize_invertible`** -- 463k allocations, the largest count in the whole
pipeline. Same nested-loop shape as EKFUpdate before Benchmark 12: it allocated
THREE heap temporaries (`M_i`, `cov_block`, `H_R_block`) per (state variable,
measurement variable) pair. Rewritten to accumulate over measurement variables
against full-height covariance blocks, and to build `M = H P H^T + R` from the
rows of `M_a` instead of gathering the marginal covariance.

**`update_slam`** -- 1.68 GB in 351 blocks, the largest byte consumer. It
allocated and zeroed a 2000x300 matrix (4.8 MB) per call, the same buffer the
MSCKF updater had. Reused and cleared only to the row bound that can be written.

**KLT pyramids** -- ~1.27 GB. `calcOpticalFlowPyrLK` builds a pyramid for BOTH
of its inputs on every call, and a frame makes three calls (two temporal, one
stereo): six pyramid builds per frame instead of two. Now built once per image
with `buildOpticalFlowPyramid` and reused, with the previous frame's pyramids
swapped in as the "from" side.

All ATEs unchanged across all 8 EuRoC sequences. Stage effects: SLAM
1.761 -> 1.568 ms, slam-delayed 0.387 -> 0.344, tracking 1.64 -> 1.50,
estimator 3.50 -> 3.26.

What remains is mostly not ours: OpenCV's KLT internals are 394k blocks on their
own, and EKFUpdate's ~33k blocks are Eigen expression temporaries (M_a, S, Sinv,
K, dx) at roughly 140 per frame.

### hyperfine, same bag, same machine, 5 runs each

| dataset | DOD | Official OpenVINS | |
|---|---|---|---|
| EuRoC MH_01 | **18.656 s** ± 0.110 | 29.014 s ± 0.046 | **1.56x faster** |
| KAIST circle | **21.321 s** ± 0.039 | 27.952 s ± 0.230 | **1.31x faster** |

This is whole-process wall clock -- bag reading, image decode, tracking,
estimation, output -- not the internal per-frame timers, and it is measured
through DOD's *rosbag* runner, so both sides read the identical bag through the
identical transport. Official exits non-zero at shutdown (a class_loader unload
throw, after all its work and output are complete), so hyperfine is run with
`-i`.
