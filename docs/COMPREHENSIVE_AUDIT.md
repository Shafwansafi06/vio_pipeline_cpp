# VIO Pipeline Three-Way Audit

## Scope and baselines

- Remote host: `moonlab (lab x86 box)`. All three implementations live
  **inside the running Docker container `ros_container_v2`**, not on the host
  filesystem directly — `/workspace/...` paths below are container paths
  (`docker exec ros_container_v2 ...`).
- Python baseline: `/workspace/vio_pipeline_v2` (plain rospy node, run with
  `PYTHONPATH=/workspace`, not a catkin package).
- DOD C++ candidate: `/workspace/vio_pipeline_cpp`, mirrored from this
  workspace. ROS1 runner built via
  `cmake .. -DVIO_BUILD_ROS1=ON -DVIO_BUILD_OPENCV_FRONTEND=ON` in
  `build-ros1/`.
- Official baseline: `rpng/open_vins` commit
  `69488123ed9362dd44b6f28e7f4680abbff1442b`, symlinked into
  `/workspace/openvins_catkin_ws/src/open_vins` and built with `catkin_make`.
- Datasets: `/workspace/KAIST VIO dataset/circle/circle.bag` (191 s) and
  `infinity/infinite.bag`; `/pose_transformed` is the common ground-truth
  source.

This document records evidence before estimator changes. A passing item means
the stated scope was actually exercised; it does not imply end-to-end parity.

> **2026-07-21 correction**: an earlier session claimed a completed DOD
> `circle.bag` run with result files "remaining on remote disk." On resuming,
> no such result files existed anywhere in the container and
> `vio_rosbag_runner` had never been built (`VIO_BUILD_ROS1` was off in the
> last build). The run below is the first one that actually produced
> verifiable output. See `portdocs/Benchmark.md` Benchmark 4 for the full
> three-way numbers and `docs/results/` for the raw CSVs/JSON this table is
> built from.

## Current production readiness

| Area | Python v2 | DOD C++ | Official OpenVINS |
|---|---|---|---|
| ROS1 image/IMU ingestion | Present; smoke-tested | Present (`VIO_BUILD_ROS1=ON`); full `circle.bag` run 2026-07-21 | Present; smoke-tested |
| Stereo KLT frontend | Present | Present (`core/tracker.cpp`, gated behind `VIO_BUILD_OPENCV_FRONTEND`); processed 4,758 stereo frames on `circle.bag` | Present |
| Static initialization | Present; initialized at bag-relative 1.022 s in smoke test | Present; initialized on `circle.bag` | Present; did not initialize in first five bag seconds |
| Dynamic initialization | Configurable but unimplemented (dynamic.py is empty) | Missing | Present |
| IMU propagation | Present | Present | Present |
| Clone augmentation/marginalization | Present | Present | Present |
| MSCKF feature update | Present | Partial; camera-0 assumption | Present |
| SLAM delayed initialization/update | Present | Empty functions | Present |
| ZUPT | Present | Partial | Present |
| FIFO/LIFO hover logic | Present (per user, does not detect real hovers) | Implemented from Kottas/Wu/Roumeliotis paper (`msckf/hover_detector.cpp`); disabled, over-triggers ~100% of frames on real data, root cause open | Not an upstream feature |
| Active retriangulation | Present | Missing | Present via normal frontend/update flow |
| ROS odometry/path/tracking output | Present | Missing (CSV-only; no odometry/path topics published) | Present |
| Direct rosbag execution | Present | Present via `ros/vio_rosbag_runner.cpp` (reads `rosbag::View` directly, not a live ROS node) | Present |

### `circle.bag` three-way run (2026-07-21)

The camera-0-only MSCKF gap and missing SLAM/hover logic (items above) show up
directly in accuracy: full-trajectory ATE RMSE is 0.685 m (DOD) vs 0.067 m
(official OpenVINS) vs 0.656 m (Python v2) — DOD and Python land within 5% of
each other despite different gaps, both roughly 10× worse than official
OpenVINS. DOD is faster per frame than Python (11.19 ms vs 99.9 ms, no
real-time frame drops) and uses less memory than Python (390 MB vs 941 MB
peak RSS) but is not yet as fast or as memory-light as official OpenVINS
(8.34 ms, 75 MB). Full table and interpretation: `portdocs/Benchmark.md`
Benchmark 4.

## Confirmed C++ implementation gaps

1. `VioManager::feed_measurement_camera_tracks` accepts already-associated,
   normalized tracks. There is no image API, KLT tracker, stereo matcher,
   histogram preprocessing, mask support, track-rate control, or feature-ID
   lifecycle equivalent to Python v2.
2. `UpdaterSLAM::delayed_init` and `UpdaterSLAM::update` have empty bodies.
3. The manager does not construct or invoke a SLAM updater.
4. FIFO/LIFO switching, backward feature chains, rotation-prior gating,
   descriptor/reprojection costs, Hungarian assignment, temporal smoothing,
   hover covariance-only updates, and active-track retriangulation are absent.
5. The manager now receives FEJ, integration method, and clone capacity through
   StateOptions; YAML parsing and the rest of the Python configuration surface
   are still missing.
6. The MSCKF update calls the feature Jacobian with camera 0, so stereo and
   mixed-camera tracks are not equivalent to either baseline.
7. ~~No ROS1/ROS2 node or bag adapter is built by CMake.~~ **Resolved
   2026-07-21**: `ros/vio_rosbag_runner.cpp` builds behind the
   `VIO_BUILD_ROS1`/`VIO_BUILD_OPENCV_FRONTEND` CMake options and runs a full
   bag end-to-end; it writes CSVs directly rather than publishing ROS
   odometry/path topics (no rospy-equivalent live visualization yet).
8. Estimator orchestration, initialization, propagation, state-helper, and updater
   OOP shells have been converted to POD contexts and free functions. Strict DOD
   is still not complete: dynamic `Eigen::MatrixXd`/`VectorXd`,
   `conservativeResize`, strings, and heap-backed decompositions remain in the
   update/type paths; the arena is not yet the exclusive temporary allocator.
9. The current `compare_openvins` target is a synthetic mock with random
   Jacobians; it is not evidence about official OpenVINS.
10. The synthetic manager benchmark bypasses image tracking and most visual
    updates. Its latest roughly 0.343--0.348 ms per 0.60 s synthetic window is
    useful for regression detection only, not an end-to-end throughput claim.

## Verification evidence collected

- Remote Python v2 five-second circle smoke test:
  initialized, published odometry/path/feature image, executed ZUPT, hover
  detection and FIFO processing, and reached MSCKF logic without crashing.
- Remote official OpenVINS build:
  `ov_core`, `ov_init`, `ov_msckf`, and lightweight `ov_data` metadata package
  built successfully with ROS Noetic.
- Remote official five-second circle smoke test:
  node and visualization topics were live; initialization was repeatedly
  rejected because measured acceleration excitation stayed below the configured
  0.60 threshold. The cause is now traced: pinned upstream KAIST configuration
  has `try_zupt: false`, while Python v2 changed it to `true`; both managers set
  `wait_for_jerk` according to whether a ZUPT updater exists. This is an
  as-shipped configuration difference, not an initializer-equation difference.
- Controlled official run using the exact Python v2 config:
  initialized successfully and produced the same printed orientation, gyro
  bias, zero velocity, and position to displayed precision. Acceleration bias
  differed slightly (`[0.0002, -0.0000, -0.0072]` official versus
  `[0.0003, -0.0000, -0.0077]` Python), so deterministic message-window capture
  is still required before judging numerical equivalence.
- Remote C++ warnings + ASan/UBSan build succeeded. The only registered test
  passed and reported matching Python nominal initialization and propagated
  state values.
- The latest Release build passes both registered tests: verify_math and
  verify_noalloc_propagation. The latter disables Eigen heap allocation around
  a complete propagation/covariance step and verifies finite, symmetric
  covariance output.
- With packed active covariance scratch storage, the steady synthetic
  propagation kernel measured about 0.398 ms per 1.0 s IMU window on the
  remote host, versus the earlier approximately 0.433 ms baseline (about 8%
  faster). This is a propagation-only result, not an end-to-end VIO claim.
- Stateless helper classes plus the Propagator, updater, and manager classes
  were replaced by POD contexts and free functions. The remote Release build and
  both registered tests remained green after each conversion.
- That verifier does **not** yet cover covariance blocks, transition/noise
  Jacobians, FEJ observability, triangulation, nullspace projection, gating,
  EKF update, ZUPT, SLAM, stereo frontend, hover logic, or trajectory output.
- Trusted SSH X11 forwarding is working. Remote RViz is displayed locally and
  receives relayed Python path/tracking topics at roughly 3 Hz with bag playback
  intentionally slowed to 0.2x.
- **2026-07-22 continued further still**: audited online camera-intrinsics
  calibration (`do_calib_camera_intrinsics=true`) after the "better tracking
  precision makes things worse" paradox. Found and fixed two real bugs: the
  EKF's estimated intrinsics were never resynced into (a) the updater's own
  camera-model buffer or (b) the tracker's separate camera-model copy used
  for computing `uv_norm` — both silently used stale, init-time intrinsics
  regardless of what the filter estimated. Both fixed (matches official's
  `StateHelper::EKFUpdate` resync behavior) — but produced no measurable ATE
  change. Control test (disabling intrinsics calibration entirely) also
  produced no meaningful change, ruling out online intrinsics calibration
  itself as a major factor. Covariance-conditioning check: negative
  eigenvalues present but at floating-point-roundoff magnitude near the
  system's genuine unobservable directions, not a real PSD violation; one
  unexplained anomaly (max eigenvalue frozen for 80+ seconds) flagged but
  not resolved. `circle.bag` ATE remains ~0.685m vs official's 0.067m after
  this entire session's investigation. Full writeup: `portdocs/Benchmark.md`.
- **2026-07-22 continued further (persistent /goal session)**: static-init
  covariance blocks fixed (position/velocity per-block overrides matching
  official) and stereo timestamp switched to cam0-only (both negligible
  effect, kept). Direct numeric comparison of DOD's vs official's actual
  computed initial gyro bias on the identical bag found a real difference
  (DOD ~0 vs official's meaningfully nonzero) — but a decisive injection
  test (hardcoding official's exact init values into DOD) left ATE
  unchanged, ruling this out as the cause. Error-over-time analysis showed
  gradual accumulation, not a single bad frame. **Major finding**: only
  ~4-5% of MSCKF feature-update attempts succeed; ~52% of triangulation
  rejections are condition-number/insufficient-parallax, ~30% near-distance
  — MSCKF is severely constraint-starved on this real dataset, the first
  concrete mechanistic explanation found this session. Tried fixing via
  tracking precision (real FAST detector + sub-pixel refinement, then
  sub-pixel alone isolated) — both made things WORSE (0.87m, 1.18m ATE),
  ruling out detector choice and implicating something downstream that
  doesn't tolerate better measurement precision — leading suspect: online
  camera-intrinsics calibration, never audited. Full writeup:
  `portdocs/Benchmark.md`. `circle.bag` ATE remains ~0.68m vs official's
  0.067m; full parity not reached despite exhaustive checking.
- **2026-07-22 continued, reference switched to official C++ source**: Python
  dropped as a correctness reference (never itself validated against
  official). Re-audited tracker, MSCKF update math, and feature-init config
  directly against `/workspace/open_vins_official` (commit `6948812`). Fixed:
  tracker mono-track survival (official keeps a track alive if either camera
  succeeds; port required both) + RANSAC threshold (`2/max(fx,fy)` not
  `1/fx`) — kept, but measured effect on `circle.bag` was a wash (0.675→0.69m
  ATE). Fixed a real wiring bug: `update_msckf()` ignored
  `VioManagerOptions::feat_init_opt` entirely, using a disconnected local
  default. Tried official's actual KAIST-effective threshold VALUES
  (`max_cond_number=25000`, `min_dist=0.10`, etc.) — caused a catastrophic
  169m-ATE divergence, reverted to the port's own safer defaults (kept the
  wiring fix). MSCKF Jacobians/chi2/nullspace-projection/compression verified
  mathematically equivalent to official (different method, same result).
  Core ~10x accuracy gap to official OpenVINS remains unresolved after
  checking propagation, update math, and config against the real official
  source — see `portdocs/Benchmark.md` for full writeup.
- **2026-07-22 deep debugging session**: user correctly flagged that the
  DOD trajectory *shape* (not just ATE magnitude) was visibly wrong —
  raw estimate traced a jagged, non-repeating open path instead of the
  circle. Root-caused and fixed two real bugs: (1) `core/tracker.cpp` was
  missing the RANSAC epipolar-consistency check present in the Python
  reference (`core/track_KLT.py`), letting KLT tracks that converged onto
  the wrong structure feed straight into every update; (2) the DOD runner
  unconditionally ran ZUPT every frame, while official OpenVINS's own
  `kaist_vio` config ships `try_zupt: false` — confirmed ZUPT was firing on
  68–84% of `circle.bag` frames during real (slow) flight. Both fixed and
  kept (`VioManagerOptions::enable_zupt`, `core::ransac_inliers`). Also
  confirmed the RANSAC fix resolves the SLAM updater's earlier instability
  (45,000,000 m divergence) — bad feature tracks were the root cause there
  too — so `enable_slam` is now on by default. Implemented the paper-exact
  hovering detector (Kottas/Wu/Roumeliotis) but it over-triggers (~100% of
  frames classified as hovering on a real slow circular flight); shipped
  disabled. Full writeup: `portdocs/Benchmark.md`. Remaining gap to
  official OpenVINS accuracy is not fully explained; propagation math and
  SLAM landmark count were ruled out as causes.
- **Real `circle.bag` three-way run, 2026-07-21**: DOD `vio_rosbag_runner`
  (4,758 stereo frames, initialized, 11.19 ms/frame mean, 390 MB peak RSS),
  official `run_subscribe_msckf` (2,620 updates, 8.34 ms/frame mean, 75 MB
  peak RSS), Python v2 `main.py` (1,629 processed frames under real-time
  bag pressure, 99.9 ms/frame mean, 941 MB peak RSS). All three timestamp-
  associated and Umeyama-aligned to the same `/pose_transformed` recording;
  full and hover-segment ATE in `portdocs/Benchmark.md` Benchmark 4. Raw
  CSVs/JSON in `docs/results/`, trajectory overlay in
  `docs/circle_trajectory_comparison.png`.

## Retry-storm bug (2026-07-22) — root cause found via lockstep instrumentation, not a net ATE win

Discovery method: built a second, isolated copy of official's source
(`/workspace/open_vins_instrumented`, own catkin workspace, never touches the
untouched `open_vins_official` baseline used for the reference numbers),
instrumented `UpdaterMSCKF.cpp` with per-call accept-rate counters, and
compared directly against equivalent counters added to DOD's `update_msckf`.

Finding: official's real accept rate is ~68-73% on ~5 features/call; DOD's was
~4-5% on ~150 features/call. Cause: `msckf/vio_manager.cpp` was un-marking
`to_delete` on features after a failed MSCKF update, so failed features were
retried every subsequent frame against the same stale measurement history
forever, instead of being discarded once like official does.

Fix: removed the un-delete override in `msckf/vio_manager.cpp` so DOD now
matches official's `update_msckf(...)` + `cleanup()` one-shot-per-feature
behavior exactly.

Measured effect: path length and trajectory shape improved substantially on
both bags (much less scale-collapse; circle.bag path 23.6 m and infinity.bag
path 21.1 m, both much closer to their ~29-30 m ground truth than before), but
aggregate ATE RMSE got *worse*: circle.bag 0.685 m → 0.785 m, infinity.bag
0.727 m → 1.070 m. Official remains unchanged at 0.067-0.070 m. Full details,
numbers, and the reproducible SLAM+retry-storm-fix crash (`EKFPropagation`
diagonal at state 178, now worked around by disabling SLAM) are in
`portdocs/Benchmark.md`'s "Retry-storm bug found and fixed" section.

**Status: full numerical parity with official OpenVINS was not achieved.**
This is the most concrete, rigorously-verified structural bug found in the
entire investigation (direct lockstep comparison against real official C++
source), but it did not close the ATE gap. Next unpursued lead: re-measure
DOD's true single-shot (no-retry) triangulation/chi2 accept rate with fresh
debug counters, now that the retry-storm's masking effect on that rate is
gone.

## Required mathematical gates before completion

1. Quaternion convention, multiplication order, reset Jacobian, and rotation
   direction (`G->I`, `I->C`) must be identical at every boundary.
2. State ordering/IDs and all covariance block operations must be checked after
   augmentation, update, and marginalization.
3. Propagation mean, `F`, `G`, discrete `Phi`, and `Qd` must be compared for
   discrete, RK4/analytic covariance modes and calibrated/uncalibrated IMU
   models.
4. Camera radtan/equidistant projection and both Jacobians must be finite-
   differenced over nominal, edge, and near-singular points.
5. Feature triangulation and Gauss-Newton refinement must share frame
   conventions, cheirality, baseline, depth, conditioning, and convergence
   rejection rules.
6. MSCKF residual/Jacobians, FEJ evaluation points, nullspace projection,
   compression, chi-square gating, and covariance update must match on identical
   tracks.
7. SLAM delayed initialization, anchored inverse-depth transforms, reanchoring,
   update batching, and marginalization must be implemented and compared.
8. ZUPT IMU windows, disparity windows, jerk policy, gating, and covariance
   propagation must be compared, including the observed initialization
   divergence.
9. FIFO/LIFO hover mode must be tested for entry/exit hysteresis, assignment
   failures, low-feature frames, timestamp tolerance, state rollback, and
   covariance positive-semidefiniteness.
10. Long-run invariants: finite normalized quaternion, symmetric PSD covariance,
    monotonic timestamps, bounded queues, valid clone/landmark IDs, and no
    out-of-bounds/heap growth.

## End-to-end comparison gates

Both bags must be run with identical calibration, topic mapping, start/end
times, playback rate, CPU affinity/thread settings, and logging. Required
outputs are initialization time/success, ATE/RPE (translation and rotation),
scale/yaw drift, endpoint drift, lost/rejected track counts, update acceptance,
NaN/reset/dropout events, CPU time, wall time, peak RSS, per-frame latency and
real-time factor. Results are invalid unless each trajectory is timestamp-
associated and aligned to `/pose_transformed` with a declared alignment model.
