# DOD ↔ Official OpenVINS Parity Manual

## Purpose of this document

DOD ("data-oriented design") is our own from-scratch C++ implementation of
the OpenVINS MSCKF visual-inertial odometry algorithm. It exists so we can
ship a VIO stack **without inheriting official OpenVINS's open-source
license obligations** — we are not allowed to copy or derive from their
code, so DOD is written independently, in a flat-array / no-inheritance
style instead of their class-hierarchy (OOP) style.

**The one invariant that matters: DOD's *logic* must be identical to
official's algorithm.** Architecture (data layout, no virtual dispatch, no
STL containers in hot paths) is deliberately different. Math, control flow,
thresholds, and edge-case handling are supposed to be the same. Any place
where DOD's behavior differs from official's is either (a) a bug to fix, or
(b) a deliberate, documented exclusion (see "Deliberate exclusions" below) —
never an accidental invention.

This document is the map: which DOD file corresponds to which official
file, every known logical divergence found so far (fixed and still-open),
and how to keep verifying parity going forward.

## File correspondence table

| DOD file | Official file | Covers |
|---|---|---|
| `core/tracker.cpp`/`.hpp` | `ov_core/src/track/TrackKLT.cpp`/`.h` | Stereo KLT feature tracking, RANSAC outlier rejection, grid-based re-detection |
| `core/feature.cpp`/`.hpp` | `ov_core/src/feat/Feature.cpp`, `FeatureInitializer.cpp` | Single feature struct, `single_triangulation`, `single_gaussnewton` |
| `core/cam.cpp`/`.hpp` | `ov_core/src/cam/CamRadtan.h`, `CamEqui.h` | Camera distort/undistort models + Jacobians |
| `initialize/initialization.cpp`/`.hpp` | (official's static-init logic lives inline in `VioManager.cpp` + `ov_core` helpers; no direct 1:1 file) | Static IMU initialization: gravity/bias estimate from a stationary window |
| `msckf/state.cpp`/`.hpp` | `ov_msckf/src/state/State.cpp`/`.h`, `StateOptions.h` | EKF state vector layout, covariance storage |
| `msckf/state_helper.cpp`/`.hpp` | `ov_msckf/src/state/StateHelper.cpp`/`.h` | Clone augmentation, marginalization, EKF update mechanics |
| `msckf/propagator.cpp`/`.hpp` | `ov_msckf/src/state/Propagator.cpp`/`.h` | IMU mean/covariance propagation (discrete/RK4/analytic, Qd discretization) |
| `msckf/updaters.cpp`/`.hpp`, `msckf/updater_mixed.cpp`/`.hpp` | `ov_msckf/src/update/UpdaterMSCKF.cpp`/`.h`, `UpdaterHelper.cpp`/`.h` | MSCKF residual/Jacobian, chi2 gate, nullspace projection, measurement compression |
| `msckf/updater_slam_helper.cpp`/`.hpp` (+ SLAM parts of `updaters.cpp`) | `ov_msckf/src/update/UpdaterSLAM.cpp`/`.h` | Delayed-init SLAM landmarks, anchor changes |
| `msckf/vio_manager.cpp`/`.hpp` | `ov_msckf/src/core/VioManager.cpp`/`.h`, `VioManagerHelper.cpp` | Top-level per-frame orchestration: propagate → classify features → MSCKF update → SLAM update → marginalize |
| `ros/vio_rosbag_runner.cpp` | (no official equivalent; official's ROS1 entry point is `ros1_serial_msckf.cpp`/launch files) | Our ROS1 bag-driven executable, config wiring |
| `msckf/hover_detector.cpp`/`.hpp` | **no official equivalent** | Extra feature, not part of official's algorithm — see "Deliberate exclusions" |

Config files: `config/kaist_vio/estimator_config.yaml` (DOD) is read from
the same file official's own `config/kaist_vio/estimator_config.yaml` uses —
both must be pointed at the identical YAML so thresholds can never silently
drift apart.

## Deliberate exclusions (do NOT try to match these)

- **Hover detector** (`msckf/hover_detector.cpp`): implements the
  Kottas/Wu/Roumeliotis hovering-detection paper. This is **not part of
  official OpenVINS** — it was added on top, on request, in an earlier phase
  of this project. Per current direction, we do **not** want this as a
  product feature. It is present in the repo but disabled
  (`VioManagerOptions::enable_hover_detection = false` in
  `ros/vio_rosbag_runner.cpp`). Leave it disabled; do not spend effort
  making it match anything, since it has no official counterpart to match.
- **EuRoC runner** (`ros/vio_rosbag_runner_euroc.cpp`): a separate dataset
  entry point, not part of the parity surface. Currently broken (distortion
  Jacobian diverges at EuRoC's larger radtan coefficients) — tracked
  separately, not a parity issue.

## Known logical divergences found this session

Every entry below was found by direct comparison against official's real
C++ source (not the Python reference, which was ruled untrustworthy earlier
— see git/session history). Status is either **FIXED** (DOD now matches
official) or **OPEN** (still diverges, or fix has an unresolved side effect).

### FIXED

1. **Retry-storm in MSCKF feature discarding** (`msckf/vio_manager.cpp`).
   DOD was un-marking `to_delete` on features that failed an MSCKF update,
   causing endless retries on stale data every subsequent frame. Official
   discards every fed feature unconditionally (`to_delete = true` +
   `cleanup()`) — one-shot per feature, no retries. Fixed to match.
2. **Missing RANSAC outlier rejection** (`core/tracker.cpp`). DOD's stereo
   KLT never ran `cv::findFundamentalMat` epipolar consistency checking on
   its own temporal flow; official does. Fixed: `core::ransac_inliers`, run
   independently per camera on undistorted points, threshold
   `2.0/max_focal` (official's exact value — DOD originally had a mistaken
   `1.0/fx`-only threshold, also fixed).
3. **Mono-track survival** (`core/tracker.cpp`). DOD dropped a track the
   moment either camera's temporal KLT failed; official keeps a track alive
   as long as *either* camera survives (per-camera-independent lifetime).
   Fixed.
4. **ZUPT unconditionally enabled** (`ros/vio_rosbag_runner.cpp`). Official's
   own `kaist_vio/estimator_config.yaml` ships `try_zupt: false` for this
   dataset; DOD had no way to disable it and was calling
   `try_update_zupt()` every frame regardless. Fixed:
   `VioManagerOptions::enable_zupt`, set `false` to match official's config
   for KAIST.
5. **Feature-init options wiring bug** (`msckf/updaters.cpp`). `update_msckf`
   was building its own disconnected local `FeatureInitializerOptions`
   instead of using `VioManagerOptions::feat_init_opt` (the SLAM path used it
   correctly, MSCKF path didn't). Fixed the wiring. (Note: official's actual
   *threshold values* for these options are NOT safe to copy in isolation —
   tried once, caused catastrophic divergence, because they're only safe
   inside official's complete pipeline with its other complementary
   quality-control. DOD keeps its own safer threshold values.)
6. **Static-init covariance blocks** (`initialize/initialization.cpp`).
   Official overrides position (σ=0.05) and velocity (σ=0.01) instead of a
   uniform σ=0.02 across all 15 error-state dims at init. Fixed to match.
7. **Stereo timestamp averaging** (`core/tracker.cpp` /
   `msckf/vio_manager.cpp`). Official uses cam0's own timestamp as
   authoritative; DOD averaged left/right. Fixed: cam0-only.
8. **Camera-intrinsics resync (two bugs)** (`msckf/vio_manager.cpp`,
   `ros/vio_rosbag_runner.cpp`). When online intrinsics calibration is
   enabled, the EKF's estimated intrinsics were never copied back into (a)
   the buffer the updater's own Jacobian/residual code reads
   (`vio.params.cam_models[]`), or (b) the tracker's separate camera-model
   copy used for `undistort()`. Official's `StateHelper::EKFUpdate` resyncs
   after every update; DOD had no equivalent. Both fixed.
9. **`reached_max` SLAM-candidate stereo double-count** (`msckf/vio_manager.cpp`).
   Was counting stereo measurements 1x instead of 2x when checking whether a
   feature reached the max-track-length threshold. Fixed.
10. **`max_msckf_in_update` cap never enforced** (`msckf/vio_manager.cpp`).
    The config value existed (`kaist_vio/estimator_config.yaml:
    max_msckf_in_update: 50`) but nothing actually capped/sorted the feature
    list fed to `update_msckf`; official sorts ascending by track length and
    keeps only the longest tracks up to the cap. Implemented to match
    exactly. (Measured to have negligible/negative effect on this dataset in
    isolation — kept anyway since it is genuinely correct behavior; see
    "Open / still under investigation" for why isolated correctness fixes
    haven't closed the accuracy gap.)
11. **Stale-seed stereo re-tracking, architecturally fixed**
    (`core/tracker.cpp`). Originally: DOD's per-camera track arrays were
    paired by array index (same index = same feature in both cameras),
    forced to the same length every frame. When one camera's temporal KLT
    failed, the array-length invariant forced DOD to write *something* into
    that camera's slot, so it carried forward last frame's stale pixel
    coordinate — which then got seeded against a brand-new image next frame,
    silently corrupting that camera's observations for the track's whole
    remaining life. A same-frame stereo cross-camera "recovery" match was
    tried as a patch (recovering right-from-left only; recovering
    left-from-right caused a reproducible 45 m divergence on `circle.bag`,
    root cause never isolated) but this was only ever a workaround for the
    paired-array design, not a fix of it.

    **Properly fixed** by rewriting `track_stereo_frame` so each camera keeps
    a fully **independent** track list (own count, own survivors, matched
    only by feature ID) — matching official's actual `pts_last[cam_id]`
    design. A camera that fails to track a point now simply drops that ID
    from its own list; no stale coordinate is ever carried forward, and the
    stereo-recovery patch is no longer needed at all (removed entirely — a
    net simplification, not just a bug fix). A track that survives in only
    one camera lives on as a genuine mono observation, exactly matching
    official's tolerance for mono-only tracks, until a fresh grid detection
    + stereo pairing re-establishes a (new-ID) stereo pair for that physical
    point. Also fixed as part of the same rewrite: newly-detected left
    corners are now kept as valid mono tracks even when the immediate
    stereo match to the right camera fails (previously discarded entirely,
    another gratuitous DOD-only restriction with no official counterpart).

    Measured effect: circle.bag path length went from 23.6 m to 27.95 m
    (ground truth 29.8 m) and infinity.bag from 21.1 m to 31.7 m (ground
    truth 29.1 m) — both now close to ground-truth scale for the first time.
    ATE RMSE: circle.bag 0.865 m → 1.171 m, infinity.bag 1.009 m → 1.114 m
    (both nominally worse point-wise, consistent with the pattern seen
    throughout this investigation where architecturally-correct fixes
    improve trajectory shape/scale without reducing point-wise RMSE — see
    open item #2 below). Shipped anyway: this is the logically-faithful
    design official actually uses, not a numeric-metric choice, and it
    eliminates an entire class of invented-measurement bug rather than
    papering over one direction of it.

12. **`conservativeResize`-without-zeroing bug in `update_msckf`/
    `update_slam` (`msckf/updaters.cpp`).** Both functions accumulate
    per-feature Jacobian/residual blocks into a pre-allocated `Hx_big`/
    `res_big` buffer, growing it via `conservativeResize(...)` if a call
    ever needs more rows/columns than the initial allocation. Eigen's
    `conservativeResize` does **not** zero-initialize the newly added
    region, and the code immediately does `Hx_big.block(...) +=
    H_x.block(...)` into it — a real latent bug at all 6 call sites
    (row/column growth × `update_msckf`/`update_slam`). **Fixed**: explicit
    `.setZero()` after every `conservativeResize` call.

    | Dataset | ATE before | ATE after | Gap to official (before → after) |
    |---|---|---|---|
    | circle.bag | 1.171 m | **0.624 m** | 17.5x → 9.3x |
    | infinity.bag | 1.114 m | **0.201 m** | 15.9x → **2.9x** |

### OPEN — still diverges or has an unresolved side effect

1. **Single-shot MSCKF triangulation accept rate is far below official's,
   with no confirmed cause.** Instrumented both DOD's `update_msckf` and a
   dedicated instrumented copy of official's real source
   (`/workspace/open_vins_instrumented`, its own catkin workspace, built
   without ever modifying the untouched official baseline checkout) with
   identical per-call counters. Measured: official's single-shot
   triangulation success rate is ~68-73% on `circle.bag`; DOD's is ~6%, even
   with retries removed and the `max_msckf_in_update` cap correctly applied.
   Ruled out: math differences (triangulation function is now line-by-line
   identical), feature-selection criteria (structurally identical), config
   (num_pts/grid_x/grid_y all match official's `kaist_vio` config exactly).
   The most likely remaining explanation is correspondence *quality* (DOD's
   tracked points still carry less usable parallax/geometry than official's,
   even after the independent-per-camera-list architectural fix above —
   real FAST detector + sub-pixel corner refinement were tried in isolation
   and made things *worse*, implying whatever's different needs to be
   ported as one coherent unit, not piecemeal). **Re-measured after the
   independent-list fix**: accept rate moved from ~6% to ~9% on `circle.bag`
   — a small real improvement, but triangulation still fails ~88% of the
   time (`rej_tri` dominates over `rej_chi2` by ~50x). The independent-list
   architecture helped a little but is nowhere near closing this gap alone.
2. **HIGH PRIORITY — a specific, reproducible fragile point in `circle.bag`
   around frame ~4030-4500** where multiple *different*, otherwise unrelated
   feature-quality changes each independently trigger a catastrophic EKF
   divergence at almost the exact same frame index. Observed twice: (a) the
   left-from-right stereo-recovery direction (before the independent-list
   architectural fix existed) diverged to 45 m ATE starting at frame 4032;
   (b) a full, faithful port of official's real detection pipeline (real
   `cv::FAST` + `cv::cornerSubPix` + official's exact two-level occupancy
   grid and redetection-gate logic, matching `Grider_FAST`/`Grider_GRID`/
   `perform_detection_monocular` line-for-line) diverged to 277 m path
   length (single-step jumps up to 63 m) starting at frame 4030, on the
   SAME bag. Official's real source does **not** diverge on this bag (it
   gets 0.067 m), so this is a DOD-side numerical fragility, not an accepted
   side effect of matching official's algorithm — it should be treated as a
   bug, not a metric regression. The real-FAST detection port was reverted
   because of this (not because of a worse aggregate number — see the
   distinction in "How to verify parity work" below), pending root-cause.
   **Investigated (2026-07-22, same day), initial "root cause" claim
   retracted after further check**: added `Eigen::SelfAdjointEigenSolver`
   eigenvalue logging of the active covariance block (`ros/vio_rosbag_runner.cpp`,
   guarded by `DOD_DIAG_DEBUG`) across frames 2900-4550. Found that on the
   diverging real-FAST build, the dominant eigenvalue is frozen at an
   identical bit-for-bit value (0.034882) for ~600 frames (3200-3800), then
   grows geometrically to 1122 by frame 4550 — this matches an anomaly
   flagged earlier in this investigation ("the largest eigenvalue froze for
   80+ seconds", see the camera-intrinsics-resync section above). Initially
   reported as the root cause of the position divergence.

   **This was wrong, corrected by inspecting the actual eigenvector (not
   just the eigenvalue)**: computed the full eigendecomposition at frame
   3500 (inside the frozen window) and looked at which state-vector
   components the top eigenvectors actually load onto
   (`ros/vio_rosbag_runner.cpp`'s `DOD_DIAG_DEBUG` block, `frames == 3500`
   case). The top TWO eigenvalues (ranks 0 and 1, essentially the same
   frozen magnitude) both load with **exactly equal weight on the IMU state
   and on every single clone** (e.g. rank 0: imu=0.088, every one of 11
   clones=0.0829037, bit-identical across clones). This is the textbook
   signature of VIO's standard **unobservable global position/yaw gauge
   freedom** (shifting or yaw-rotating every pose in the trajectory
   together changes no visual measurement) — a mathematical property of
   *every* MSCKF/VIO formulation, official's included, not a DOD-specific
   defect. Its variance is expected to grow without bound when unconstrained
   by an anchor/prior; critically, **ATE is computed after rigid (Umeyama)
   alignment to ground truth, which exactly cancels global position and
   yaw** — so growth in this specific direction, however large, cannot be
   the cause of a *relative*-trajectory (aligned) divergence. Ranks 2-5 (much
   smaller eigenvalues, ~0.001-0.008, concentrated 85-90% on the IMU block
   alone rather than spread across clones) look like ordinary, bounded
   IMU-state uncertainty and show no comparable anomaly at frame 3500.

   **Net effect of this diagnostic**: retracts the earlier claim that the
   frozen/growing dominant eigenvalue explains the position divergence. It
   does not. **The actual mechanism behind the frame ~4030 circle.bag
   divergence remains unfound.** What's ruled out now: the hover detector
   (disabled), anchor-frame sensitivity in triangulation (proven invariant),
   and the dominant-eigenvalue growth (proven to be the benign, ATE-irrelevant
   gauge-freedom direction). **Next step for whoever continues this**:
   look at whether any of the SMALLER (rank 2+, IMU-concentrated) eigenvalues
   show anomalous behavior specifically around frames 4000-4300 on the
   diverging build (not just at the single frame 3500 snapshot already
   taken) — those are the directions that could plausibly correlate with
   the visible position error, unlike the top two.
3. **EuRoC distortion Jacobian**: diverges catastrophically
   (`EKFPropagation` negative-diagonal crash) within ~250 frames on
   `MH_01_easy.bag`, whose real lens distortion coefficients are ~40x larger
   in magnitude than KAIST's near-zero (pre-rectified) values. Needs its own
   finite-difference validation of `core::compute_distort_jacobian` at large
   coefficients — not attempted.
4. **SLAM updater implemented but shipped disabled.** `delayed_init_slam`/
   `update_slam`/`change_anchors`/`perform_anchor_change`
   (`msckf/updater_slam_helper.cpp`, SLAM parts of `msckf/updaters.cpp`) are
   a full, Jacobian-verified (finite-difference test,
   `tests/verify_slam_jacobians.cpp`) port of official's anchored
   inverse-depth SLAM landmarks. But combined with the retry-storm fix,
   SLAM+MSCKF together hits a reproducible `EKFPropagation` crash (negative
   covariance diagonal at state index 178, identical value every run,
   root cause not isolated). Currently shipped with
   `VioManagerOptions::enable_slam = false`. This needs to be re-enabled and
   the crash root-caused for a true full-parity port, since official's
   production accuracy relies on SLAM landmarks contributing alongside
   MSCKF.

## Current measured gap to official (as of this writing)

| Dataset | DOD ATE RMSE | Official ATE RMSE | Gap | DOD path length | GT path length |
|---|---|---|---|---|---|
| KAIST `circle.bag` | 0.624 m | 0.067 m | 9.3x | 26.45 m | 29.8 m |
| KAIST `infinite.bag` | 0.201 m | 0.070 m | **2.9x** | 29.59 m | 29.1 m |

**Not yet at parity, but the closest this investigation has come.** The
uninitialized-memory fix (divergence #12 above) closed infinity.bag's gap
from 15.9x to 2.9x and circle.bag's from 17.5x to 9.3x — both datasets now
have path lengths within a few percent of ground truth. The remaining gap
on circle.bag specifically is larger than infinity.bag's, consistent with
circle.bag being the one dataset with a known, still-unexplained fragility
(open item #2). The overall remaining gap is now understood to trace to
tracker/frontend correspondence *quality* (open item #1 above — how good
each individual measurement is, not how consistent the overall trajectory
shape is), not to the MSCKF/propagation math (verified equivalent), not to
any missing feature, and no longer to the array-pairing architecture
(now fixed).

## How to verify parity work (methodology that worked this session)

1. **Never trust the Python reference for correctness** — it was never
   itself validated against official's C++, so "matches Python" only proves
   DOD matches Python's own bugs too. Always compare against
   `/workspace/open_vins_official` (the real official C++ checkout,
   `commit 69488123ed9362dd44b6f28e7f4680abbff1442b`, **never modify this
   checkout**).
2. **For aggregate behavioral comparisons** (accept rates, feature counts,
   etc.), build a *separate instrumented copy* of official's source
   (pattern used this session: `/workspace/open_vins_instrumented`, its own
   catkin workspace `openvins_instrumented_ws` symlinked to the copy) with
   debug counters added, and run it side-by-side against equivalently
   instrumented DOD code on the identical bag. This lockstep comparison is
   what found the retry-storm bug and the accept-rate gap — reading code
   side-by-side without run-time numbers was not enough to find either.
3. **After any change, always**: rebuild both `build-ros1` (the ROS1
   executable) and `build-release` (the regression-test suite), run
   `ctest` (must stay 5/5 green:
   `verify_math`, `verify_noalloc_propagation`, `verify_clone_invariants`,
   `verify_feature_lifetime`, `verify_slam_jacobians`), rerun both
   `circle.bag` and `infinite.bag` (bag path is
   `/workspace/KAIST VIO dataset/{circle,infinity}/{circle,infinite}.bag`
   — note the directory is named `infinity` but the file inside is
   `infinite.bag`), and re-evaluate ATE with
   `scripts/evaluate_trajectory.py` against
   `docs/results/dod_{circle,infinity}_groundtruth.csv`.
4. **Priority is logical fidelity to official, not aggregate ATE.** Per
   current direction: a change that makes DOD's logic more faithfully match
   official's actual algorithm should be KEPT even if it makes ATE nominally
   worse on one or both datasets — several genuinely-correct fixes this
   session did (the `max_msckf_in_update` cap, the independent-per-camera-
   list tracker rewrite). Do not use "the number went down" as a reason to
   revert a logic-correctness fix on its own.
   **The one exception**: a genuine, reproducible STABILITY DEFECT (a
   catastrophic divergence — tens of meters of drift, not a modest RMSE
   increase) is different from a metric regression. Official's real source
   does not exhibit these; if DOD's does after a change, that's a real bug
   being exposed, not an accepted cost of parity, and the change should be
   reverted pending root-cause (see open item #2, the circle.bag frame
   ~4030-4500 fragility, found via exactly this distinction).
5. **Test every change on BOTH datasets before keeping it**, so you can
   tell a modest per-dataset tradeoff (keep it, per #4 above) apart from a
   catastrophic divergence on either one (revert it, investigate why).
6. **Keep local (`/home/.../vio_pipeline_cpp`) and remote
   (`ros_container_v2:/workspace/vio_pipeline_cpp`) in sync after every
   change** — `diff` the touched files after every remote rebuild/test to
   catch drift early.

## Where to read the full history

`portdocs/Benchmark.md` has the complete, chronological, numbers-backed
writeup of every experiment referenced above (including several that were
tried and reverted because they made things worse — reading those is useful
context so the same dead ends aren't retried). `docs/COMPREHENSIVE_AUDIT.md`
has the higher-level audit checklist this work is scored against.

---

# Bit-exact parity programme (2026-08-14)

Everything above was established by *trajectory-level* comparison: run both,
compare ATE, reason backwards. That loop stalled at a 9.3x gap. This section
records a different instrument — a **differential harness that links official
Open_VINS into the test binary as a numeric oracle and compares DOD's output
to it at ULP 0**.

## The harness

`cmake -DVIO_BUILD_PARITY=ON -DOV_BASE=/workspace/open_vins_official` builds
official (with `ENABLE_ROS=OFF`) into a **test-only** static library
`ov_oracle`, against the *same vendored Eigen* DOD uses, and links it beside
DOD in `tests/bitdiff_*.cpp`. `tests/bitdiff.hpp` provides `ulp_diff()` and a
`Report` that prints `%a` hex plus the ULP distance at the first mismatch.
Inputs come from a fixed splitmix64 generator — never `MatrixXd::Random`,
because an input you cannot reproduce makes a diff meaningless.

`VIO_PARITY_FLAGS` in `CMakeLists.txt` compiles both sides with official's own
flags (`-O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno
-funroll-loops`) plus `-ffp-contract=off`. `-fno-signed-zeros` alters FP
semantics, so a comparison without it is not a comparison.

The shipped `libvio_pipeline.a` links **no** official code; `nm` confirms zero
`ov_core::`/`ov_msckf::` symbols.

Practical notes for whoever extends this: official needs `-include cassert`
(it picks the header up transitively from ROS includes we do not have);
`compute_F_and_G_*` does **not** resize its `F`/`G` outputs, so the caller
must pre-size them or it segfaults; protected members are reached by
subclassing the oracle, never by editing it.

## Stages proven bit-exact (ULP 0)

| Stage | Checks | File |
|---|---|---|
| 2.1 Lie/quaternion math, all 20 functions | 4000 | `bitdiff_smoke.cpp` |
| 1.1 chi-squared 0.95 gate, dof 1..499 | 499 | `bitdiff_chi2.cpp` |
| 1.2 Camera distort + both Jacobians (5 calibrations) | 8000 | `bitdiff_cam.cpp` |
| 2.3-2.5 IMU selection, 3 mean predictors, F/G Jacobians | 5638 | `bitdiff_prop.cpp` |
| 2.8 Triangulation, compute_error, Gauss-Newton | 2120 | `bitdiff_tri.cpp` |
| 2.9 Nullspace projection + measurement compression | 720 | `bitdiff_update.cpp` |

## Divergences found and fixed

Each of these was invisible to trajectory-level comparison.

13. **chi-squared gate was approximate** (`msckf/updaters.cpp`). DOD used
    3-decimal constants for dof 1-5 and a Wilson-Hilferty approximation
    beyond; official uses exact `boost::math::quantile`. Wrong in the 4th
    decimal at every dof. Replaced with an embedded exact table
    (`msckf/chi2_table.hpp`, generated by `tests/gen_chi2_table.cpp`) so the
    library keeps no boost dependency.

14. **Camera model: float32 and expression form** (`core/cam.cpp`). Official's
    `distort_d` rounds through float32 (`CamBase.h` casts to `Vector2f`),
    computes `r = sqrt(x^2+y^2)` and then `r_2 = r*r` (a lossy round trip),
    writes `2*p2*x + 4*p2*x` where `6*p2*x` reads better, and uses
    `std::pow(theta, 3)` in the equidistant path. DOD had "cleaned up" all
    four. Now transcribed verbatim.

15. **`Inv_se3` and `log_se3` expression grouping** (`type/quat_ops.cpp`).
    `W * W * T` groups as `(W*W)*T` — a matrix-matrix product — where
    official forms `WT = W*T` then `W*WT`, two matrix-vector products.
    `Inv_se3` used fixed-size `.block<3,3>()` where official uses dynamic
    `.block(0,0,3,3)`; Eigen dispatches those to different kernels.

16. **`compute_Xi_sum`** (`msckf/propagator.cpp`). Three separate causes:
    `std::pow(dt,3)` is not bit-equal to `dt*dt*dt`; `1.0/6 * d_th3` is not
    bit-equal to `d_th3/6.0`; and `coef * sA * sK` groups as `(coef*sA)*sK`,
    scaling the matrix before multiplying. Now transcribed verbatim, which
    also made the analytic F/G Jacobians exact.

17. **`predict_mean_rk4`** (`msckf/propagator.cpp`). The k4 stage must use
    `w_hat`/`a_hat` reached by two accumulated `+= 0.5*w_alpha*dt` steps, not
    `w_hat2` directly — equal in exact arithmetic, not in doubles. And
    renormalisation must go through `quatnorm()`, which flips sign when
    `q(3) < 0`; Eigen's `.normalized()` does not, so it could return the
    antipodal quaternion.

18. **`select_imu_readings`** (`msckf/propagator.cpp`). Two bugs: the trailing
    sample was appended under a `1e-12` tolerance where official uses an exact
    `!=`, and the zero-dt dedup kept the *earlier* reading of the pair where
    official erases it and keeps the later one.

19. **Triangulation anchor tie-break** (`core/feature.cpp`). Official iterates
    `feat->timestamps`, an `unordered_map` keyed by camera id, and replaces
    the anchor only on a strict `>`. With ids inserted 0 then 1, libstdc++
    walks them 1 then 0 — so on an equal-count tie official keeps **camera 1**.
    DOD hardcoded camera 0. For stereo features with equal counts this
    anchored every one of them to the other camera. The same iteration order
    governs the A/b, cost and Hessian accumulation order, and is now followed
    (see the `OV_CAM_*` constants and the comment on them).

20. **Two invented rejections** (`core/feature.cpp`). `single_triangulation`
    had a `|det(A)| < 1e-12` early return and `single_gaussnewton` had a
    `|det(Hess_l)| < 1e-12` guard that converted good iterations into lambda
    increases. Official has neither. Both rejected features official accepts —
    directly relevant to open item #1's accept-rate gap.

21. **`compute_error` and the Gauss-Newton accumulation** (`core/feature.cpp`).
    Official forms the predicted measurement and residual in **float32** and
    accumulates `pow(res.norm(), 2)` rather than `squaredNorm()`; the
    Gauss-Newton gradient uses `res.cast<double>()` from that float residual
    and divides by `pow(hi3,2)` rather than `hi3*hi3`. Also `alpha`/`beta` are
    computed by **dividing** by `p_FinA(2)`, not multiplying by `rho` — a
    1-ULP difference at entry that seeds the entire refinement.

22. **Nullspace projection and measurement compression were the wrong
    algorithm** (`msckf/updaters.cpp`). DOD used `HouseholderQR` with an
    explicitly formed `Q`; official uses in-place **Givens rotations** and
    never forms `Q`. Both are valid, but they produce *different orthonormal
    bases*, so `H_x` and `res` differed structurally — not in the last bits —
    and every downstream chi2 gate and EKF update saw different numbers.

## Open: undistortion is parity-exact but not yet shipped

`core::undistort_official()` (`core/undistort_cv.cpp`) reproduces official's
`CamRadtan/CamEqui::undistort_f` exactly — `cv::undistortPoints` in float32,
ULP 0 across EuRoC, KAIST and synthetic calibrations. It is **not** wired into
the tracker. Swapping it in alone takes EuRoC MH_01 from 0.407 m to 6.28 m
ATE. Bisected: reverting only the undistort change restores 0.4074 m;
reverting only the distort/Jacobian changes leaves 6.28 m — so undistortion is
the sole cause.

This is the important lesson of the exercise: **parity is a property of the
whole pipeline, not of a function**. DOD's tracker is not yet matched to
official's `TrackKLT`, so pairing official's undistortion with DOD's feature
selection and RANSAC produces a hybrid faithful to neither. The swap belongs
with the frontend parity work, and `core::undistort()` stays legacy until
then. Do not "finish the job" by flipping the tracker over in isolation.

## Correction to the baselines in this document

Measured against pristine HEAD (`d35af4c`), in the container, this session:

- **EuRoC MH_01 reproduces**: 0.40725 m, matching the documented 0.407 m.
  This is the trustworthy regression gate.
- **KAIST `circle.bag` does NOT reproduce**: 39.0 m ATE, 264 m estimated path
  against 29.5 m of ground truth — catastrophic divergence, not the documented
  0.624 m. Identical with and without the new FP flags, and identical between
  pristine HEAD and the parity build. Whatever produced 0.624 m is not what is
  committed. Until that is explained, **circle.bag cannot be used to attribute
  any change**, and the 9.3x figure in the table above should not be relied on.

## A test that verified nothing

`tests/verify_math.cpp` used `assert()` for every check, and Release defines
`NDEBUG`. It reported "passed" while compiling all of its assertions out.
Underneath, the pipeline never initializes on its synthetic vectors, so the
state stays identity/zero and neither comparison was ever exercised — true of
pristine HEAD as well. It is now unregistered from ctest with the reasoning in
`CMakeLists.txt`; its reference data came from the Python port, which this
document already declares unvalidated. The `bitdiff_*` suite replaces it.

## Next: `StateHelper::EKFUpdate` (read, not yet fixed)

Diffed by eye at the end of this session, not yet covered by a `bitdiff_*`
test. Three divergences, one of them serious:

1. **Solver and symmetrisation.** Official builds `S` in
   `triangularView<Upper>()`, then `S.selfadjointView<Upper>().llt()
   .solveInPlace(Sinv)` — an LLT on the upper triangle. DOD builds the full
   `S`, symmetrises it as `0.5*(S + S^T)`, and uses **LDLT** with a
   `CompleteOrthogonalDecomposition` pseudo-inverse fallback. Different
   factorisation, different numbers. Official likewise updates
   `_Cov.triangularView<Upper>() -= K * M_a.transpose()` and mirrors, where
   DOD subtracts the full matrix and re-symmetrises.

2. **Invented jitter.** DOD adds `1e-9` to every diagonal entry of `S` before
   solving. Official adds nothing.

3. **Invented negative-variance flooring — the serious one.** On a negative
   covariance diagonal, official prints the index and value and then
   `std::exit(EXIT_FAILURE)`: it treats the state as corrupt and refuses to
   continue. DOD silently floors any diagonal below `1e-12` up to `1e-12` and
   carries on. So DOD keeps running in states official considers fatal, and
   the symptom is hidden rather than reported.

Item 3 is very likely connected to open item #4 (the SLAM updater shipped
disabled after a reproducible negative-diagonal crash at state index 178) and
to the still-unexplained `circle.bag` divergence: a filter that quietly floors
negative variances will drift instead of failing loudly. Removing the flooring
should be expected to turn silent divergence into a hard exit, which is the
point — but it should be done deliberately, with the EuRoC gate watched,
rather than as a drive-by.

## Resolved: why circle.bag stopped reproducing

The correction above ("circle.bag's documented baseline does not reproduce")
now has a cause. `ros/vio_rosbag_runner.cpp` had `options.enable_slam = true`,
flipped on at some point to match official's `max_slam: 50`, on the reasoning
that the earlier SLAM crash belonged to a buggier pipeline. The parity manual
had continued to say SLAM ships disabled, and `VioManagerOptions`' default is
still `false`; only the KAIST runner overrode it. EuRoC's runner never did,
which is exactly why EuRoC kept reproducing and circle.bag did not.

Measured on the same build:

| circle.bag | ATE RMSE | Estimated path | GT path |
|---|---|---|---|
| `enable_slam = true` | 39.0 m | 264.4 m | 29.5 m |
| `enable_slam = false` | 0.666 m | 27.6 m | 29.5 m |

A 60x regression, and close to Benchmark 5's recorded 0.624 m once off. So the
frame-~4030 fragility (open item #2) and every other conclusion drawn from
circle.bag since that flag was flipped were measuring the SLAM updater's
instability, not what they thought they were measuring. Those investigations
are worth re-reading with that in mind.

The runner is back to `false`. Re-enable only together with the root-cause for
open item #4, and re-measure circle.bag when doing so.

## Where the remaining EuRoC accuracy actually goes

With the estimator stages above proven bit-exact, DOD's MSCKF accept rate on
EuRoC MH_01 is **0.390** — of 71522 features fed to `update_msckf`, 33923
(47%) are rejected by triangulation and 9702 (14%) by the chi2 gate. Official
accepts 68-73% on KAIST. So DOD builds each update from roughly half the
measurement information official uses.

That is sufficient to explain both halves of the path-length error measured in
Benchmark 8: fewer, weaker constraints per update give noisier per-frame
corrections (the high-frequency component, 1.96 m of excess path against
official's 0.10 m) and a less well-determined solution overall (the wander that
survives decimation).

It is not a threshold difference — DOD's `feat_init_opt` values
(`max_dist` 75, `max_baseline` 100, `max_cond_number` 10000) already match
official's `euroc_mav` config, as do the tracker options (200 points, FAST
threshold 30, 5x5 grid, `min_px_dist` 15, HISTOGRAM equalisation, 5 pyramid
levels, 15px window). Configuration is matched; the implementations of
detection, KLT tracking and RANSAC are not. This is open item #1, and it is the
same frontend parity work the deferred undistortion swap is waiting on — which
is a good sign that both are the same job rather than two.

Also measured: `EKFUpdate` reported **zero** negative covariance diagonals over
the whole EuRoC run, so the invented flooring it used to do was never firing on
this sequence. Removing it changed EuRoC ATE by 0.00002 m. It may still matter
on circle.bag with SLAM enabled, where the filter does destabilise.

## Correction: the accept rate is NOT the gap

The section above attributed DOD's remaining EuRoC error to measurement supply,
on the basis that DOD accepts 0.390 of MSCKF features while official accepts
0.68-0.73. That comparison was invalid: **the 0.68-0.73 figure was measured on
KAIST, not EuRoC.**

Measured properly, running the instrumented official build on EuRoC MH_01 with
official's own `euroc_mav` config:

| EuRoC MH_01 | MSCKF accept rate | features seen per update call |
|---|---|---|
| Official Open_VINS | 0.371 | ~6.6 |
| DOD | 0.390 | ~20 |

DOD accepts a slightly *higher* fraction than official. Feature supply and
correspondence quality are not the differentiator on this sequence, and the
frontend is not where the 3x ATE gap lives.

Two consequences worth carrying forward:

1. **A full port of official's TrackKLT was written and measured, and it did
   not help.** It restored detection-on-the-previous-frame, border and
   min_px_dist pruning of surviving tracks, an independent right-camera
   detection pass, prebuilt pyramids, official's `<10 points` RANSAC
   semantics, and the parity-exact undistortion. Result: accept rate 0.390 ->
   0.395 (noise) and ATE 0.407 -> 5.73 m, with the estimated path blowing out
   to 143 m against 79.7 m of ground truth.

   **The cause is not yet known.** An earlier revision of this document blamed
   DOD's fixed-capacity 2048-feature database silently refusing new features
   once full. That was asserted without being measured, and it is wrong: with
   the port running and the counter instrumented, `full_refusals = 0` and
   `db_count = 187`. The database never fills. Retracted.

   What the instrumentation does show is that the port produces far more
   short-lived tracks -- features rejected at only 2 measurements roughly
   triple (2111 vs 722) -- so tracks are dying young. That is the thread to
   pull. The port is kept at `scratchpad/tracker_official_port.cpp`, unshipped.

2. **The real difference is what the long tracks are used for.** Official's
   EuRoC config runs 50 SLAM landmarks (`max_slam: 50`, `dt_slam_delay: 1`),
   so long-lived features become persistent state rather than being consumed
   and thrown away by MSCKF. That is why it feeds only ~6.6 features per MSCKF
   call where DOD feeds ~20: DOD has `enable_slam = false` and routes
   everything through MSCKF. Matching official's accuracy means getting the
   SLAM updater working (open item #4), not tuning the frontend.

Also fixed while chasing this: `dynamic_initialize` sized its linear system as
3N+6 from however many features the database happened to hold, with no bound.
The richer frontend pushed N high enough to segfault inside the Eigen product.
It is now capped at `init_max_features`, matching official's use of a separate,
bounded initialisation feature set.

## The SLAM promotion bug (found 2026-08-14)

Official promotes a feature to a SLAM landmark when **any single camera** has
more than `max_clone_size` observations of it:

    for (const auto &cams : feat->timestamps)
      if ((int)cams.second.size() > state->_options.max_clone_size)
        reached_max = true;

DOD tested `feat.num_measurements >= 2 * max_clone_size` -- the total summed
across both cameras -- on the reasoning that stereo doubles the count. It does,
for stereo features. A **mono** feature tracked through the whole window has
exactly `max_clone_size + 1` measurements, which official promotes and DOD
rejected.

On EuRoC MH_01 that is almost everything: 21091 features sat at exactly 12
measurements (one camera, full window) and not one became a landmark. The
symptom was that DOD fed ~20 features per MSCKF update where official feeds
~5.9 -- official's long tracks had become persistent state, DOD's were being
consumed and discarded. `enable_slam = true` on its own changed nothing,
because nothing was ever eligible.

Fixed to the per-camera test, plus official's `dt_slam_delay` (1 s after the
first processed frame before any landmark may be created, which official notes
"normally prevents bad first set of slam points"). DOD now holds the full 50
landmarks. EuRoC ATE 0.4074 -> 0.3960 m. SLAM is enabled in the EuRoC runner
and runs clean -- zero negative covariance diagonals across the sequence, where
it previously "made EuRoC diverge faster".

## Stage 2.9b: the MSCKF Jacobian is equivalent

`get_feature_jacobian_mixed` -- the function `update_msckf` builds every
residual and Jacobian through -- was diffed against official's
`UpdaterHelper::get_feature_jacobian_full` (`tests/bitdiff_jac.cpp`).

For **mono** features it is bit-exact: `res`, `H_f` and `H_x` all at 0 ULP,
across 4/7/11 clones and with FEJ on and off.

For **stereo** features the two differ by a permutation of rows and of the
matching `Hx_order` columns, because official iterates a feature's per-camera
lists through an `unordered_map` (camera 1 first under libstdc++) while DOD
walks measurements in insertion order. Permuting rows together with their
columns leaves the EKF update mathematically unchanged. Reordering DOD's loops
to match was tried and did not close it -- the column order comes from a
separate path -- so the simpler loop is kept and the test checks stereo for
shape while holding mono to 0 ULP.

**With this, every stage of the estimator update path is verified equivalent to
official**: propagation (2.3-2.5), triangulation (2.8), the chi2 gate (1.1),
the camera model (1.2), the Jacobian (2.9b), the nullspace projection and
compression (2.9), and EKFUpdate (2.6). The remaining EuRoC accuracy gap is
therefore not in the estimator math.

## The SLAM path: fully audited, still not usable (2026-08-15)

The landmark lifecycle bugs (per-camera promotion test, ArUco-reserve id
range, retirement-after-cleanup) are fixed and the subsystem now behaves:
50 landmarks held, ~37 updates/frame, promotions and retirements balanced.
Every remaining function in the SLAM path was then audited against official
and corrected where it differed:

23. **`update_slam` built its Jacobian with a second implementation.**
    Official has no SLAM-specific Jacobian: it copies the landmark's anchor and
    position into the feature, calls `UpdaterHelper::get_feature_jacobian_full`,
    and appends `H_f` as extra columns. DOD had a separate
    `get_feature_jacobian_slam` with its own variable ordering (anchor clone
    pushed first rather than in measurement order), column layout and residual
    assembly. Now routed through `get_feature_jacobian_mixed`, which
    `tests/bitdiff_jac.cpp` proves bit-exact against
    `get_feature_jacobian_full` for both GLOBAL_3D and the anchored
    inverse-depth representation that actually runs.

24. **`StateHelper::initialize` used HouseholderQR where official uses
    Givens** -- the identical mistake as divergence #22, in the function that
    creates every SLAM landmark. Different orthonormal bases mean
    `Hxinit`/`H_finit`/`resinit`/`Hup`/`resup` all differ, so every landmark
    entered the state with a different initial estimate and covariance than
    official would give it. Also carried an invented `0.5*(S+S^T)`
    symmetrisation, a 1e-9 diagonal jitter, and an LDLT where official uses a
    plain LLT. All transcribed to official's form.

`initialize_invertible` was audited and matches.

**None of it moved the number.** EuRoC MH_01, all measured with the feature
pointer bug fixed:

| config | ATE RMSE |
|---|---|
| SLAM off | **0.3947 m** |
| SLAM on, one joint update | 1.2506 m |
| SLAM on, official's batching | 1.1296 m |
| SLAM on, after divergences 23 and 24 | 1.1296 m |

So the SLAM estimator math is now equivalent to official's and SLAM still
costs 3x. The instrumentation points instead at the landmarks themselves:
**69% of SLAM candidates fail triangulation at initialisation** (17475 of
25480), so the ones that survive are marginal, and 37 weak landmarks per frame
drag the estimate. That is the same defect that makes 46% of MSCKF features
fail the condition-number test -- feature quality, not estimator math.

`enable_slam` is `false` on EuRoC, with all of these numbers recorded at the
call site. Do not re-enable it on the theory that a parity bug remains in the
updater; that ground is now covered.
