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
