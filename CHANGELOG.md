# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `LICENSE` (BSD-3-Clause), `CONTRIBUTING.md`, `CITATION.cff`, CI
  (`.github/workflows/ci.yml`: build + `ctest` on every push, plus a
  separate OpenCV-frontend build).
- `VIO_INIT_DEBUG` environment variable: per-candidate diagnostics for both
  the static and dynamic initializers.
- Drop counters on the frame-input path (`imu_buffer_evictions`,
  `imu_stale_drops`, `imu_window_truncations`, `frame_unpropagated_drops`,
  `feat_meas_overflow`), printed by every runner's summary line.
- `paper/` -- an ICRA-style writeup of the design, the ten-sequence
  benchmark, and the ablation, plus `paper.md`/`paper.bib` for JOSS.
- `scripts/error_profile.py` -- relative pose error over a 1 s window, to
  separate initialization-transient accuracy loss from steady-state drift.
- `scripts/summarize_reeval.py`, `scripts/vicon_gt_to_csv.py` -- the
  same-transport ten-sequence re-evaluation harness and its V1_* ground
  truth fix.

### Fixed
- `tests/verify_dynamic_init.cpp`'s `R_GtoI` lambda returned an Eigen
  `Product<>` expression referencing an already-destroyed temporary (missing
  the `.eval()` its sibling lambdas `v_G`/`p_G` correctly had) -- undefined
  behaviour, not a numerical precision issue. It happened to read back
  correct values on one toolchain and a zero matrix on another (confirmed by
  direct inspection: `R_GtoI(0)` came back as the zero matrix, not
  `R_GtoI0`), which cascaded into most of the test's 16 synthetic features
  never registering a valid projection and the test failing with no relation
  to the actual code under test. Found while setting up CI: the same source,
  same compiler flags, same vendored Eigen gave different results on two
  x86_64 machines, which does not happen for a real numerical-precision
  difference and pointed straight at undefined behaviour instead. Fixed with
  a one-line `.eval()`; verified `ctest` 5/5 on both machines afterward with
  no other change.
- IMU ring buffer evicted the *newest* sample once full instead of the
  oldest; a run whose initializer had not fired within ~50 s could freeze
  permanently. Now evicts the oldest quarter and counts it.
- `select_imu_readings` silently truncated at its scratch-buffer capacity
  and reported success; callers now refuse a saturated window.
- `propagate_and_clone`'s boolean failure return was discarded by its
  caller; a non-advancing timestamp now aborts the frame instead of running
  every later stage against an unpropagated state.
- `dynamic_initialize`'s gravity-convergence tolerance (`1e-3`, absolute)
  was tight enough that legitimate x86-vs-aarch64 floating-point rounding
  through an `EigenSolver` QR iteration crossed it, rejecting a correct
  solution on ARM (Jetson Orin Nano). Widened to `3e-3` with the measured
  spread recorded alongside it.
- `select_imu_readings`/`compute_disparity` capacity literals (`2048`,
  `64`) named and asserted at init instead of left bare.

### Removed
- The Python port (`msckf/*.py`, `core/*.py`, `type/*.py`,
  `initialize/*.py`, `config/*.py`, `setup.py`) -- superseded by the C++
  pipeline for a long time; nothing built or ran it.
- Two accidentally committed, unrelated agent-session transcripts under
  `docs/`.

## [0.1.0] - 2026-08-18

Initial DOD (data-oriented design) C++ MSCKF pipeline.

### Added
- Full MSCKF visual-inertial odometry pipeline: propagation, MSCKF and SLAM
  updates, feature triangulation, static and feature-less dynamic
  initialization (sqrtVINS Sec. V-A), hovering classifier (Kottas, Wu &
  Roumeliotis), optional SchurVINS-style update path (off by default).
- ROS1 rosbag runners (EuRoC and KAIST topic layouts) and a ROS-free ASL
  dataset runner.
- Five `ctest`-registered `verify_*` invariant tests and a `bitdiff_*`
  parity suite against official OpenVINS.
- `tools/sweep.sh` and the ten-sequence accuracy gate (EuRoC MH_01-05,
  V1_01-03; KAIST circle/infinity).

### Performance (measured against official OpenVINS, same rosbag transport)
- 2026-08-15/16: three EKF-update and allocation optimizations
  (measurement compression, chi-squared gate restructuring, 62% allocation
  cut via DHAT), each verified accuracy-neutral.
- 2026-08-17: feature database key/payload separation (removed ~9,556
  memcpy/frame), a squared-root-form EKF update, exact chi-squared
  early-accept, and a `STATE_COV_CAPACITY` right-sizing -- four
  independently measured optimizations, 1.34x combined, ten ATEs
  bit-identical before and after.
- 2026-08-17: benchmarked against the SchurVINS authors' own
  `ov_SchurVINS` release -- 1.97x faster on identical data.

### Fixed
- MH_02's dynamic-initializer velocity divergence, resolved via a
  feature-less initializer (sqrtVINS Sec. V-A) rather than the two
  workarounds it replaced.
