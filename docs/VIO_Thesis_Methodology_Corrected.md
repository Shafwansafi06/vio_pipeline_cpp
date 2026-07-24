# Corrected Thesis Draft: System Architecture and Methodology

This document provides a corrected, implementation-aligned writeup of:

1. Chapter 3: System Architecture
2. Chapter 4: Methodology
3. A factual problems list found in the prior outline
4. A missing-items list for thesis completeness

The text below is written to match the current Python implementation in this repository.

---

## Chapter 3: System Architecture

### 3.1 Overview

The implemented system is a Python Visual-Inertial Odometry (VIO) pipeline integrated with ROS. It fuses IMU and monocular/stereo camera data to estimate pose in real time.

The estimator follows an error-state EKF architecture derived from MSCKF-style filtering. It supports normal FIFO operation (propagation + clone augmentation + visual update) and a hover-aware LIFO supervisory mode (propagation-only with frozen clone window and deferred covariance reconciliation).

The runtime is organized around:

1. A high-rate IMU-driven propagation path.
2. A camera queue processed against available IMU history.
3. A central manager (`VioManager`) orchestrating initialization, propagation, updates, and mode transitions.

### 3.2 Runtime Topology and Concurrency Model

The ROS node maintains:

1. An IMU callback that always feeds inertial measurements and publishes fast propagated odometry.
2. A timestamp-sorted camera queue protected by a mutex.
3. A camera update worker that processes queued frames once temporal alignment with IMU is feasible.

Important correction: the processing model is not strictly always two concurrent threads. It is configuration-dependent:

1. In asynchronous mode, camera processing runs in a spawned worker thread.
2. In synchronous mode, the same update routine runs inline.

Thus, the architecture is best described as a decoupled queue/worker design with optional multithreading.

### 3.3 Module Decomposition

The pipeline is decomposed into the following major subsystems.

#### 3.3.1 ROS Interface Layer (`main.py`)

Responsibilities:

1. Subscribe to IMU and camera topics.
2. Convert ROS messages into internal data types.
3. Rate-limit incoming camera frames.
4. Maintain sorted camera queue.
5. Trigger camera updates based on IMU-time availability.
6. Publish IMU-rate and post-update odometry/path outputs.

#### 3.3.2 Front-End Tracker (`TrackKLT`)

Responsibilities:

1. KLT optical flow tracking over image pyramids.
2. Monocular and stereo tracking paths.
3. Geometric outlier rejection using RANSAC with fundamental matrix estimation in normalized coordinates.
4. Feature detection refill using FAST on grid constraints.
5. Optional image histogram equalization / CLAHE preprocessing.
6. Update of persistent feature database.

#### 3.3.3 Feature Database (`FeatureDatabase`)

Responsibilities:

1. Persist per-feature per-camera history.
2. Store timestamps, distorted image coordinates, and normalized coordinates.
3. Provide query operators used by back-end scheduling, including feature sets containing target timestamps and stale tracks for marginalization/update.

#### 3.3.4 Inertial Initialization (`InertialInitializer`, `StaticInitializer`)

Responsibilities:

1. Gate initialization using visual disparity and IMU variance windows.
2. Estimate initial gravity-aligned orientation, gyroscope bias, accelerometer bias.
3. Initialize covariance and estimator state ordering.

#### 3.3.5 IMU Propagator (`Propagator`)

Responsibilities:

1. Buffer and interpolate IMU data over requested intervals.
2. Correct inertial measurements for bias/intrinsics.
3. Propagate state mean using one of three methods: discrete, RK4, analytical.
4. Propagate covariance via composed transitions and process noise accumulation.
5. Support both `propagate_and_clone` (FIFO) and `propagate_only` (LIFO).

#### 3.3.6 Updater Suite

1. `UpdaterMSCKF`: nullspace-projected feature constraints without augmenting feature state.
2. `UpdaterSLAM`: delayed initialization and update of persistent landmarks in the filter state.
3. `UpdaterZeroVelocity`: stationary-motion constraints with chi-square gating and disparity override checks.

#### 3.3.7 VIO Orchestrator (`VioManager`)

Responsibilities:

1. Coordinate tracker, propagator, initializer, and all updaters.
2. Execute FIFO/LIFO supervisory state machine.
3. Apply LIFO backward-chain association and quality filtering.
4. Apply deferred covariance strategy in LIFO and reconciliation on FIFO re-entry.

### 3.4 Coordinate Frames and Conventions

The implementation uses these frames:

1. Global frame `G`.
2. IMU frame `I`.
3. Camera frame `C` (one per camera).
4. Anchor frame `A` for anchored feature parameterizations.

Quaternion convention is JPL with ordering `[x, y, z, w]`, and orientation perturbations use left-multiplicative error-state updates.

Gravity in code is represented as `g = [0, 0, g_mag]^T` and appears in dynamics as a subtraction term in velocity/position propagation. This sign convention must remain consistent between equations and frame narrative.

### 3.5 End-to-End Data Flow

Per IMU packet:

1. IMU data is buffered.
2. Fast-state propagation provides immediate odometry output.
3. Camera worker may be triggered if idle.

Per camera frame:

1. Tracker computes feature correspondences and updates database.
2. Estimator propagates to camera timestamp:
   - FIFO: propagate + clone augmentation.
   - LIFO: propagate only.
3. Hover detection and smoothing are evaluated from frame-to-frame bearing residuals.
4. Mode transition logic executes.
5. Update path executes:
   - FIFO: MSCKF/SLAM/marginalization pipeline.
   - LIFO: state-only MSCKF using backward-chain features plus deferred covariance buffering.

### 3.6 FIFO/LIFO Supervisory State Machine (Architecture View)

Mode semantics:

1. `FIFO` (default): clone window grows each frame then oldest clones are marginalized.
2. `LIFO` (hover mode): clone window is frozen; propagation continues; backward-chain feature association supplies robust short-baseline constraints.

Entry/exit guards include smoothed hover decisions, baseline/velocity checks, and hard safety exits based on velocity and hover timeout.

### 3.7 Outputs and Interfaces

The node publishes:

1. Fast IMU-rate odometry (propagated estimate).
2. Post-update odometry.
3. Path and visualization products.

The architecture therefore separates high-rate propagation outputs from lower-rate full-update outputs.

### 3.8 Configuration Surface

Runtime behavior is extensively parameterized in options/config loaders, including:

1. Noise and process settings.
2. Integration method selection.
3. Feature tracking and update thresholds.
4. Hover detection and LIFO association parameters.
5. ZUPT thresholds and gating behavior.

---

## Chapter 4: Methodology

### 4.1 State Representation and Error-State Formulation

The nominal state is composed of:

1. IMU core state: orientation, position, velocity, gyro bias, accel bias.
2. Optional calibration sub-states: camera time offset, camera extrinsics/intrinsics, IMU intrinsic terms and gravity sensitivity.
3. Clone states (pose history).
4. Optional SLAM landmarks.

The error-state uses additive perturbations for Euclidean variables and left-multiplicative perturbations for orientation.

Covariance propagation/update is performed in this error-state space.

### 4.2 IMU Measurement Model and Propagation

#### 4.2.1 Corrected IMU Signals

Raw IMU measurements are corrected by:

1. Bias removal.
2. Intrinsic correction matrices (`D_w`, `D_a`).
3. Sensor-frame alignment rotations.
4. Gravity sensitivity coupling matrix `T_g` in gyro channel.

#### 4.2.2 Continuous-Time Dynamics Used by the Filter

The estimator follows standard rigid-body inertial kinematics, with propagated orientation, position, velocity, and random-walk biases.

Given code convention, gravity enters as a subtraction term in velocity/position updates.

#### 4.2.3 Discrete Integration Methods

Three modes are implemented:

1. Discrete first-order method.
2. RK4 propagation.
3. Analytical propagation using precomputed `Xi` terms and associated Jacobians.

The analytical path computes `Xi_1` to `Xi_4` blocks and uses them for both mean and linearized transition/noise mapping.

#### 4.2.4 Covariance Propagation

For each IMU sub-interval:

1. Transition `F` and noise mapping `G` are computed.
2. Discrete noise `Q_d = G Q_c G^T` is formed.
3. Interval terms are accumulated over full camera interval.

Then EKF propagation applies to relevant state blocks, preserving cross-covariances with existing variables.

#### 4.2.5 Clone Augmentation vs Propagation-Only

1. FIFO uses `propagate_and_clone`: propagated state then stochastic clone augmentation.
2. LIFO uses `propagate_only`: no clone augmentation; clone window remains frozen.

When camera-time offset calibration is active, clone augmentation includes additional covariance coupling to the offset state.

### 4.3 Visual Front-End Methodology

#### 4.3.1 Tracking

KLT optical flow is performed with:

1. Window size `15x15`.
2. Pyramid levels = 5.
3. Initial-flow use for prediction/refinement.

#### 4.3.2 Outlier Rejection

Tracked correspondences are filtered via RANSAC fundamental matrix estimation in normalized image coordinates.

#### 4.3.3 Feature Management

Feature observations are inserted into a persistent database and later queried for:

1. Lost tracks.
2. Tracks tied to marginalization timestamps.
3. Tracks available for update windows.

### 4.4 Feature Triangulation and Refinement

#### 4.4.1 Linear Triangulation

For each feature:

1. Anchor camera/timestamp is selected.
2. Normal-equation system is accumulated from bearing orthogonality constraints.
3. Triangulation is accepted only if condition/depth checks pass.

#### 4.4.2 1D Triangulation Variant

A depth-only variant along anchor bearing is available for stronger numerical stability in weak baseline geometry.

#### 4.4.3 Nonlinear Refinement

Levenberg-Marquardt optimization refines inverse-depth parameters with adaptive damping and convergence thresholds.

### 4.5 MSCKF Update Method

#### 4.5.1 Constraint Construction

For each candidate feature:

1. Build full Jacobians wrt state and feature.
2. Nullspace project to remove explicit feature dependence.
3. Compress measurement system via helper routines.

#### 4.5.2 Statistical Gating

Innovation covariance is formed from marginalized covariance and pixel noise.

Feature constraints are rejected if:

1. Innovation conditioning is poor.
2. Chi-square test fails under configured multiplier.
3. Projected system is degenerate or non-finite.

#### 4.5.3 EKF Update

Accepted constraints are stacked and applied in EKF form to update state mean and covariance.

### 4.6 SLAM Landmark Management

The estimator supports delayed initialization and update of persistent landmarks with multiple representation choices (global and anchored forms).

Landmark housekeeping includes:

1. Initialization from quality-approved tracks.
2. Update and failure handling.
3. Re-anchoring when clone history changes.
4. Marginalization logic to maintain bounded state size.

### 4.7 Zero-Velocity Update (ZUPT)

ZUPT is applied under stationary conditions.

Residuals are built from corrected gyro/accel channels across IMU windows and whitened using measurement noise and interval duration.

Acceptance combines:

1. Innovation chi-square test.
2. Velocity magnitude threshold.
3. Optional visual-disparity override checks.

Important implementation note: default residual Jacobian structure does not always include velocity block unless integrated-accel mode is enabled in the updater logic.

### 4.8 Hover-Aware FIFO/LIFO Supervisory Framework

This is the primary custom framework in the implementation.

#### 4.8.1 Hover Detection Statistic

For consecutive camera frames, normalized bearings are compared after compensating by IMU-derived relative rotation:

`r_i = || b_curr,i - C_q_hat b_prev,i ||^2`

This residual is computed on inlier sets selected by baseline-dependent policy:

1. Larger baseline: epipolar-consistent filtering.
2. Smaller baseline: bearing-threshold filtering.

Decision statistic `d_k`:

1. Mean residual when inliers are sufficient.
2. Robust fallback median over all residuals when strict inlier count is too small.

Binary hover decision is thresholded by `hovering_threshold`.

#### 4.8.2 Temporal Smoothing and Guarding

Binary decisions are smoothed using fixed-length deque hysteresis (`hover_smoothing_required`).

FIFO to LIFO entry requires:

1. Smoothed hover-positive decision.
2. Baseline not exceeding `hover_entry_max_baseline`.
3. Velocity not exceeding `hover_entry_max_velocity`.

LIFO to FIFO exit occurs when any holds:

1. Smoothed hover decision turns negative.
2. Velocity exceeds `hover_exit_velocity_threshold`.
3. LIFO duration exceeds `hover_max_duration_sec`.

#### 4.8.3 LIFO Backward-Chain Association

In LIFO mode, current observations are associated against backward feature chains.

Association pipeline:

1. Build per-camera candidate costs.
2. Solve assignment with Hungarian matching.
3. Optionally require forward/backward mutual consistency.
4. Use descriptor fallback under configured conditions.
5. Update per-chain quality and misses.
6. Prune weak chains and gate output by quality/min-length.

#### 4.8.4 Deferred Covariance Strategy in LIFO

Per LIFO frame:

1. Apply MSCKF update to state mean.
2. Restore covariance snapshot (state-only update effect retained, covariance deferred).
3. Accumulate hover measurements for later batch covariance reconciliation.

This intentionally relaxes strict one-step mean-covariance coupling as an engineering choice for near-degenerate hover geometry.

#### 4.8.5 FIFO Re-Entry Reconciliation

On LIFO exit:

1. Perform one covariance update using accumulated hover measurements.
2. Restore mean snapshot handling as implemented.
3. Resume standard FIFO update and clone augmentation.

### 4.9 Inertial Initialization Procedure

Initialization is gated by:

1. IMU window duration and variance checks.
2. Visual disparity checks across two temporal windows.
3. Stillness/jerk logic used to decide static initializer invocation.

Static initializer estimates:

1. Initial orientation aligned to gravity.
2. Initial gyro bias from average angular rate.
3. Initial accel bias from average specific force and gravity model.
4. Initial covariance for estimator startup.

---

## Problems List (Factual Issues in Prior Outline)

1. Gravity sign/frame statement was inconsistent.
   The narrative claimed negative-z gravity while equations and code usage reflected a positive-z gravity vector subtracted in dynamics.

2. Forced hover and forced ZUPT via MSCKF failure-counter coupling was described but is not clearly implemented in current supervisor logic.

3. LIFO association text overstated details not exactly present.
   The implementation uses mutual Hungarian consistency and conditional descriptor fallback, not a universal nearest-neighbor-ratio formulation.

4. ZUPT Jacobian description implied always including velocity sensitivity, while default updater configuration does not always include velocity block.

5. Threading statement implied always-two-thread operation, but actual execution can be synchronous or asynchronous depending on configuration.

6. Hover decision statistic wording missed fallback behavior details.
   Implementation uses mean for sufficient inliers and median fallback over full residual set when inlier set is sparse.

---

## Missing Things List (Recommended Additions to Thesis)

1. Explicit assumptions and observability conditions section.
   State all assumptions behind hover degeneracy handling, camera geometry, and bias behavior.

2. Consistency-analysis protocol.
   Provide exact NEES/NIS computation protocol, confidence bounds, sampling windows, and acceptance criteria.

3. Runtime and computational complexity section.
   Include per-module timing, asymptotic complexity, and update-latency profiling.

4. Reproducibility and experiment protocol section.
   Document config files, commit hash, datasets, split policy, seed control, and hardware/software environment.

5. Threats-to-validity section.
   Discuss sensitivity to texture scarcity, blur, calibration drift, and dataset domain shift.

6. Safety and failure-management section.
   Define estimator confidence degradation behavior and fail-safe handling for UAV deployment contexts.

7. Detailed notation table.
   Add a unified symbols table for frames, rotations, quaternions, residuals, Jacobians, and covariance terms.

8. Figure set completion.
   Add architecture graph, data-flow timeline, FIFO/LIFO state machine diagram, and deferred-covariance sequence diagram.

9. Ablation and sensitivity plan.
   Include planned sweeps for hover thresholds, smoothing length, quality parameters, and ZUPT gates.

10. Dataset characterization subsection.
   Provide motion-regime bins, texture level characterization, and lighting/blur conditions for each evaluation set.
