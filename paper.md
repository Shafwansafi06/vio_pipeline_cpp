---
title: 'vio_pipeline_cpp: A Data-Oriented C++ MSCKF Visual-Inertial Odometry Pipeline'
tags:
  - C++
  - robotics
  - visual-inertial odometry
  - state estimation
  - data-oriented design
authors:
  - name: Shafwan Safi
    affiliation: 1
  - name: Nikhil Vashista
    affiliation: 1
affiliations:
  - name: Indian Institute of Science Education and Research (IISER) Bhopal, India
    index: 1
date: 20 August 2026
bibliography: paper.bib
---

# Summary

`vio_pipeline_cpp` is a C++17 implementation of the multi-state constraint
Kalman filter (MSCKF) for visual-inertial odometry (VIO)
[@mourikis2007msckf], the state estimation problem of recovering a moving
sensor's pose from a stream of camera images and IMU measurements. It is
designed as a data-oriented reimplementation of the algorithm popularized by
OpenVINS [@geneva2020openvins]: the estimator equations are OpenVINS's, and
every stage is validated bit-identical against an independently built copy
of OpenVINS on ten public benchmark sequences; what differs is memory
layout. The per-frame estimator state is held in fixed-capacity structs with
compile-time-asserted bounds, the frame path performs no heap allocation
(verified with an automated test and with `valgrind`/DHAT allocation
profiling), and the feature database separates its search index from its
payload storage so that inserting or removing a track does not move
2.4 kB of unrelated feature data. The package builds under CMake, links only
against a vendored Eigen for the core estimator, and optionally against
OpenCV and ROS1 for the stereo tracker and dataset/rosbag runners; five
`ctest`-registered invariant tests and a bit-exact parity suite against
OpenVINS form its continuous test gate.

# Statement of need

VIO is used on platforms -- small UAVs, embedded rovers, AR/VR headsets --
where per-frame compute budget and worst-case memory footprint are
first-class constraints, not afterthoughts. OpenVINS is a widely used,
carefully engineered research platform, but its C++ implementation follows
conventional object-oriented design: polymorphic state variable types,
`std::vector`-based containers, and per-feature and per-measurement heap
allocation. This is a reasonable choice for a research platform prioritizing
extensibility, but it leaves open the practical question of what a
fixed-capacity, allocation-free reimplementation of the *same* estimator
buys, and what it costs, for teams whose deployment target is
compute-constrained. `vio_pipeline_cpp` exists to answer that question with
a runnable artifact rather than an estimate: a drop-in-comparable MSCKF
pipeline that a researcher or engineer working on embedded/constrained VIO
can build, benchmark against OpenVINS on their own data, and read as a
worked example of applying data-oriented design constraints -- no per-frame
allocation, no virtual dispatch, existence-based processing, bounded
capacity asserted at initialization -- to a non-trivial, published
estimation algorithm rather than to a synthetic benchmark. The codebase's
own house-style document (`CLAUDE.md`, `.claude/skills/dod-style/`) records
the constraints and the profiling discipline behind them, so the design
rationale ships with the code rather than living only in a paper.

# Design and functionality

The pipeline is organized as tables and transforms rather than objects: a
fixed-capacity `FeatureDatabase`, a fixed-capacity covariance and clone
window in `State`, and free functions (`propagate_and_clone`,
`update_msckf`, `update_slam`, `feed_measurement_camera_tracks`, ...) that
read one region of state and write another, with no method carrying hidden
state. Camera count, clone-window size, and feature-database capacity are
all runtime-configurable but statically bounded, and every bound is asserted
against its configured use at `init_*()` time rather than checked lazily on
the frame path. Three published techniques are integrated and cited in
place at their point of use: a square-root-form EKF update and a
feature-less dynamic initializer following sqrtVINS [@peng2025sqrtvins], and
a hovering-aware clone marginalization policy following Kottas, Wu, and
Roumeliotis [@kottas2013hover]. An optional, disabled-by-default MSCKF
update path implements SchurVINS's Schur-complement restructuring
[@fan2024schurvins] in the same data-oriented layout, usable as a
second reference update strategy without being the pipeline's default.

Two dataset transports are supported: ROS1 rosbag playback (matching how
OpenVINS is typically benchmarked, so the two can be compared through an
identical transport rather than through different dataset readers) and a
ROS-free reader for the EuRoC ASL folder layout, for use without a ROS
installation. Validation is layered: `bitdiff_*` tests compare individual
stages against a recorded OpenVINS oracle bit-for-bit; `verify_*` tests
(registered with `ctest`) check invariants such as allocation-free
propagation, clone-pointer and covariance-offset consistency across
marginalization, and finite-difference agreement of the SLAM measurement
Jacobian; and an end-to-end accuracy sweep over ten public EuRoC and KAIST
sequences serves as the acceptance gate for any change to the estimator.

# Acknowledgements

We thank the OpenVINS, SchurVINS, and sqrtVINS authors for the published
algorithms this pipeline implements and benchmarks against.

# AI usage disclosure

Substantial portions of this codebase, its documentation, and this paper
were produced with the assistance of Claude (Anthropic), across several
model versions during development (Claude Opus 5, Claude Sonnet 5), used
for code generation, refactoring, profiling-driven optimization, test
scaffolding, documentation, and drafting of this paper and the associated
technical writeup (`paper/main.tex`). All AI-assisted output was reviewed,
tested against the project's accuracy and allocation gates, and edited by
the human authors, who made all design decisions, including the
data-oriented constraints the codebase enforces, the choice of reference
algorithms to integrate, and the acceptance criteria (bit-identical
trajectories, `ctest` invariants) used to validate every change.

# References
