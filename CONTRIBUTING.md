# Contributing

## Before you write code

Read `CLAUDE.md` and `.claude/skills/dod-style/SKILL.md` first. This is a
data-oriented C++ codebase with a small, enforced set of house rules (no
per-frame allocation, no virtual dispatch, fixed-capacity state, every
performance change measured before and after). A change that violates them
will not be reviewed favourably, however fast it is.

## Building

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
ctest --output-on-failure
```

The core estimator (`vio_pipeline`) and the five `ctest`-registered
`verify_*` tests need only Eigen (vendored in `vendor/eigen`) -- no ROS, no
OpenCV. The OpenCV-dependent tracker and the ROS-free EuRoC runner are
opt-in:

```bash
cmake -DVIO_BUILD_OPENCV_FRONTEND=ON -DVIO_BUILD_ASL_RUNNER=ON ..
```

The ROS1 rosbag runners (`VIO_BUILD_ROS1`) additionally need a ROS Noetic
environment; see `CLAUDE.md` for how the authors set that up in a container.

## The accuracy gate

A performance or refactoring change is not finished when it compiles and
`ctest` passes. It is finished when the ten-sequence ATE sweep
(`tools/sweep.sh`, documented in `CLAUDE.md`) is bit-identical to the
recorded baseline, or the change is argued and measured explicitly as an
accuracy change. This is not a formality -- every optimisation in this
repository's history was validated this way, and the two that were reverted
(the KAIST init-latency "fix", the naive angle-ranking dynamic-init "fix")
were caught precisely because this gate was run before accepting them, not
after.

## Style

- Match the surrounding code: `snake_case` functions/variables,
  `PascalCase` structs, `SCREAMING_SNAKE_CASE` constants. 100-column limit,
  4-space indent.
- Every new capacity constant needs a comment enumerating the worst case
  that justifies the number, and an `init_*()`-time assert against it.
- Every new rejection/drop path needs its own counter, visible in the
  runner's summary output -- see `imu_buffer_evictions`,
  `db_full_refusals` for the pattern.
- Comments earn their place by carrying a measurement or a root cause, not
  by restating the code.

## Reporting issues

Use the GitHub issue tracker. For a numerical or accuracy issue, include
which sequence, which stage (`tools/dod_asl_runner.cpp` prints per-stage
timing and drop counters), and, if possible, the output of `VIO_INIT_DEBUG=1`
for anything initialization-related.

## Tests

New tests are one of three kinds (`tests/`, see `CLAUDE.md` Sec. 13):

- `bitdiff_*` -- exact match against a recorded oracle.
- `verify_*` -- an invariant that must hold (registered with `ctest`).
- `benchmark_*` -- timing only, never part of the correctness gate.

Register new `verify_*`/`bitdiff_*` tests with `add_test` in
`CMakeLists.txt`; a test that is not in `ctest` does not exist as far as CI
is concerned.
