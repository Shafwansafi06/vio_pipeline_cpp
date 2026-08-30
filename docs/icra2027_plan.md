# ICRA 2027 — history & task plan

Last updated: 2026-08-26. Branch: `research/altitude-covariance`.

## 1. Where the project stands (repo history)

DOD (this codebase) is a from-scratch DOD/C++ MSCKF VIO pipeline benchmarked
against OpenVINS ("official"). Chronological arc from git log + memory:

1. **Baseline correctness** — DOD ported from Python, feature DB restructured
   for cache locality (`dfbe0bd`), benchmarked ~2x faster than authors' own
   ov_SchurVINS (`35db45d`), Python port deleted once C++ was solid (`cedf7f2`).
2. **Full 10-sequence matrix run** (`1ec3389`) exposed the real state: DOD won
   6/8 EuRoC sequences but **diverged on MH_02** and was **12–22x worse on
   KAIST**. Not robust yet.
3. **Root-cause hunt, not guesswork** — per CLAUDE.md's "measure, never guess"
   rule, several plausible explanations were tested and killed:
   - circle.bag 39m divergence → `enable_slam=true` bug in KAIST runner, not
     an algorithm defect (60x fix, one flag).
   - SLAM promotion accept-rate — wrong hypothesis, gap was frontend not math.
   - Track lifetime — DOD tracks live *longer* than official's; "die young"
     theory retracted.
   - **Resolved**: IMU coverage was the actual EuRoC gap. DOD → 0.1301 m vs
     official 0.1333 m once fixed. DOD then **beat official on both accuracy
     and speed**: 0.1132 vs 0.1180 m, 6.77 vs 7.15 ms.
4. **Initialisation identified as the last lever** (`76acf5c`) — same-transport
   re-eval: ATE parity is 5-5 across sequences, DOD's drift rate already beats
   official's, so the entire remaining accuracy gap is in *when/how* the
   filter initialises, not the filter itself. KAIST circle inits at 25s on a
   jerk gate; tested initialising while still — 24% *worse*, measured and
   reverted (`init_latency_is_not_the_accuracy_lever`).
5. **Hardening pass** (`ae2128f`, `f4b10ac`, `486129a`, `b6e9aa2`) — silent
   drop paths made loud (with counters, per repo convention), a real
   `dynamic_initialize` gravity-convergence bug found on aarch64 and fixed,
   and the repo aligned for **JOSS submission** (license, CI, docs).
6. **Current recorded baseline** (the bit-identical accuracy gate,
   `tools/sweep.sh`): MH_01 0.1131, MH_02 0.1744, MH_03 0.2223, MH_04 0.4580,
   MH_05 0.3074, V1_01 0.0494, V1_02 0.0551, V1_03 0.0560, circle 0.0374,
   infinity 0.0261. Speed: MH_01 18.656s vs official 29.014s (1.56x); KAIST
   circle 21.321s vs 27.952s (1.31x). DHAT allocation baseline: 825,790
   blocks / 2.02 GB over 12s of MH_01.

So: **the JOSS-grade software paper is essentially done and merged to
`main`.** The open frontier — and the ICRA angle — is a specific unsolved
weakness: performance at **mid/high altitude**, where stereo baseline becomes
small relative to depth and standard isotropic pixel-noise MSCKF updates
under-weight the resulting depth uncertainty.

## 2. This branch: altitude-aware stereo covariance

Uncommitted work in progress on `research/altitude-covariance`
(`first do 1 and do it in a new branch`):

- `core/feature.{hpp,cpp}` — new `parallax_noise_scale()`: computes a
  variance multiplier `1 + (lambda * Z/B)^2` (clamped to `max_scale`) from
  depth-to-baseline ratio `ρ = Z/B`, where `B` is the widest translational
  baseline among clones that observed the feature. **Not** an altitude
  threshold — CLAUDE.md's rule that tuned constants don't transfer datasets
  ruled that out; the ratio is geometry, dataset-independent.
- `msckf/updaters.{hpp,cpp}` — wired into all three update paths
  (`update_msckf`, `delayed_init_slam`, `update_slam`) via measurement
  *whitening* (divide `H_x`/`H_f`/`res` by `sqrt(scale)`), because the
  existing Givens-QR compression (`measurement_compress_inplace`) requires
  isotropic R — a non-diagonal R wasn't an option without touching that step.
  New `UpdaterOptions::parallax_noise_lambda` defaults to `0.0` → guaranteed
  bit-exact no-op for every existing runner/test.
- `tools/mid_altitude_options.hpp` — `VIO_PARALLAX_LAMBDA` env knob, only
  path that ever sets the new field non-zero.
- `tests/verify_parallax_noise.cpp` — new property test (5 properties),
  registered in `CMakeLists.txt`.
- Also pre-existing on this branch (uncommitted before this work started):
  `ros/vio_rosbag_runner_mid_altitude.cpp`, `scripts/convert_fgi_groundtruth.py`
  — a mid-altitude runner + FGI dataset groundtruth converter, i.e. the
  evaluation harness this feature needs.

**Status: implemented, not yet verified.** Last action was a build attempt
that hit exit 137 (killed — resource limit, not diagnosed) on
`cmake --build build --target verify_parallax_noise -j8`. Nothing has run yet:
no unit test, no `ctest`, no accuracy-gate sweep.

## 3. Immediate tasks (finish what's on this branch)

- [ ] **Fix the build.** Retry `-j2`/`-j1`, capture full output (not `tail`-
      truncated) to see whether exit 137 was OOM or masked a compile error.
- [ ] Run `./build/verify_parallax_noise`, confirm all 5 properties pass.
- [ ] Run full `ctest` (expect 6/6 with the new test added).
- [ ] Run the 10-sequence EuRoC/KAIST sweep (`tools/sweep.sh`) and diff
      against the recorded ATEs above — **must be bit-identical**, since
      `parallax_noise_lambda` defaults to 0 everywhere except the mid-altitude
      runner. This is the accuracy gate; nothing lands without it.
- [ ] Only after the gate passes: commit. Not done yet, no commit made this
      session.

## 4. Not yet started / not yet authorized by the user

Implied by earlier planning context but **out of scope until asked**:

- Sweeping `VIO_PARALLAX_LAMBDA` (e.g. 0/40/60/80/100) against the FGI
  mid-altitude dataset to see if the model actually helps at altitude.
- Wiring the mid-altitude runner + FGI groundtruth into the standard sweep
  script.
- Any paper writing (`paper/main.tex` shows local edits already, unreviewed
  in this session).
- Benchmark-methodology freeze for the ICRA submission.

## 5. Path to ICRA 2027 (proposed shape, needs confirmation)

The JOSS paper covers "we built a faster/competitive MSCKF." The ICRA paper
needs its own contribution — natural candidate given the current branch:
**altitude-robust stereo VIO**, i.e. characterize where standard MSCKF
degrades as depth/baseline grows, fix it with the geometric covariance model,
and show it on a dataset that actually reaches those altitudes (FGI mid-
altitude data > EuRoC's flat indoor baseline).

Rough milestones (dates TBD — check ICRA 2027 CFP once posted, typically
paper deadline ~Sept 2026):

1. Finish + merge altitude-covariance branch (tasks in §3).
2. Get mid-altitude runner + FGI groundtruth into the standard sweep;
   establish an altitude-stratified accuracy baseline (current model, λ=0).
3. Sweep λ, find whether/where it helps; if it doesn't, that's a result too —
   CLAUDE.md's culture here is "measure, report the finding," not "make the
   idea work."
4. Decide the paper's actual claim from what the sweep shows (this order
   matters — don't write the abstract before the data exists).
5. Paper draft + submission.

This section is a draft, not a commitment — confirm scope/dates before
treating it as the plan.
