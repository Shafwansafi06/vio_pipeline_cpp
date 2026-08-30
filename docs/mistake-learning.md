# Mistake -> learning

Every entry here cost real time. The point of writing them down is not
confession, it is **routing**: before starting a ticket, skim the "Reaches
forward to" lines and see which of these is about to happen again.

Format:

- **Believed** — what was asserted before measuring.
- **True** — what the measurement said.
- **Cost** — time, or the wrong turn it caused.
- **Rule** — the generalisation, if there is one. Not every mistake has one.
- **Reaches forward to** — which open tickets this can still bite.

Newest at the top within each section. Last updated: 2026-08-30 (M-16, T-001).

---

## A. Process mistakes (the expensive ones)

### M-16 — Splitting a commit at the wrong boundary

- **Believed:** the parallax model and the diagnostics are entangled across
  `core/feature.{hpp,cpp}`, `msckf/updaters.cpp` and `CMakeLists.txt`, so
  separating them needs line-level patch surgery inside three shared hunks.
- **True:** they are not entangled where it matters. `parallax_noise_scale()`
  with no callers cannot change a trajectory, so it belongs on the *safe* side
  of the split. Moving the boundary to "does this have a caller" put it exactly
  on hunk boundaries: one `git apply --cached` of a single hunk, and the risky
  commit shrank from three mixed files to 50 lines in two.
- **Cost:** none — caught before the surgery. Would have been an hour of
  hand-built patches and a real chance of a mis-split commit.
- **Rule:** split a commit by **blast radius, not by topic**. Ask "which lines
  can change a result?" rather than "which lines belong to this feature". Dead
  code is free to ride along with the safe half, and doing so usually makes the
  cut fall on boundaries the tooling already understands.
- **Corollary worth keeping:** `git stash push --keep-index` forces worktree ==
  index, so the split half can be built and tested *as it will exist after the
  commit*. A split that is not built in isolation is only a claim. T-001 used
  it; ctest was 6/6 on the T-001 tree alone.
- **Reaches forward to:** T-002, and every commit split after it.

### M-01 — Reading a resource limit as a code failure

- **Believed:** `cmake --build build --target verify_parallax_noise -j8`
  exited 137, so the branch does not build. Recorded in `icra2027_plan.md` as
  "Fix the build" — the top open task.
- **True:** 137 is SIGKILL. The box has 7 GB of RAM and 12 cores; `-j8` over
  Eigen-heavy translation units OOMs. `-j2` builds every target clean, zero
  errors, and the test passed on its first run. There was never a build defect.
- **Cost:** the branch's real state was unknown for four days, and the plan doc
  put a phantom blocker at the head of the queue.
- **Rule:** an exit code from the *process supervisor* (137 SIGKILL, 143
  SIGTERM, 139 SIGSEGV) is not a compiler diagnostic. Read the code before
  reading the log tail — and never diagnose from `tail`, which is exactly what
  hid this.
- **Reaches forward to:** every build on this laptop. Use `-j2`.

### M-02 — Working everything at once

- **Believed:** the altitude branch is one piece of work, so it can be one
  session.
- **True:** by 2026-08-30 the branch held four separable things — a new noise
  model, two diagnostics, an FGI harness, and a derived `max_dist` fix — none
  committed, some verified and some not, with the verified/unverified boundary
  recorded nowhere. State lived only in the session.
- **Cost:** four days of measured findings one bad checkout from gone, and no
  way to revert one idea without reverting the others.
- **Rule:** if two changes could be reverted independently, they are two
  commits and two tickets. Split at the point of independent revert, not at the
  point of "feels done".
- **Reaches forward to:** T-001, T-002 (the split), and everything after.

### M-03 — Findings recorded only in code comments

- **Believed:** documenting a killed experiment in the comment next to the
  constant is enough.
- **True:** `tools/mid_altitude_options.hpp` is the *only* record that
  timeoffset calibration, `dt_slam_delay=0.3`, `max_cond_number=3000`, and
  `init_imu_thresh=0.5` were each tested and lost, with numbers. The plan doc
  does not mention any of them, so a plan-doc reader would re-run all four.
- **Cost:** none yet. Would have been four duplicated experiments.
- **Rule:** the comment next to the constant is the right place for *why the
  constant is what it is*. It is the wrong place for *what has already been
  tried*, because nobody greps a header before planning. Killed hypotheses go
  in the ticket and here.
- **Reaches forward to:** T-005, T-006.

### M-04 — Letting the plan doc go stale

- **Believed:** `docs/icra2027_plan.md` describes the current state.
- **True:** four days out of date, wrong about the build, wrong about what is
  verified, silent on a week of FGI measurement.
- **Rule:** one document owns the queue. Everything else links to it. This
  board is now that document; the plan doc becomes narrative history (T-011).

---

## B. Technical hypotheses that were measured and lost

These are the entries that justify the CLAUDE.md rule. Every one was argued
confidently, and every one was settled by a profiler or a sweep in minutes.

### M-05 — "The back-end is slower because we feed ~2x the features"

- **True:** capping at official's 40 features moved the MSCKF stage by 3%
  (1.581 vs 1.538 ms) and cost accuracy on 6 of 8 sequences.

### M-06 — "Our `M_a` loop is structured differently from official's"

- **True:** official has the identical nested loop. Restructuring it was still
  a 2.2x win — for a reason the guess did not identify. **A fix that works for
  a reason you did not predict has not confirmed your model.**

### M-07 — "Building `S` block-wise avoids the gather, so it must be faster"

- **True:** slower, 0.285 -> 0.404 ms. It trades two dense products for ~169
  small ones per feature.

### M-08 — "SLAM promotion accept-rate explains the accuracy gap"

- **True:** wrong; the gap was in the frontend, not the math.

### M-09 — "Our tracks die young"

- **True:** DOD's tracks live *longer* than official's. Retracted after
  measurement.

### M-10 — "Initialising earlier will fix KAIST"

- **True:** KAIST circle inits at 25 s on a jerk gate. Initialising while still
  is **24% worse**. Measured, reverted.

### M-11 — "The EuRoC gap is the filter"

- **True:** it was IMU coverage. Once fixed, 0.1301 vs official 0.1333, then
  0.1132 vs 0.1180. ATE parity is now 5-5 and DOD's drift rate already beats
  official's — the residue is initialisation, not the estimator.

### M-12 — "circle.bag diverges by 39 m, so the algorithm is broken on KAIST"

- **True:** `enable_slam=true` in the KAIST runner. One flag, 60x.
- **Rule:** before blaming the algorithm, diff the *configuration* against the
  one that works. Cheaper than any profile.

**Rule for all of B:** profile or sweep first. On this codebase, confident
reasoning about where the time or the error goes has been wrong more often than
right. Three of the above were settled in minutes after being argued at length.

**Reaches forward to:** T-004, T-005, T-006 — all three are sweeps whose
outcomes are currently being predicted in conversation.

---

## C. Traps with teeth (these break the filter, not just the schedule)

### M-13 — Collapsing the covariance update

`Cov -= K * M_a.transpose()` is symmetric in exact arithmetic and **not** in
floating point. The asymmetry drives covariance diagonals negative and
`EKFPropagation` aborts on MH_01 within ~16 s. Do not "simplify" it.

### M-14 — Assuming a tuned constant transfers between datasets

`max_msckf_in_update` is 75 on EuRoC and 50 on KAIST. `init_imu_thresh` at 0.25
fixes MH_01 and breaks V1_01 outright. `max_dist=75.0` was tuned for <5 m
indoor depth and silently discards every valid ground feature above ~55 m
altitude — the single constant behind 60_4.bag's divergence.

- **Rule:** any constant inherited from another dataset is wrong until
  measured. Mark it PLACEHOLDER in the config the moment it is copied.
- **Rule (stronger, learned from `max_dist`):** prefer a constant **derived
  from geometry** over one tuned on data. `max_dist=200` came from
  `h/cos(41.67 deg)` over the dataset's altitude envelope, so it needs no
  re-tuning per altitude. `parallax_noise_scale` uses the ratio `Z/B` for the
  same reason — a ratio is dataset-independent in a way a threshold is not.

### M-15 — Silent drops

Paths that discarded measurements without counting them hid real defects, one
of which could kill the filter (fixed in `ae2128f`). KAIST saturates
`FEATURE_MAX_MEASUREMENTS`; cutting it to 24 would be a bug, and only the
counters show that.

- **Rule:** every reject path gets a counter. A diagnostic that no decision
  reads is cheap; a drop nobody counts is invisible.

---

## D. Open self-checks

Things suspected but not yet measured. Promote to a ticket or delete — do not
let them rot here.

- ~~The new `tri_depth_hist` / `epi_*` diagnostics are asserted write-only.~~
  **Checked 2026-08-30 (T-001), passed.** Only reads are the four runners'
  shutdown `fprintf`s and the unit test; `updaters.cpp` casts the epipolar
  score to `(void)`. Re-run that grep if any of them ever gains a caller.
- `max_msckf_in_update=75` in the FGI config is EuRoC's value, marked
  PLACEHOLDER, unmeasured at 16 Hz stereo and 40-100 m depth. Per M-14 assume
  it is wrong.
- `parallax_noise_max=100.0` is an unjustified cap. If T-006 ever hits the
  clamp, the cap is doing the tuning and the result is about the cap, not the
  model. Log clamp hits before reading that sweep.
