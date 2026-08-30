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

Newest at the top within each section. Last updated: 2026-08-30 (M-20, T-004).

---

## A. Process mistakes (the expensive ones)

### M-20 — Fixing the constant that was derivable, not the one that was dominant

- **Believed:** `max_dist = 75.0` was the cause of 60_4.bag's divergence. The
  evidence was good — a clean FOV derivation, and `reject_maxdist = 43579` on
  60_4 against 0 on 40_4, exactly the sequence boundary between working and
  broken.
- **True:** raising it to 200 does not make 60_4 converge. It changes the
  failure from a 461 km drift into a negative-covariance abort at ~frame 800.
  And the counters that were never printed side by side show why: on 80_4,
  `reject_cond = 64947` against `reject_maxdist = 13744`. The distance cap was
  never the dominant gate at altitude; triangulation *conditioning* is, by
  4.7x. The filter reaches zero SLAM features and diverges for lack of
  measurements, not for lack of range.
- **Cost:** none wasted — the `max_dist` fix is correct within its scope and
  had to happen. The cost was in the plan: T-005 was scoped around
  initialisation for four days when the larger lever was a triangulation
  constant nobody had ranked.
- **Rule:** **rank the reject counters before choosing which one to fix.** One
  counter that is large and has a satisfying explanation is not evidence that
  it is the largest. This is the DHAT lesson from CLAUDE.md in a new place:
  ranking by blocks and by bytes pointed at different defects, and ranking
  rejections by cause does the same.
- **Second rule:** a constant that can be *derived* is attractive to fix
  because the fix feels principled. That is a bias toward the tractable, not
  toward the dominant. `max_cond_number = 10000` has no derivation available,
  which is probably why it went unexamined.
- **What the derivation did earn:** at 40 m, md75 and md200 produce
  byte-identical output, exactly as the slant-range argument predicted. The
  derivation was right about its scope. It was the diagnosis of *the* cause
  that overreached.
- **Reaches forward to:** T-005 (reshaped around this), T-006, T-008.

### M-19 — A tidy explanation, accepted one check too early

- **Believed:** the EuRoC baseline drift was an OpenCV version difference. Host
  runs 4.5.4, container runs 4.2.0, the KLT frontend is OpenCV's, and the two
  sequences that matched the recorded table exactly had been run in the
  container. Every observation fit. I wrote it into CLAUDE.md, the board and a
  commit message as the leading hypothesis.
- **True:** building the ASL runner in the container returned 0.1293 for MH_01,
  against the host's 0.1282 — a 1% difference, not the 13% the theory needed.
  The real answer was one commit away in the project's own history: `1ec3389`
  had re-recorded MH_01 at 0.1293 the day after the table was written. And the
  table was worse than stale — `a7addf6`, the commit that introduced it,
  returns 0.1496. The figure never reproduced anywhere.
- **Cost:** an hour, and three documents briefly asserting something false.
- **Rule:** the danger sign is a hypothesis that explains *everything*
  available. That is usually a sign the available evidence is thin, not that
  the theory is strong. Fitting all the data is the beginning of a test, not
  the end of one.
- **Second rule:** search the project's own history before theorising about the
  environment. `git log -S` on the disputed number found the answer in seconds
  and would have cost nothing had it come first.
- **Third rule:** when a hypothesis reaches a written document, it carries a
  label saying it is a hypothesis, and whoever wrote it owns going back to
  correct it. Done here: CLAUDE.md now states the measured position.
- **Partly right, and worth keeping:** environment really does affect the
  result — host and container disagree by 23% on MH_02 and 8% on V1_01 for
  identical code. That justified pinning the gate to an environment, which
  stands. It just was not the cause of this discrepancy.
- **Reaches forward to:** T-004 especially, where a single diverging sequence
  will invite exactly this kind of tidy story.

### M-18 — Trusting a recorded baseline as the oracle

- **Believed:** the ten ATEs in CLAUDE.md are the accuracy gate. Run the sweep,
  compare against them, done.
- **True:** eight of the ten do not reproduce on the lab host — MH_01 comes back
  0.1282 against a recorded 0.1131. Not from this branch: the pre-branch control
  tree gives byte-identical output, and the lab's own 2026-08-26 binary gives
  0.1328. The recorded EuRoC numbers appear to have been taken in the container
  (OpenCV 4.2.0 / g++ 9.4.0), while the host is OpenCV 4.5.4 / g++ 11.4.0, and
  the KLT frontend is OpenCV's. The two sequences that were run in the container
  matched to four decimals.
- **Cost:** none to the conclusion, because a control tree was run alongside.
  Without it, the honest reading of "MH_01 moved 13%" would have been "the
  parallax commit broke the filter", and the next hours would have gone into
  debugging a commit that provably changes nothing.
- **Rule:** **a recorded number is not a control; a control is a control.** When
  the question is "did my change move this", run the unchanged tree in the same
  environment on the same data and diff the outputs. A stored constant silently
  encodes an environment, and environments drift while the constant does not.
- **Second rule:** compare artefacts, not summaries. `cmp` on the estimate CSVs
  answers "did anything change" exactly; ATE to four decimals can hide a change
  and, worse, can invent one.
- **Third rule:** a gate needs its environment pinned next to its numbers.
  Compiler and OpenCV version belong in the gate definition, because the
  frontend is a third-party library and its output is part of the result.
- **Reaches forward to:** T-004, T-006, T-008 — all compare against baselines,
  and T-013 exists to pin this one down.

### M-17 — Announcing a pass from a test that tested nothing

- **Believed:** `benchmark_full` is a synthetic, data-free run of the whole VIO
  pipeline, so diffing its output across the change is a local bit-exactness
  oracle. Ran it both sides, filtered timing lines, got "identical", said so.
- **True:** it prints three lines and two of them are timings. The filter left
  a single line — the banner. The comparison was vacuous, and it was reported
  as a pass before anyone looked at what was being compared.
- **Cost:** small, caught within a minute. The cost if it had not been caught
  is the entire point: a fabricated green on the one property the commit most
  needed evidence for.
- **Rule:** before believing a diff is empty, look at what was in it. A
  comparison that survives filtering with almost nothing left has not passed,
  it has been emptied. Print the line count of what is actually being compared.
- **Second rule:** an oracle has to be *shown* to discriminate. If the check
  cannot fail, it cannot pass. Verify the harness detects a deliberate change
  before trusting it to detect an accidental one.
- **Reaches forward to:** T-003, T-004, T-006 — every remaining ticket is
  settled by comparing outputs, and all of them can be faked this way.

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
- Only 4 of the dataset's 12 bags are on moonlab: 40/60/80/100 m, all at
  4 m/s. The altitude axis is testable; the speed axis is not, and no claim
  about speed can be made from what is downloaded.
- 60_4's md200 abort is a covariance blow-up from admitting deep,
  badly-conditioned triangulations at full confidence — the exact case
  `parallax_noise_scale` was written for. Recorded as T-006's prediction so it
  can be falsified rather than confirmed after the fact.
- `parallax_noise_max=100.0` is an unjustified cap. If T-006 ever hits the
  clamp, the cap is doing the tuning and the result is about the cap, not the
  model. Log clamp hits before reading that sweep.
- `update_msckf_schur()` does not whiten. `use_schur_msckf` is false
  everywhere, so no live gap, but a lambda sweep run against the schur path
  would return a null result that looks like evidence. Re-check the flag before
  T-006.
- ~~Bit-exactness of `15b0860` rests on an argument, not a measurement.~~
  **Measured 2026-08-30 (T-003): 10/10 sequences byte-identical against the
  pre-branch control tree.** The argument now has evidence behind it.
- What configuration produced the withdrawn 0.1131 table is unknown. Closed as
  not worth chasing (T-013); the control-tree method does not depend on it.
