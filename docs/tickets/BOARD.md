# Ticket board

One file, one board. A ticket gets its own file (`docs/tickets/T-0NN.md`) only
when its working log outgrows a few lines here — most will not.

Status: `todo` / `doing` / `blocked` / `done` / `killed`.
**One ticket in `doing` at a time.** That is the point of this board.

Rules of engagement (from `../mistake-learning.md`, which is the other half of
this system):

1. Read `../mistake-learning.md` before starting a ticket. Every entry there
   cost real hours; most of them are still reachable from here.
2. A ticket states its **exit condition as a number or a diff**, never as
   "works". If you cannot write down what result closes it, the ticket is not
   ready to start.
3. A hypothesis that gets measured and loses is a **result**. Record it in the
   ticket, add it to `mistake-learning.md` if it teaches something, revert the
   code. Do not retry it later from memory.
4. Nothing lands without the accuracy gate (T-003) green.

Last updated: 2026-08-30 (T-001, T-002, T-003, T-013 closed).

---

## Now

| ID | Title | Status | Blocks |
|----|-------|--------|--------|
| T-001 | Commit the FGI harness + diagnostics | **done** 862b3b8 | — |
| T-002 | Commit the whitening wiring (lambda=0) | **done** 15b0860 | — |
| T-003 | Accuracy gate: 10-sequence sweep on moonlab | **done** PASS | — |
| T-004 | Does 60_4 converge with max_dist=200? | **next** | T-005, T-006, T-008 |

## Next

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| T-005 | Sweep VIO_INIT_MIN_REC_COND on 60_4 | todo | T-004 |
| T-007 | FGI into tools/sweep.sh | todo | T-004 |
| T-008 | Altitude-stratified baseline, lambda=0, 12 bags | todo | T-004, T-007 |
| T-006 | Sweep VIO_PARALLAX_LAMBDA over the envelope | todo | T-002, T-008 |

## Later

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| T-009 | Decide the ICRA claim from the data | todo | T-006 |
| T-010 | ICRA paper draft | todo | T-009 |
| T-011 | Refresh docs/icra2027_plan.md | todo | — |
| T-012 | Land the JOSS author-name fix | todo | — |

---

## T-001 — Commit the FGI harness + diagnostics

**Status:** done — `862b3b8`, 2026-08-30.

**Split boundary moved during the ticket, for the better.** The plan was to put
`parallax_noise_scale()` in T-002 with the wiring, which would have meant
splitting three files at line granularity. It does not need splitting: a
function with no callers cannot move a trajectory. So T-001 took the whole
function, its unit test, and its counters, and T-002 keeps only what actually
touches results. The split now lands on hunk boundaries — one `git apply
--cached` of a single hunk in `updaters.cpp` — and T-002 shrank to **50 lines
in 2 files**, which is the smallest possible thing to revert if the gate moves.
See M-16.

**Landed:**
- `tools/mid_altitude_options.hpp`, `ros/vio_rosbag_runner_mid_altitude.cpp`,
  `scripts/convert_fgi_groundtruth.py` — FGI harness, `max_dist` 75 -> 200
  derived from `h/cos(41.67 deg)`, FGI-runner-only.
- `tri_depth_hist` / `tri_depth_sum`, `epipolar_consistency_score` + `epi_*`
  counters, the shutdown dumps in four runners.
- `core::parallax_noise_scale()` + `tests/verify_parallax_noise.cpp`, no callers.
- `.gitignore`: `/data/`, `paper/.agents/`, the 16 MB vendored PDF.

**Write-only check (the ticket's stated risk) — passed.** Grep of every
diagnostic symbol: the only reads are shutdown `fprintf`s in the four runners
and the unit test. `updaters.cpp` casts the epipolar score to `(void)`. No
branch anywhere reads one.

**Verified:** `git stash push --keep-index` of the T-002 remainder to force
worktree == index, then a full build and `ctest` **6/6 on the T-001 tree
alone**. This is what proves the split is real rather than assumed.

**Not verified:** T-003. No EuRoC/KAIST bags on this machine. Committed anyway
on a research branch, deliberately — the alternative was leaving four days of
measurement in a dirty tree (M-02), and nothing in this commit is reachable
from a non-FGI runner. The gate still governs before this reaches `main`.

---

## T-002 — Commit the whitening wiring

**Status:** done — `15b0860`, 2026-08-30. 62 lines, 3 files.

**Landed:** the two `UpdaterOptions` fields and the whitening in
`update_msckf`, `delayed_init_slam`, `update_slam`.

**Two defects found while landing it, both fixed in the same commit:**
- `update_slam` built its `ClonesCamera` unconditionally — ~4 KB of pose
  writes per update on every EuRoC/KAIST run, for an object
  `parallax_noise_scale` returns before touching at lambda = 0. Now gated.
  Declared without an initialiser, because `ClonesCamera{}` would zero the
  same 4 KB the branch avoids writing.
- `update_msckf_schur()` has no whitening. `use_schur_msckf` is false
  everywhere today, but flipping it would silently disable the model in the
  MSCKF path. Note left at the flag's declaration. **Re-check before T-006** —
  a lambda sweep against the schur path would measure nothing and look like a
  null result.
- Also: `<cmath>` was arriving transitively via Eigen.

**Verified locally:** build clean, `ctest` 6/6. Only writer of
`parallax_noise_lambda` in the tree is the FGI runner under
`VIO_PARALLAX_LAMBDA` (grepped).

**Not verified:** bit-exactness. There is no local oracle — see M-17. T-003
settles it.

---

## T-003 — Accuracy gate

**Status:** done, 2026-08-30. **PASS — 10/10 sequences byte-identical.**

**Method, and why it is stronger than the gate as written.** The gate compares
ATEs against a recorded table. That only works if the table reproduces, which
it turned out not to (below). So instead of trusting the table, both trees were
run side by side: `862b3b8..15b0860` (head) and `b6e9aa2` (base, the pre-branch
tip) shipped to moonlab with `git archive`, built with identical flags, run on
identical data, and the **estimate CSVs compared byte-for-byte**.

| | |
|---|---|
| EuRoC, 8 seq, host ASL path | head == base, byte-identical, 8/8 |
| KAIST circle + infinity, container ROS path | head == base, byte-identical, 2/2 |
| Oracle discriminates? | yes — 8 distinct md5s across the 8 sequences |

That last row is the M-17 check: a comparison that cannot tell two things apart
proves nothing. Distinct hashes per sequence show the comparison has resolution.

**Conclusion: `15b0860` is a true no-op at lambda = 0**, measured rather than
argued. The IEEE-754 reasoning in the commit message is now backed by output.

**Environments used (they are not interchangeable — see T-013):**
- Host `stork`: OpenCV 4.5.4, g++ 11.4.0, `dod_asl_runner`, ASL directories.
- `ros_container_v2`: OpenCV 4.2.0, g++ 9.4.0, `vio_rosbag_runner`, bags.

Reproduce: trees at `/media/storage/moonlab/vio_parity/acv/{head,base}`,
outputs in `acv/runs/`; container copies at `/workspace/acv/`.

---

## T-013 — Why the recorded gate table does not reproduce

**Status:** done, 2026-08-30. **My hypothesis was wrong, and the answer is
worse than it: the old table never reproduced at all.**

**The exit condition was "build the ASL runner in the container and run MH_01;
0.1131 confirms the OpenCV hypothesis."** It returned **0.1293** — within 1% of
the host's 0.1282. The environment was not the explanation.

**What is true instead:**
1. `1ec3389` (2026-08-17) re-evaluated all ten sequences and recorded MH_01 at
   0.1293 — exactly today's container number. The CLAUDE.md table was written
   2026-08-16 in `a7addf6` and superseded the next day, never updated.
2. Built `a7addf6` itself and ran it: **0.1496**. So 0.1131 did not come from
   that commit either. It is not stale-but-once-true; it is unreproducible at
   every commit tested, on both environments, on both the official ASL zip and
   the bag conversion.
3. Built `1ec3389` and compared to HEAD across all eight EuRoC sequences:
   **byte-identical, 8/8**. Nothing between 2026-08-17 and this branch — the
   silent-drop fixes, the aarch64 gravity fix, the JOSS pass, the parallax
   work — has moved a EuRoC trajectory by one bit.

**Environment does matter, just not enough to explain 0.1131.** Host vs
container on identical code and data agree within ~1% on six sequences but
differ by 23% on MH_02 and 8% on V1_01. The KLT frontend is OpenCV's
(4.5.4 host, 4.2.0 container). An ATE quoted without its environment cannot be
checked.

**Delivered:** CLAUDE.md's gate now carries a measured, environment-pinned
table (container, 2026-08-30) plus the host/container comparison, and states
that the old figures were withdrawn rather than chased.

| | MH_01 | MH_02 | MH_03 | MH_04 | MH_05 | V1_01 | V1_02 | V1_03 | circle | infinity |
|---|---|---|---|---|---|---|---|---|---|---|
| new baseline | 0.1293 | 0.2080 | 0.2343 | 0.4267 | 0.3285 | 0.0545 | 0.0482 | 0.0550 | 0.0374 | 0.0261 |
| withdrawn | 0.1131 | 0.1744 | 0.2223 | 0.4580 | 0.3074 | 0.0494 | 0.0551 | 0.0560 | 0.0374 | 0.0261 |

KAIST was correct all along and is unchanged.

**Left open deliberately:** what configuration produced 0.1131 is unknown and
is not worth further archaeology. The control-tree method (T-003) does not
depend on the answer.

---

## T-004 — Does 60_4 converge with max_dist=200?

**Status:** todo. **This is the gating experiment for the entire ICRA line.**

60_4.bag diverges from t=0. Two independent causes are on record:
1. `max_dist=75.0` (EuRoC indoor constant) sits below the 80.3 m slant range a
   nadir camera sees at 60 m altitude -> `reject_maxdist=43579`. Already
   derived from FOV geometry and fixed to 200.0 in the runner. **Untested.**
2. The featureless `[v,g]` solve is badly conditioned: `rec_cond 0.0034` vs
   40_4's 6.9e-05, gravity-vs-accel 6.1 deg vs 0.94. Levers are wired but
   unswept (that is T-005).

**Exit condition:** one run of 60_4.bag with the committed runner. Record ATE,
`reject_maxdist`, `rec_cond`, and the `tri_depth_hist` buckets. Two outcomes,
both useful: it converges (cause 1 was the whole story, T-005 may be
unnecessary) or it still diverges (cause 2 is real and T-005 becomes the
critical path).

**Do not start T-006 before this closes.** lambda is untestable while the
envelope diverges for a reason unrelated to lambda.

---

## T-005 — Sweep VIO_INIT_MIN_REC_COND on 60_4

**Status:** todo. Blocked on T-004.

Knobs already wired: `VIO_INIT_MIN_REC_COND`, `VIO_INIT_WINDOW`,
`VIO_INIT_NUM_POSE`. Measured `rec_cond` vs window on 60_4: 0.0034 (3 frames),
0.0068 (4 s), 0.065 (6 s). EuRoC's healthy solves sit at 1.6e-4 .. 2.3e-4, so a
threshold between the regimes should reject the bad early solves and let the
pre-init database grow.

**Exit condition:** a table of `min_rec_cond` x ATE on 60_4 and 40_4, plus
confirmation that EuRoC stays bit-identical (default 0 = off).

**Prior art, do not repeat:** `init_imu_thresh` 1.5 -> 0.5 was tested and is
byte-identical here (the static initializer never fires on this dataset).
Initialising earlier was tested on KAIST and was 24% *worse*.

---

## T-006 — Sweep VIO_PARALLAX_LAMBDA

**Status:** todo. Blocked on T-002 and T-008.

`VIO_PARALLAX_LAMBDA` in 0/40/60/80/100 across the 40/60/80/100 m bags. The
model is `sigma_eff^2 = sigma_pix^2 * (1 + (lambda*Z/B)^2)`, capped at
`parallax_noise_max`.

**Exit condition:** ATE vs lambda, per altitude, against the T-008 lambda=0
baseline. **"It does not help" closes this ticket successfully** — the
prediction under test is that the benefit grows with altitude, and a flat curve
falsifies it. Record either way, do not tune until it wins.

---

## T-007 — FGI into tools/sweep.sh

**Status:** todo. Depends on T-004.

Mid-altitude runner + `scripts/convert_fgi_groundtruth.py` wired into the
standard sweep so altitude runs are one command, like EuRoC.

**Exit condition:** one invocation produces ATEs for all 12 FGI bags; the ten
EuRoC/KAIST numbers are untouched by the change.

---

## T-008 — Altitude-stratified baseline (lambda=0)

**Status:** todo. Depends on T-004, T-007.

All 12 bags (40/60/80/100 m x 2/3/4 m/s), current model, lambda=0. This is the
number T-006 is measured against, and the figure the paper needs regardless of
what lambda does.

**Exit condition:** a committed table of 12 ATEs + divergence flags, and a
plot of error vs altitude.

---

## T-009 — Decide the ICRA claim

**Status:** todo. Depends on T-006.

Write the claim from the data, not before it. Candidates, in the order the
evidence would support them:
- lambda helps and the gain scales with altitude -> "altitude-robust stereo VIO."
- lambda is flat but the max_dist/rec_cond fixes carry the envelope -> the
  contribution is the characterisation plus two geometric constants, which is a
  smaller but honest paper.
- Nothing carries 80/100 m -> report where MSCKF breaks and why. Still a paper.

**Exit condition:** one paragraph, the abstract's claim, with the figure
numbers that back each sentence named next to it.

---

## T-010 — ICRA paper draft

**Status:** todo. Depends on T-009. Check the ICRA 2027 CFP for the real
deadline (historically ~Sept).

---

## T-011 — Refresh docs/icra2027_plan.md

**Status:** todo. Cheap.

The plan doc (2026-08-26) says the branch is "implemented, not yet verified"
and that the build is broken. Both are stale as of 2026-08-30: build was an OOM,
tests pass 6/6, and a week of FGI measurement it does not mention is recorded
in `tools/mid_altitude_options.hpp` comments. Point it at this board rather
than duplicating the task list.

---

## T-012 — Land the JOSS author-name fix

**Status:** todo. One character: `Vashista` -> `Vashisht` in `paper/main.tex`,
plus the rebuilt `main.pdf`. Trivial, but it is a co-author's name on a
submission, so it should not ride along inside an unrelated commit.
