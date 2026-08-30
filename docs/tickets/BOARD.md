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

Last updated: 2026-08-30. Altitude line parked after T-006; the speed paper is the active work.

---

## Now — the speed paper (ICRA)

**Spine chosen: systems paper.** "The fastest and most predictable MSCKF
implementation, validated on the hardware where layout actually matters."
Rationale and what it costs us in `docs/paper_spine.md`.

| ID | Title | Status | Cost |
|----|-------|--------|------|
| P-01 | Predictability: p99, jitter, latency CDF, RSS | **done — p99 wins, RSS does not** | ~free, data on disk |
| P-05 | Toolchain confound (already measured) | **done — paper Sec. toolchain + Table** | free, T-013 output |
| P-02 | Close the embedded loop on Orin Nano | todo | **gating; needs device** |
| P-03 | DOD-Schur vs ov_SchurVINS head-to-head | **done — row 9 dropped, withdrawal documented** | medium |
| P-04 | Transport confound, full 10x2x2 matrix | todo | medium |
| P-06 | Convert speed headroom into accuracy | todo | tight for 2 weeks |
| P-07 | Artifact: docker + scripts + CSVs | todo | small |

Two-week ordering: P-01 and P-05 first (both nearly free and both directly
support the thesis), then P-02 as soon as the device is reachable, then P-03.
P-04 and P-06 only if those land.

## Parked — altitude line

| ID | Title | Status | Blocks |
|----|-------|--------|--------|
| T-001 | Commit the FGI harness + diagnostics | **done** 862b3b8 | — |
| T-002 | Commit the whitening wiring (lambda=0) | **done** 15b0860 | — |
| T-003 | Accuracy gate: 10-sequence sweep on moonlab | **done** PASS | — |
| T-004 | Does 60_4 converge with max_dist=200? | **done — NO** | — |

## Next

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| T-005 | Triangulation conditioning + init at altitude | **done — no constant fixes it** | — |
| T-007 | FGI into tools/sweep.sh | **parked** | — |
| T-008 | Altitude-stratified baseline | **parked** | — |
| T-006 | Sweep VIO_PARALLAX_LAMBDA over the envelope | **done — model refuted** | — |

## Later

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| T-009 | Decide the altitude claim | **parked — claim refuted by T-006** | — |
| T-010 | Altitude paper draft | **parked** | — |
| T-011 | Refresh docs/icra2027_plan.md | todo | — |
| T-012 | Land the JOSS author-name fix | todo | — |

---

## P-01 — Predictability: p99, jitter, latency CDF, RSS

**Status:** done, 2026-08-30. `scripts/predictability.py`,
`docs/results/predictability/`, `paper/figs/latency_cdf.png`.

**The headline: the tail advantage is larger than the mean advantage.**
Geometric means across all ten sequences:

| statistic | DOD advantage |
|---|---|
| p50 latency | 1.80x |
| **p99 latency** | **2.06x** |
| jitter (p99 - p50) | 2.35x |

Per-frame latency, milliseconds:

| seq | DOD p50 | DOD p99 | DOD max | OV p50 | OV p99 | OV max |
|---|---|---|---|---|---|---|
| MH_01 | 3.40 | 6.20 | 11.32 | 6.66 | 14.76 | 36.96 |
| MH_02 | 3.55 | 6.91 | 8.37 | 6.78 | 14.55 | 21.52 |
| MH_03 | 3.94 | 7.31 | 11.59 | 6.91 | 15.41 | 21.49 |
| MH_04 | 3.58 | 7.29 | 8.70 | 6.52 | 14.66 | 22.69 |
| MH_05 | 3.70 | 7.05 | 27.19 | 6.58 | 15.35 | 24.36 |
| V1_01 | 4.36 | 7.27 | 19.16 | 7.39 | 16.64 | 33.32 |
| V1_02 | 4.29 | 7.87 | 15.01 | 7.22 | 15.54 | 32.04 |
| V1_03 | 3.93 | 7.82 | 17.59 | 6.48 | 14.00 | 20.19 |
| circle | 3.43 | 6.03 | 8.79 | 6.45 | 11.10 | 19.35 |
| infinity | 3.71 | 6.02 | 7.92 | 6.98 | 11.82 | 14.31 |

This belongs in the abstract. A claim of bounded per-frame state predicts
exactly this shape — the advantage should grow toward the tail, and it does,
1.80x to 2.06x to 2.35x. Mean throughput alone could not have shown it.

**Max RSS: DOD is worse in absolute terms and much better in variation.**

| | min | max | spread | stdev |
|---|---|---|---|---|
| DOD | 191.1 MB | 204.0 MB | **6.8%** | 3.38 MB |
| OpenVINS | 107.5 MB | 154.8 MB | 44.0% | 16.35 MB |

DOD costs about 1.4x the peak memory and holds it nearly flat across ten
sequences; OpenVINS is smaller and tracks its workload. This is the
preallocation showing up honestly: `FEATURE_DB_CAPACITY = 2048` and the
fixed-size measurement arrays are a ceiling paid up front, not a footprint that
grows.

**Report it as a trade, not a win.** For a fixed-RAM embedded target a flat
200 MB can be preferable to a variable 110-155 MB, because the system is sized
for the worst case regardless — but the absolute number is higher and the paper
should say so plainly. Whether the ceiling can be lowered (a smaller
`FEATURE_DB_CAPACITY`) without touching accuracy is a separate, cheap question.

**Method notes.** Timing came entirely from `*_timing.csv` files that already
existed; no new estimator runs were needed for the latency half. OpenVINS
records seconds and DOD milliseconds — the reducer normalises. RSS needed
fresh runs: `/usr/bin/time` is not installed in the container, so
`tools/p01/withrss.py` (wait4 + `ru_maxrss`) stands in.

**One harness, not two.** This ticket briefly had two reducers — a numpy one
under `scripts/` and the stdlib `tools/p01/` set. They were cross-checked on
identical inputs and agree (p50 1.80x, p99 2.06x, jitter 2.35x from both;
the percentile method differs only in the last digit, 14.78 vs 14.76 ms on
MH_01 OpenVINS p99). `tools/p01/` is canonical and the numpy duplicate is
deleted: it is stdlib-only so it runs on-device for P-02, and `p01_run.sh`
drives fresh runs of both estimators rather than reusing archived output.
Archived run sets with the older filenames get symlinked into
`dod_<SEQ>`/`ov_<SEQ>` shape; the one-liner is in the reducer's docstring.

---

## P-05 — The toolchain confound

**Status:** done, 2026-08-30. Written into `paper/main.tex` as a methods
subsection (\texttt{sec:toolchain}) with the T-013 table (container vs host,
$\Delta$ relative to container). Numbers unchanged from T-013; nothing
re-measured. P-04's caution about the unreproducible 0.1131 ASL figure is
unaffected and still open.

---

## P-02 — Close the embedded loop

**Status:** todo — **runbook written** (`docs/orin_runbook.md`); waiting on
device access. Everything else in the two-week plan proceeds without it.
On connect: record the environment block first, decide the ROS1 route
(docker arm64 noetic preferred), stage bags, mirror the P-01 harness.

Current embedded section is one sequence, ASL transport, no baseline, no
latency or memory numbers, and says so honestly. That honesty reads as a gap
because the embedded platform is where the layout thesis is strongest: 2x on a
6-core A78AE with a small cache means more than 2x on a 7950X.

**Needs:** ROS 1 on the device (the current blocker forcing ASL transport),
OpenVINS built on-device, full EuRoC + KAIST, and P-01's latency/RSS numbers on
both platforms so the comparison is like-for-like.

**Exit condition:** the same table as x86, on aarch64, with OpenVINS beside it.

---

## P-03 — DOD-Schur vs ov_SchurVINS

**Status:** blocked, 2026-08-30. **The authors' fork, as shipped, cannot
complete MH_01 in this environment. The paper's ablation row 9 (1.97x/2.16x
vs ov_SchurVINS) is currently unreproducible from the on-disk source and must
be either reproduced or dropped.**

**What works:**
- `VIO_SCHUR` env knob added to both bag runners (same pattern as
  `dod_asl_runner.cpp`), default off. **No-op verified**: p03 tree, knob
  unset, MH_01 bag_start 0 — estimate CSV **byte-identical** to the p01 run.
- DOD-Schur path runs the full sequence: MH_01 bag_start 40, 2897 frames,
  exit 0 (`/workspace/acv/p03/out/mh01_dodschur_*`).

**What does not:** `ov_SchurVINS` rebuilt from the pristine on-disk source
(`/workspace/ov_schurvins_src`, uniform extraction mtimes — no local edits)
in `/workspace/schur_ws`, both Release and Debug:
- Serial mode: deterministic abort at frame **896** of MH_01, inside their
  Schur stacking ("Stacking Extra W for Landmark 4699, state 21") — same
  landmark, same frame count, both build types. At bag_start 40 it diverges
  first (p_IinG ~ 19 km, "No features to update after triangulation" x1386).
- Subscribe mode + rosbag play: aborts on **negative covariance diagonals**
  in their own `StateHelper::SV_EKFUpdate()` — the exact M-13 failure mode
  (dense-form asymmetry) that DOD's rank-k update was written to fix.
- Their own shipped `config/euroc_mav/estimator_config.yaml` enables camera
  pose calibration their own code aborts on ("not implemented yet"). A
  working configuration requires calib_cam_extrinsics/intrinsics/timeoffset
  false and use_fej false (now in `/workspace/ovrun/schur_euroc/`).

**Provenance gap, again (M-18/T-013):** the paper's row-9 numbers came from a
build that no longer exists, with a protocol (bag_start 40 + static init)
that does not survive against the current source in either init mode. No
launch script, working config, or estimate output from that run survives.
Treat row 9 like the withdrawn 0.1131 table: not citable until reproduced.

**Decision (user, 2026-08-30): drop row 9.** Done — related-work sentence,
wall-clock paragraph, ablation row and limitations all updated in
`paper/main.tex`; limitations now states the withdrawal and why. The fork's
crash evidence stays recorded here if anyone ever wants to root-cause it.

Artifacts: `/workspace/acv/p03/` (container); p03 tree = d05d5a0 + 2-file
runner-knob patch (`/tmp/p03_runner_knob.patch` on the moonlab host).

---

## P-04 — Transport confound, full matrix

**Status:** todo. Currently n=1 (MH_01, 0.1131 ASL vs 0.1296 bag) supporting a
claim that this confound is "present, unremarked, in several papers". That
claim needs 10 sequences x 2 estimators x 2 transports.

**Caution learned today:** the 0.1131 figure in that comparison is one of the
numbers T-013 could not reproduce at any commit. Re-measure both sides before
building an argument on it.

---

## P-06 — Convert speed headroom into accuracy

**Status:** todo, tight. The strongest reframing available ("same cost, better
accuracy" beats "faster, same accuracy") and the least likely to fit two weeks.

**Two warnings from this session.** Capping features at OpenVINS's 40 already
lost accuracy on 6 of 8 sequences, and every constant swept today either
transferred badly or turned out to be a different constant in disguise. If this
is attempted, it is one lever, swept, gated — not a retune.

---

## P-07 — Artifact release

**Status:** todo, small. Docker image, run scripts, and the timing/ATE CSVs.
The control-tree method from T-003 is the reproducibility story: ship the
script that builds two commits and diffs their estimate CSVs.

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

**Status:** done, 2026-08-30. **Answer: no.** The fix is correct and necessary;
it is nowhere near sufficient, and the diagnostics point at a different gate
than the ticket assumed.

Run in `ros_container_v2`, `vio_rosbag_runner_mid_altitude`, with the new
`VIO_MAX_DIST` knob providing the control.

| run | ATE (m) | est path | gt path | outcome |
|---|---|---|---|---|
| 40_4 md75 | 12.814 | 921.8 | 718.7 | converges |
| 40_4 md200 | 12.814 | 921.8 | 718.7 | **byte-identical to md75** |
| 60_4 md75 | 137208 | 461006 | 975.3 | diverges, runs to end |
| 60_4 md200 | 1367.7 | 5567 | 111.7 | **aborts** ~frame 800 |
| 80_4 md200 | 56303 | 187345 | 1015.0 | diverges |
| 100_4 md200 | 94897 | 316338 | 1056.2 | diverges |

**The FOV derivation is confirmed exactly where it predicted.** At 40 m the
slant range is 53.5 m, below both constants, so `max_dist` cannot matter — and
md75 and md200 produce identical output. The derivation was right about its own
scope. 40_4's ATE of 12.814 also reproduces the recorded 12.81.

**60_4 does not converge; it fails differently.** With md75 it drifts to 461 km
of path. With md200 it dies at
`EKFPropagation() - diagonal at 101 is -3.86858e-10`, the negative-covariance
abort, after ~800 frames. Raising the cap admitted deep features that then
destabilised the filter. Both runs share an identical initialisation
(`rec_cond=0.00336631`, `gravity-vs-accel=6.106 deg`), which confirms
`max_dist` does not touch init — so cause 2 from the ticket is real and
untreated.

**The unexpected result, and the reason T-005 changes shape.** 80_4's
triangulation counters:

```
[TRI] accept=3857 reject_cond=64947 reject_mindist=9919 reject_maxdist=13744
[DB]  slam_features=0        (40_4: 4)
[CLS] slam_update=1.16/frame (40_4: 41.57)
```

`reject_cond` is **4.7x larger than `reject_maxdist`**. The dominant gate at
altitude is the triangulation conditioning number, not the distance cap.
`skip_tri` is 98.6% at 80 m and 97.7% at 100 m, against 64% at 40 m, and the
filter reaches zero SLAM features. It is not diverging because it mis-weights
features; it is diverging because it has almost none.

**Prediction for T-006, recorded now so it can be wrong.** 60_4's md200 failure
is a covariance blow-up caused by admitting deep, badly-conditioned
triangulations at full confidence. That is precisely the case
`parallax_noise_scale` exists for. If the model works at all, lambda > 0 should
make md200 survive where md75 could not. If 60_4 still aborts with lambda
swept, the model does not do the job it was written for.

---

## T-005 — Triangulation conditioning and init at altitude

**Status:** done, 2026-08-30. **Conclusion: no constant in this pipeline
rescues 60/80/100 m.** Three levers swept, 30+ runs. Two bugs found on the way,
one of them serious.

### Lever 1 — `max_cond_number` (the one T-004 pointed at)

| altitude | 1e4 | 1e5 | 1e6 | 1e7 |
|---|---|---|---|---|
| 40 | **12.814** | 14.299 | 16.855 | 16.855 |
| 60 | 1368 (abort) | 134421 | 134421 | 134421 |
| 80 | **56303** | 75686 | 75686 | 75686 |
| 100 | 94897 | 1935 | 2955 | **1916** |

Loosening helps 100 m by 50x, hurts 40 m and 80 m, rescues nothing. And the
reason it plateaus is instructive: at 1e7 on 60 m the rejects have simply moved
house — `reject_cond` 6, but `reject_mindist` 44804 and `reject_maxdist` 42888.
The triangulations were always degenerate, landing at ~0 or ~infinity; the
conditioning gate was labelling them, not causing them. At 40 m,
`reject_maxdist` is 0.

### Lever 2 — initialisation

`VIO_INIT_MIN_REC_COND` works: 0.01 and 0.05 both reject 60_4's bad solve and
stop the abort. They do not stop the divergence (66623 and 139302). `rec_cond`
is `smin/smax`, so higher is better-conditioned, and 60_4's 0.00337 is
*better* than EuRoC's 1.6e-4 while diverging — the conditioning number is not
what separates these regimes.

`VIO_INIT_WINDOW` is **inert** on this dataset — see the bugs below.

### Lever 3 — temporal baseline (`VIO_MAX_CLONES`)

The stereo rig is 0.30 m against 40-100 m of depth (Z/B of 130-330), so
temporal baseline is the only parallax available. Lengthening the window helps
once and then falls off a cliff:

| altitude | 11 | 15 | 17 | 19 |
|---|---|---|---|---|
| 40 | **12.814** | 12.810 | 39117 | 95527 |
| 60 | 1368 | 206549 | — | 161223 |
| 80 | 56303 | **6975** | 60716 | 73191 |
| 100 | 94897 | 125921 | 157775 | 111138 |

80 m improves 8x at 15 clones. Beyond 15 **every** sequence collapses,
including 40 m, which is healthy at 11 and 15. That is a code coupling, not
physics.

### Bug A (serious, fixed) — `max_clone_size = 20` silently starves the filter

`state_helper.cpp:430` stops inserting at `num_clones < 20`; marginalisation
fires on `num_clones > max_clone_size`. At 20 that condition is unreachable, so
the window never slides. 40_4 goes from ATE 12.810 to 94149.750, accepted
triangulations 29433 -> 36, `meas_overflow` 45428 — and the run completes
normally and reports a trajectory. No warning. Clamp fixed to 19, invariant
documented at the declaration. `FEATURE_MAX_CLONES_SUPPORTED` claims 23, which
is misleading given the hard 20 in `state_helper`.

### Bug B (open) — `VIO_INIT_WINDOW` does nothing on the featureless path

`w4` and `w6` produce **byte-identical** output with identical
`rec_cond=0.00336631`. `init_featureless` uses `init_dyn_num_pose`, not
`init_window_time`, and logs `pairs=3 times=3` regardless. The claim recorded
in `mid_altitude_options.hpp` — "rec_cond vs window on 60_4: 0.0034 (3 frames),
0.0068 (4 s), 0.065 (6 s)" — does not reproduce through this knob. Either the
knob is not wired to the path this dataset takes, or that measurement came from
somewhere else. **Not fixed; needs its own ticket.**

### What collapses past 15 clones — characterised, not explained

At 40 m, going 11 -> 17 clones: `maxtrack` per frame **rises** 3.49 -> 10.09,
while `marginal` falls 9.74 -> 0.45, `slam_update` falls 41.57 -> 0.99 and
`slam_features` reaches 0. My first explanation was that tracks cannot survive
a longer window; the counters say the opposite — more features reach max track,
and it is SLAM promotion and marginalisation that die. Mechanism not
identified. Recorded rather than guessed.

**Exit:** the altitude regime is not reachable by tuning. T-006 is the last
in-branch candidate, and its prediction from T-004 is still standing.

---

## T-006 — Sweep VIO_PARALLAX_LAMBDA

**Status:** done, 2026-08-30. **The model is refuted.** Everything it appeared
to achieve is reproduced exactly by a flat measurement-noise inflation with the
model switched off, and the flat version is strictly better at every altitude.

**The T-004 prediction was right in outcome and wrong in mechanism.** It said
lambda > 0 should let 60_4 survive where max_dist=200 alone aborted. It does —
1367.7 (abort) becomes 48.25. But lambda is not what did it.

**The clamp counter, added before the sweep, is what caught it.** At every
lambda >= 20, `pnw_clamped` is within a handful of `pnw_computed`
(168481/168483) and `mean_scale` reads exactly 100.000. Every measurement gets
the cap. `parallax_noise_scale` degenerates to the constant `max_scale`, which
is algebraically `sigma_pix *= sqrt(max_scale)` — and that is why lambda 40,
60, 80 and 100 give byte-identical output.

**The control settles it.** `VIO_SIGMA_PIX=10` with the model off:

| altitude | lambda sweep | flat control | 
|---|---|---|
| 40 | 15.419 | 15.419 |
| 60 | 48.252 | 48.251 |
| 80 | 32.068 | 32.067 |
| 100 | 4461.8 (abort) | **110.271** |

Identical at three altitudes and far better at 100 m. The cap-shape sweep
agrees: `max_scale=25` gives 25.409 and `sigma_pix=5` gives 25.409, exactly as
`sqrt(25) = 5` requires.

**In its actual graded regime the model is erratic and never wins.** At
Z/B of 20-300, any lambda >= 1 saturates any usable cap, so the graded regime
needs lambda ~ 0.01-0.1 with the cap lifted. On 80 m: 0.005 -> 49482,
0.01 -> 43941, 0.02 -> 49.9, 0.05 -> 174332, 0.1 -> 45.2. Non-monotonic by four
orders of magnitude — the filter is sitting on a stability cliff and the
weighting decides which features survive, not how much they are trusted. Best
graded result (45.2) loses to flat sigma=15 (23.2). Applied on top of the flat
optimum it destroys it: 25.4 -> 1256.7.

**What actually fixes altitude: sigma_pix, an inherited EuRoC placeholder.**

| altitude | sigma=1 (as shipped) | best flat sigma | improvement |
|---|---|---|---|
| 40 | **12.814** (sigma=1) | 12.814 | — |
| 60 | 1368 (abort) | 48.25 (sigma=10) | 28x |
| 80 | 56303 | **23.17** (sigma=15) | 2430x |
| 100 | 94897 | **110.27** (sigma=10) | 861x |

The whole envelope now runs. Drift is 1.8% of path at 40 m and 2-11% at
60-100 m, against three orders of magnitude of divergence before. The optimal
sigma rises with altitude (1, 10, 15, 10), which is an altitude-dependent noise
story — just not the Z/B one this branch was built on.

This is M-14 again, and the config file predicted it: every constant there is
marked PLACEHOLDER except `sigma_pix`, which was the one that mattered.

**Consequence for T-009.** "Altitude-robust stereo VIO via a parallax-scaled
covariance model" is dead as a claim. What survives is measured and real: a
characterisation of where MSCKF breaks at altitude, two genuine bugs, and the
finding that the dominant lever is a pixel-noise constant nobody had questioned.

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
