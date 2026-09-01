# Working notes for this repository

## Performance work: measure, never guess

**Every performance change must be justified by a profile taken before it and
verified by a measurement after it.** This is not a style preference — on this
codebase, confident reasoning about where the time goes has been wrong more
often than right:

- "The back-end is slower because we feed ~2× the features per MSCKF update."
  Wrong. Capping at official's 40 features changed the MSCKF stage by 3%
  (1.581 vs 1.538 ms) and cost accuracy on 6 of 8 sequences.
- "Our `M_a` loop is structured differently from official's." Wrong — official
  has the identical nested loop. (Restructuring it was still a 2.2× win, but
  for a reason the guess did not identify.)
- "Building `S` block-wise avoids the marginal-covariance gather, so it must be
  faster." Wrong. It is slower: 0.285 → 0.404 ms, because it trades two dense
  products for ~169 small ones per feature.

Three of those were settled in minutes by a profiler after being argued about
at length. Profile first.

## Tools

Both are installed on the lab box (`moonlab (lab x86 box)`). `sudo` there wants
a password, so neither was installed with a package manager — do not assume
`apt` is available.

| tool | location | notes |
|---|---|---|
| `hyperfine` 1.18.0 | `~/bin/hyperfine` on the host | static musl binary, no deps |
| `hyperfine` | `/usr/local/bin/hyperfine` inside `ros_container_v2` | same binary, `docker cp`'d in |
| `valgrind` 3.18 | `/usr/bin/valgrind` on the host | preinstalled; **not** present in the container |

To reinstall hyperfine elsewhere: the lab network is behind a captive portal, so
pipe the release through a machine that has real internet —
`python3 -c "...urlopen..." | ssh <host> 'cat > f'`. `curl` on the laptop is
intercepted by a shell hook and fails.

## Allocation profiling (valgrind DHAT)

DOD's central design claim is bounded, allocation-free-per-frame state. That
claim is checked, not asserted:

```bash
valgrind --tool=dhat --dhat-out-file=/tmp/dhat.out \
    ./build/dod_asl_runner data/mav0 /tmp/vgrun 12      # 12 s of data is enough
```

DHAT prints totals to stderr; rank the sites by parsing the JSON (`pps[]`,
`tb` = total bytes, `tbk` = total blocks, `fs` → `ftbl` for the stack). **Rank by
blocks AND by bytes — they point at different defects.** The 463k-allocation
site (`initialize_invertible`) and the 1.68 GB site (`update_slam`) were
invisible to each other's ranking.

Current baseline, 12 s of EuRoC MH_01: **825,790 blocks / 2.02 GB**. Roughly half
the blocks are OpenCV's KLT internals and are not ours to remove. Regressions
above this baseline want an explanation.

## Wall-clock benchmarking (hyperfine)

Benchmark whole processes through the **same transport** on both sides, so the
comparison is not confounded by how data is read. That means DOD's *rosbag*
runner against official's, not the ASL runner (ASL vs bag is worth ~15%: MH_01
is 0.1131 m via ASL and 0.1296 m via the bag path):

```bash
docker exec ros_container_v2 bash -lc '
  source /opt/ros/noetic/setup.bash; source $HOME/catkin_ws/devel/setup.bash
  hyperfine -i --warmup 1 --runs 5 \
    -n DOD      "./build_ros/vio_rosbag_runner_euroc /workspace/EuROC/MH_01_easy.bag /tmp/hf" \
    -n OpenVINS "rosrun ov_msckf ros1_serial_msckf _config_path:=/workspace/ovrun/euroc_mav/estimator_config.yaml _path_bag:=/workspace/EuROC/MH_01_easy.bag _bag_start:=0 _bag_durr:=-1 _verbosity:=SILENT"'
```

`-i` is required: official exits non-zero at shutdown (a `class_loader` unload
throw) *after* all of its work and output are complete.

Current baseline: MH_01 **18.656 s** vs official 29.014 s (1.56×); KAIST circle
**21.321 s** vs 27.952 s (1.31×).

## The accuracy gate

A performance change is only finished when the trajectories are unchanged. Run
the sweep and compare against the recorded ATEs:

```bash
DATA=$PWD/data/mav0 NAME=whatever tools/sweep.sh    # on the lab box
```

**Baseline, measured 2026-08-30, in `ros_container_v2`** (OpenCV 4.2.0,
g++ 9.4.0). EuRoC via `dod_asl_runner` on ASL directories, KAIST via
`vio_rosbag_runner` on bags:

| MH_01 | MH_02 | MH_03 | MH_04 | MH_05 | V1_01 | V1_02 | V1_03 | circle | infinity |
|---|---|---|---|---|---|---|---|---|---|
| 0.1293 | 0.2080 | 0.2343 | 0.4267 | 0.3285 | 0.0545 | 0.0482 | 0.0550 | 0.0374 | 0.0261 |

These are byte-stable: every commit from `1ec3389` (2026-08-17) through the
altitude branch produces bit-identical estimate CSVs on all eight EuRoC
sequences. An optimisation that moves them is an accuracy change wearing a
performance costume, and needs to be argued on accuracy terms.

Also run `ctest` (6/6) — `verify_dynamic_init` in particular catches
initialiser changes that the EuRoC sweep does not.

**The environment is part of the result — run the gate in the container.**
The KLT frontend is OpenCV's, and the host (OpenCV 4.5.4, g++ 11.4.0) does not
agree with the container on every sequence. Same code, same data, same day:

| | MH_01 | MH_02 | MH_03 | MH_04 | MH_05 | V1_01 | V1_02 | V1_03 |
|---|---|---|---|---|---|---|---|---|
| container | 0.1293 | 0.2080 | 0.2343 | 0.4267 | 0.3285 | 0.0545 | 0.0482 | 0.0550 |
| host | 0.1282 | 0.1595 | 0.2234 | 0.4416 | 0.3115 | 0.0499 | 0.0553 | 0.0564 |

Mostly within a percent, but MH_02 differs by 23% and V1_01 by 8%. Quoting an
ATE without naming the environment is quoting a number that cannot be checked.

**The old table (MH_01 0.1131, MH_02 0.1744, ...) was never reproducible.** Not
on the host, not in the container, and not at `a7addf6` — the commit that
recorded it, which returns 0.1496 for MH_01 on this data. Whatever produced
those figures was a configuration that no longer exists and was never written
down. They have been replaced above rather than chased. See
`docs/tickets/BOARD.md` T-013 and M-18.

**Prefer a control tree to the recorded table.** To answer "did my change move
anything", build the unchanged commit alongside it in the same environment and
`cmp` the estimate CSVs. That is exact, immune to environment drift, and it is
how T-003 established that the parallax whitening is a true no-op at
lambda = 0.

## Two traps that have already bitten

- **Do not collapse the covariance update** into one dense
  `Cov -= K * M_a.transpose()`. `K*M_aᵀ` is symmetric in exact arithmetic but not
  in floating point, and the asymmetry drives covariance diagonals negative —
  `EKFPropagation` aborts on MH_01 within ~16 s.
- **Tuned constants do not transfer between datasets.** `max_msckf_in_update` is
  75 on EuRoC and 50 on KAIST; `init_imu_thresh` at 0.25 fixes MH_01 and breaks
  V1_01 outright. Assume any constant tuned on one dataset is wrong on another
  until measured.
