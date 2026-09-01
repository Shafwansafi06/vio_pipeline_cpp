# Artifact / reproduction map

Every number in the paper traces to an archived run or a script in this
repository. This file is the map.

## Environments

- **x86 container**: `ros_container_v2` on moonlab (lab x86 box) — Ubuntu 20.04,
  ROS1 Noetic, OpenCV 4.2.0, g++ 9.4.0. All x86 tables (ATE, wall-clock,
  latency, RSS) come from inside this container. Runs of 2026-08-30.
- **aarch64 container**: `ov_ros1_20_04` on the Jetson Orin Nano
  (moonlab (Orin Nano)) — Ubuntu 20.04 arm64, same g++ 9.4.0 / OpenCV 4.2.0,
  so the ISA is the only variable against the x86 tables.

## Data on the lab box (moonlab (lab x86 box))

| path | contents |
|---|---|
| `/media/storage/moonlab/vio_parity/acv/p01/` | x86 per-frame latency + RSS matrix, all ten sequences, both estimators; `latency_table.txt`, `latency.json`, `latency_cdf.png` |
| `/media/storage/moonlab/vio_parity/acv/p02/` | the same on the Orin Nano, plus `latency_cdf_aarch64.png` |
| `/media/storage/moonlab/vio_parity/acv/p04/` | transport matrix (ASL vs bag), OpenVINS state files, RPE inputs, hyperfine logs |
| `/media/storage/moonlab/vio_parity/orin_bags/` | all ten bags staged for the Orin |

## Harnesses (in this repo)

- `tools/p01/` — x86 latency/RSS harness: `p01_run.sh` (fresh runs of both
  estimators, all ten sequences, `record_timing` for OpenVINS via its
  upstream config flag), `withrss.py` (wait4 + `ru_maxrss`; `/usr/bin/time`
  is not installed in the container), `latency_stats.py` (p50/p99/max/CDF,
  stdlib), `cdf_figure.py`, `ate_check.sh` (re-evaluates ATEs against the
  recorded gate).
- `tools/p02/` — the same for the Orin: `p02_run.sh` (docker-run based,
  `/p02` paths) and `build_on_orin.sh` (builds both estimators inside
  `ov_ros1_20_04`).
- `tools/control_tree.sh` + `control_tree_README.md` — the control-tree
  method used by every accepted optimisation: build two commits side by
  side in one container, run identical data, `cmp` estimate CSVs
  byte-for-byte.

## Where each paper number comes from

| paper table | source |
|---|---|
| Table ATE (accuracy, 10 seq) | fresh in-container runs 2026-08-30: DOD `p01/dod_*_estimate.csv` (bag) + `p04/asl_*` (V1 gt via ASL stream); OpenVINS `p04/ovstate/ov_*_est.txt` (save_total_state) |
| Table speed | `p04/speed.log` (hyperfine, 5 runs, warmup, same container) |
| Table latency (x86) | `p01/` timing CSVs + `latency_table.txt` |
| Figure CDF | `p01/latency_cdf.png` |
| Table aarch64-latency + CDF | `p02/` (Orin Nano, same-toolchain container) |
| Table toolchain | T-013 measurement (CLAUDE.md gate table) |
| Table transport | DOD ASL vs bag runs in-container (`p04/asl_*` vs `p01/dod_*`) |
| Table ablation | historical staged measurements, each gate-verified bit-identical (board T-tickets) |
| Table RPE | `p04/rpe/` via `scripts/error_profile.py` (restored from commit 76acf5c) |

## Reproduce in one line each

- Accuracy: `tools/sweep.sh` (recorded gate) or the control-tree byte-compare
  (`tools/control_tree_README.md`) — the method that makes every
  optimisation claim checkable.
- Latency/RSS: `bash tools/p01/p01_run.sh` in the x86 container;
  `bash ~/p02/p02_run.sh` (shipped in `tools/p02/`) on the Orin.
- OpenVINS outputs need no patching: `record_timing_information` and
  `save_total_state`/`filepath_est` are passed as ROS node parameters
  (`_save_total_state:=true _filepath_est:=...`) — as YAML keys they are
  silently ignored by the serial node's visualizer.
