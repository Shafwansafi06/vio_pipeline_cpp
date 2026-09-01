# Orin Nano runbook (P-02)

Goal: turn the embedded section from feasibility into the x86 table
reproduced on aarch64, with OpenVINS beside it, plus latency/RSS like-for-like.
Exit condition (board P-02): the same table as x86, on aarch64, with OpenVINS
next to it.

## Already true (do not redo)

- DOD runs on the device once already: MH_01 ASL transport, ATE 0.1344 m,
  no divergence, 98.8 s hyperfine (paper Sec. embedded). That run's build
  flags equal the x86 ones.
- The aarch64 gravity-convergence tolerance defect is fixed in-tree
  (widened to 3e-3, x86 trajectories verified unmoved).
- The x86 reference numbers this must be compared against are the P-01
  outputs (`vio_parity/acv/p01/`): per-frame latency tables, CDF figure,
  max RSS via `wait4`/`ru_maxrss` — same wrappers to reuse on-device.

## On first connect: record these (they go in the paper, unpinned)

    cat /etc/nv_tegra_release          # L4T/JetPack version
    lsb_release -d                     # Ubuntu base (R36.x = 22.04)
    g++ --version; python3 --version
    dpkg -l | grep -i opencv | head    # system OpenCV, if any
    free -h; nproc                     # confirm 8 GB / 6 cores
    df -h /home                        # need ~15 GB free for bags + builds

## ROS1 decision (the known blocker)

JetPack 6 is Ubuntu 22.04; ROS1 Noetic has no Ubuntu 22.04 debs. In order of
preference:

1. **Docker arm64 ROS1 Noetic image** (keeps the x86 methodology: both
   estimators inside one container, identical transport). Needs docker +
   nvidia runtime on the device; pull an arm64 noetic image, no GPU needed.
2. RoboStack conda ROS1 (no docker) — acceptable, but then OpenVINS must
   build against that env too.
3. If neither works in reasonable time: fall back to ASL transport for DOD
   and report OpenVINS-on-device as not available — but then P-02's exit
   condition is not met and the section stays "preliminary" (RA-L fallback
   territory per docs/paper_spine.md).

## Staging (from moonlab host)

    # bags (MH_01 already proved; take all ten for the full table)
    scp moonlab (lab x86 box):/media/storage/moonlab/vio_parity/{MH_01_easy,V1_01_easy}.bag orin:bags/
    scp moonlab (lab x86 box):/media/storage/moonlab/vio_parity/euroc_bags/*.bag orin:bags/
    # EuRoC ASL zips for the ROS-free path
    # repo tree + tools/p01 wrappers
    git -C vio_pipeline_cpp archive HEAD | ssh orin 'mkdir -p vio && tar x -C vio'

Inside the container/device: build DOD twice (ASL runner needs no ROS:
`cmake -DVIO_BUILD_ASL_RUNNER=ON -DVIO_BUILD_OPENCV_FRONTEND=ON ..`; bag
runners additionally `-DVIO_BUILD_ROS1=ON`), and build OpenVINS from the
same pinned source the x86 numbers used.

## Runs (mirror tools/p01/p01_run.sh)

- DOD: `vio_rosbag_runner_euroc BAG OUT` per EuRoC bag, `vio_rosbag_runner`
  for circle/infinite; each writes `_estimate.csv`, `_groundtruth.csv`,
  `_timing.csv`.
- OpenVINS: `ros1_serial_msckf _config_path:=... _path_bag:=... _verbosity:=SILENT`
  with `record_timing_information: true` + `record_timing_filepath` sed'ed
  per sequence (same sed as p01_run.sh; config copies must sit in the
  config's own dir — kalibr paths are relative).
- Both under `tools/p01/withrss.py` for `ru_maxrss`.
- ATE: `scripts/evaluate_trajectory.py` per sequence; compare against the
  x86 container table in CLAUDE.md (they will NOT be bit-identical across
  architectures — that comparison is x86-only. Quote the aarch64 ATEs as
  their own column, environment named).

## Deliverables

1. `p01`-style latency table + CDF, both estimators, aarch64.
2. Max RSS both estimators, aarch64.
3. ATE table aarch64 (DOD vs OpenVINS, same transport).
4. Wall-clock DOD vs OpenVINS on-device (hyperfine, `-i`, 5 runs) — the
   number the paper's thesis actually needs: does 2x survive on the small
   core, where layout is supposed to dominate.
5. Paper Sec. embedded rewritten from "preliminary" to the real section;
   environment column added everywhere.
