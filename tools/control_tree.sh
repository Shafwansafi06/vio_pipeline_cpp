#!/usr/bin/env bash
# Control-tree comparison (see tools/control_tree_README.md).
# Env: BASE=<commit> NAME=<label> DATA=<asl-mav0-dir> [SEQS="MH_01 ..."]
# Runs on the lab box inside ros_container_v2; trees land in
# /workspace/ct_$NAME/{base,head}.
set -euo pipefail
: "${BASE:?set BASE to the base commit/ref}"
: "${NAME:?set NAME}"
DATA="${DATA:?set DATA to the ASL mav0 dir}"
ROOT=/workspace/ct_${NAME}
REPO_HOST="${REPO_HOST:-/media/storage/moonlab/vio_parity/vio_pipeline_cpp}"
SEQS="${SEQS:-MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03}"

mkdir -p "$ROOT" && cd "$ROOT"
cp -r "$REPO_HOST" head 2>/dev/null || true

# base tree: archive of $BASE applied over a clean checkout
mkdir -p base
cd base
if [ -e .git ] || [ -f .git ]; then git archive "$BASE" | tar x -C .; else
  echo "NOTE: building base from $BASE requires a git checkout; see README" >&2
fi

for side in head base; do
  if [ ! -x "$ROOT/$side/build/dod_asl_runner" ]; then
    cmake -S "$ROOT/$side" -B "$ROOT/$side/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DVIO_BUILD_ASL_RUNNER=ON -DVIO_BUILD_OPENCV_FRONTEND=ON \
        > "$ROOT/${side}_cmake.log" 2>&1
    make -C "$ROOT/$side/build" dod_asl_runner -j6 >> "$ROOT/${side}_cmake.log" 2>&1
  fi
done

printf "%-8s %-10s %-34s %-34s %s\n" seq head_md5 head_ate base_md5 verdict
for SEQ in $SEQS; do
  for side in head base; do
    "$ROOT/$side/build/dod_asl_runner" "$DATA" "$ROOT/${side}_${SEQ}" \
        > "$ROOT/${side}_${SEQ}.log" 2>&1 || true
  done
  HM=$(md5sum "$ROOT/head_${SEQ}_estimate.csv" | cut -d" " -f1)
  BM=$(md5sum "$ROOT/base_${SEQ}_estimate.csv" | cut -d" " -f1)
  if cmp -s "$ROOT/head_${SEQ}_estimate.csv" "$ROOT/base_${SEQ}_estimate.csv"; then
    V=IDENTICAL
  else
    V=DIFFERS
  fi
  printf "%-8s %-10.10s %-34.34s %-34.34s %s\n" "$SEQ" "$HM" "" "$BM" "$V"
done
