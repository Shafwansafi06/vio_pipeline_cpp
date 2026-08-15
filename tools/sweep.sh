#!/usr/bin/env bash
# Run one configuration of the ASL runner and print its ATE. Config comes from
# the VIO_* env vars the runner reads.
#   NAME=slam_on VIO_ENABLE_SLAM=1 tools/sweep.sh
set -euo pipefail
ROOT="${ROOT:-/media/storage/moonlab/vio_parity}"
NAME="${NAME:?set NAME}"
DATA="${DATA:-$ROOT/data/mav0}"
"$ROOT/vio_pipeline_cpp/build/dod_asl_runner" "$DATA" "$ROOT/runs/$NAME" \
    > "$ROOT/runs/$NAME.log" 2>&1 || true
"$ROOT/venv/bin/python" "$ROOT/vio_pipeline_cpp/scripts/evaluate_trajectory.py" \
    "$ROOT/runs/${NAME}_estimate.csv" "$ROOT/runs/${NAME}_groundtruth.csv" \
    > "$ROOT/runs/$NAME.json"
"$ROOT/venv/bin/python" - "$NAME" "$ROOT/runs/$NAME.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[2]))
print("%-22s ATE %.4f m   path %.2f (gt %.2f)   p95 %.3f   assoc %d" % (
    sys.argv[1], d["ate_rmse_m"], d["estimated_path_length_m"],
    d["groundtruth_path_length_m"], d["ate_p95_m"], d["associated_samples"]))
EOF
