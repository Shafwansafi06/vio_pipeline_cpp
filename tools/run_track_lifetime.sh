#!/usr/bin/env bash
# Build and run both track dumps over the same EuRoC images, then compare.
#   OV=/path/to/open_vins MAV0=/path/to/mav0 tools/run_track_lifetime.sh [max_frames]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
OV="${OV:?set OV to the open_vins checkout (needs ov_core)}"
MAV0="${MAV0:?set MAV0 to the EuRoC mav0 directory}"
OUT="${OUT:-/tmp/track_lifetime}"
FRAMES="${1:-0}"
mkdir -p "$OUT"

INC="-I/usr/include/eigen3 -I/usr/include/opencv4"
CVLIBS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_calib3d -lopencv_features2d"

g++ -O3 -std=c++17 -o "$OUT/dod_track_dump" \
    "$HERE/tools/dod_track_dump.cpp" "$HERE/core/tracker.cpp" "$HERE/core/cam.cpp" \
    "$HERE/core/feature.cpp" "$HERE/type/quat_ops.cpp" \
    $INC $CVLIBS

g++ -O3 -std=c++14 -DENABLE_ARUCO_TAGS=0 -o "$OUT/ov_track_dump" \
    "$HERE/tools/ov_track_dump.cpp" \
    "$OV/ov_core/src/track/TrackKLT.cpp" "$OV/ov_core/src/track/TrackBase.cpp" \
    "$OV/ov_core/src/feat/Feature.cpp" "$OV/ov_core/src/feat/FeatureDatabase.cpp" \
    "$OV/ov_core/src/utils/print.cpp" \
    -I"$OV/ov_core/src" $INC $CVLIBS \
    -lboost_system -lboost_filesystem -lboost_thread -lboost_date_time -lpthread

VIO_TRACK_DUMP="$OUT/dod_tracks.txt" "$OUT/dod_track_dump" "$MAV0" "$FRAMES"
"$OUT/ov_track_dump" "$MAV0" "$OUT/ov_tracks.txt" "$FRAMES"

python3 "$HERE/scripts/track_lifetime.py" "$OUT/dod_tracks.txt" "$OUT/ov_tracks.txt"
