#!/usr/bin/env bash
export ROS_DISTRO=noetic
cd /workspace/acv/p01
REPO=/workspace/acv/head
RECORDED="MH_01:0.1293 MH_02:0.2080 MH_03:0.2343 MH_04:0.4267 MH_05:0.3285 V1_01:0.0545 V1_02:0.0482 V1_03:0.0550 circle:0.0374 infinite:0.0261"
printf "%-9s %-10s %s\n" seq ATE recorded
for f in dod_*_estimate.csv; do
    seq=$(basename "$f" | sed -e "s/dod_//" -e "s/_estimate.csv//")
    rec=""
    for pair in $RECORDED; do
        case "$pair" in ${seq}:*) rec=${pair#*:} ;; esac
    done
    if [ ! -f "dod_${seq}_groundtruth.csv" ]; then echo "$seq no-gt"; continue; fi
    python3 "$REPO/scripts/evaluate_trajectory.py" \
        "dod_${seq}_estimate.csv" "dod_${seq}_groundtruth.csv" 2>/dev/null \
        | python3 /workspace/acv/p01/ate_line.py "$seq" "$rec" || echo "$seq FAILED"
done
