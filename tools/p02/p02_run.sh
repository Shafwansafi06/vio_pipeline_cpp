#!/usr/bin/env bash
# P-02: latency + RSS matrix on the Orin, inside ov_ros1_20_04.
# Mirrors tools/p01/p01_run.sh (x86) so the two tables are comparable:
# same transport (rosbag read by each estimator directly), same toolchain
# (g++ 9.4 / OpenCV 4.2), only the ISA differs.
# Outputs: /p02/out/ -> host ~/p02/out
set -u
export ROS_DISTRO=noetic
source /opt/ros/noetic/setup.bash

P02=/p02
OUT=$P02/out
mkdir -p "$OUT"
cd "$OUT"

if ! pgrep -x roscore >/dev/null 2>&1; then
    nohup roscore >/dev/null 2>&1 &
    sleep 4
fi

OV_NODE=/p02/openvins_ws/devel/lib/ov_msckf/ros1_serial_msckf
DOD_EUROC=/p02/dod/build_ros/vio_rosbag_runner_euroc
DOD_KAIST=/p02/dod/build_ros/vio_rosbag_runner
EUROC_CFG=/p02/euroc_mav/estimator_config.yaml
KAIST_CFG=/p02/kaist_stereo/estimator_config.yaml
BAGS=/bags

run_dod() {
    local side=$1 seq=$2 bag=$3 runner=$4
    echo "=== DOD $seq $(date +%T)"
    python3 /p02/withrss.py "$runner" "$bag" "$OUT/${side}_${seq}" \
        > "$OUT/${side}_${seq}.rss" 2> "$OUT/${side}_${seq}.log"
    grep RSS "$OUT/${side}_${seq}.log"
}

run_ov() {
    local seq=$1 bag=$2 cfg=$3
    echo "=== OpenVINS $seq $(date +%T)"
    local cfgdir
    cfgdir=$(dirname "$cfg")
    sed -e "s|^record_timing_information: .*|record_timing_information: true|" \
        -e "s|^record_timing_filepath: .*|record_timing_filepath: \"$OUT/ov_${seq}_timing.txt\"|" \
        "$cfg" > "$cfgdir/cfg_p02_${seq}.yaml"
    python3 /p02/withrss.py "$OV_NODE" \
        "_config_path:=$cfgdir/cfg_p02_${seq}.yaml" "_path_bag:=$bag" \
        "_bag_start:=0" "_bag_durr:=-1" "_verbosity:=SILENT" \
        > "$OUT/ov_${seq}.rss" 2> "$OUT/ov_${seq}.log"
    grep RSS "$OUT/ov_${seq}.log"
    rm -f "$cfgdir/cfg_p02_${seq}.yaml"
}

declare -A BAGS_MAP=(
    [MH_01]="$BAGS/MH_01_easy.bag"
    [MH_02]="$BAGS/MH_02_easy.bag"
    [MH_03]="$BAGS/MH_03_medium.bag"
    [MH_04]="$BAGS/MH_04_difficult.bag"
    [MH_05]="$BAGS/MH_05_difficult.bag"
    [V1_01]="$BAGS/V1_01_easy.bag"
    [V1_02]="$BAGS/V1_02_medium.bag"
    [V1_03]="$BAGS/V1_03_difficult.bag"
    [circle]="$BAGS/circle.bag"
    [infinite]="$BAGS/infinite.bag"
)

for SEQ in MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 circle infinite; do
    BAG=${BAGS_MAP[$SEQ]}
    if [ ! -f "$BAG" ]; then echo "MISSING BAG: $SEQ"; continue; fi
    if [[ "$SEQ" == circle || "$SEQ" == infinite ]]; then
        run_dod dod "$SEQ" "$BAG" "$DOD_KAIST" || true
        run_ov "$SEQ" "$BAG" "$KAIST_CFG" || true
    else
        run_dod dod "$SEQ" "$BAG" "$DOD_EUROC" || true
        run_ov "$SEQ" "$BAG" "$EUROC_CFG" || true
    fi
done

echo "=== done $(date +%T)"
wc -l "$OUT"/*_timing.csv "$OUT"/*_timing.txt 2>/dev/null | tail -25
grep -H RSS "$OUT"/*.log 2>/dev/null
