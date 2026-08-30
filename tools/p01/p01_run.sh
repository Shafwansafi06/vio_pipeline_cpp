#!/usr/bin/env bash
# P-01: fresh runs of both estimators on all ten sequences inside
# ros_container_v2, under a wait4 wrapper that records ru_maxrss.
# Outputs in /workspace/acv/p01/:
#   dod_<SEQ>_timing.csv, dod_<SEQ>_estimate.csv   (DOD runners)
#   ov_<SEQ>_timing.txt                            (OpenVINS record_timing)
#   <side>_<SEQ>.rss / .log                        (wrapper stderr, node log)
export ROS_DISTRO=noetic
source /opt/ros/noetic/setup.bash
source /workspace/openvins_instrumented_ws/devel/setup.bash || true
source /workspace/acv/head/build_ros/setup.bash || true

P01=/workspace/acv/p01
mkdir -p "$P01"
cd "$P01"

if ! pgrep -x roscore >/dev/null 2>&1; then
    nohup roscore >/dev/null 2>&1 &
    sleep 4
fi

OV_NODE=/workspace/openvins_instrumented_ws/devel/lib/ov_msckf/ros1_serial_msckf
DOD_EUROC=/workspace/acv/head/build_ros/vio_rosbag_runner_euroc
DOD_KAIST=/workspace/acv/head/build_ros/vio_rosbag_runner

EUROC_DIR=/workspace/euroc_bags
EUROC_MH01=/workspace/EuROC/MH_01_easy.bag
KAIST_CFG=/workspace/ovrun/kaist_stereo/estimator_config.yaml
EUROC_CFG=/workspace/ovrun/euroc_mav/estimator_config.yaml

run_dod() {
    local side=$1 seq=$2 bag=$3 runner=$4
    echo "=== DOD $seq $(date +%T)"
    python3 /workspace/acv/p01/withrss.py "$runner" "$bag" "$P01/${side}_${seq}" \
        > "$P01/${side}_${seq}.rss" 2> "$P01/${side}_${seq}.log"
    tail -1 "$P01/${side}_${seq}.rss"
}

run_ov() {
    local seq=$1 bag=$2 cfg=$3
    echo "=== OpenVINS $seq $(date +%T)"
    local cfgdir
    cfgdir=$(dirname "$cfg")
    # Config copy must live in the SAME directory as the source config:
    # estimator_config.yaml points at kalibr_*.yaml by RELATIVE path, and
    # OpenVINS resolves those against the config file's own directory.
    sed -e "s|^record_timing_information: .*|record_timing_information: true|" \
        -e "s|^record_timing_filepath: .*|record_timing_filepath: \"$P01/ov_${seq}_timing.txt\"|" \
        "$cfg" > "$cfgdir/cfg_p01_${seq}.yaml"
    python3 /workspace/acv/p01/withrss.py "$OV_NODE" \
        "_config_path:=$cfgdir/cfg_p01_${seq}.yaml" "_path_bag:=$bag" \
        "_bag_start:=0" "_bag_durr:=-1" "_verbosity:=SILENT" \
        > "$P01/ov_${seq}.rss" 2> "$P01/ov_${seq}.log"
    tail -2 "$P01/ov_${seq}.log" | head -1
    rm -f "$cfgdir/cfg_p01_${seq}.yaml"
}

declare -A BAGS=(
    [MH_01]="$EUROC_MH01"
    [MH_02]="$EUROC_DIR/MH_02_easy.bag"
    [MH_03]="$EUROC_DIR/MH_03_medium.bag"
    [MH_04]="$EUROC_DIR/MH_04_difficult.bag"
    [MH_05]="$EUROC_DIR/MH_05_difficult.bag"
    [V1_01]="$EUROC_DIR/V1_01_easy.bag"
    [V1_02]="$EUROC_DIR/V1_02_medium.bag"
    [V1_03]="$EUROC_DIR/V1_03_difficult.bag"
    [circle]=/workspace/circle.bag
    [infinite]=/workspace/ovrun/infinite.bag
)

for SEQ in MH_01 MH_02 MH_03 MH_04 MH_05 V1_01 V1_02 V1_03 circle infinite; do
    BAG=${BAGS[$SEQ]}
    if [ ! -f "$BAG" ]; then echo "MISSING BAG: $SEQ $BAG"; continue; fi
    if [[ "$SEQ" == circle || "$SEQ" == infinite ]]; then
        run_dod dod "$SEQ" "$BAG" "$DOD_KAIST" || true
        run_ov "$SEQ" "$BAG" "$KAIST_CFG" || true
    else
        run_dod dod "$SEQ" "$BAG" "$DOD_EUROC" || true
        run_ov "$SEQ" "$BAG" "$EUROC_CFG" || true
    fi
done

echo "=== timing file sizes:"
wc -l "$P01"/*_timing.csv "$P01"/*_timing.txt 2>/dev/null
echo "=== RSS summary:"
grep -H "maxrss_kb" "$P01"/*.rss 2>/dev/null
