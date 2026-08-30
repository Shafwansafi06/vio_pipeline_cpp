#!/usr/bin/env bash
# P-04: fresh OpenVINS bag-transport runs, odom recorded to CSV.
export ROS_DISTRO=noetic
source /opt/ros/noetic/setup.bash
source /workspace/openvins_instrumented_ws/devel/setup.bash
OUT=/workspace/acv/p04
mkdir -p "$OUT/ovodom"; cd "$OUT/ovodom"

if ! pgrep -x roscore >/dev/null 2>&1; then nohup roscore >/dev/null 2>&1 & sleep 4; fi

OV=/workspace/openvins_instrumented_ws/devel/lib/ov_msckf/ros1_serial_msckf
CFG=/workspace/ovrun/euroc_mav/estimator_config.yaml
KCFG=/workspace/ovrun/kaist_stereo/estimator_config.yaml

run_one() {
    local seq=$1 bag=$2 cfg=$3
    timeout 900 "$OV" "_config_path:=$cfg" "_path_bag:=$bag" \
        "_bag_start:=0" "_bag_durr:=-1" "_verbosity:=SILENT" \
        "_save_total_state:=true" "_filepath_est:=$OUT/ovodom/ov_${seq}_est.txt" \
        "_filepath_std:=$OUT/ovodom/ov_${seq}_std.txt" > "ov_${seq}.log" 2>&1
    echo "$seq node_exit=$?"
    echo "$seq est_rows=$(wc -l < ov_${seq}_est.txt 2>/dev/null || echo 0)"
}

run_one MH_01 /workspace/EuROC/MH_01_easy.bag "$CFG"
run_one MH_02 /workspace/euroc_bags/MH_02_easy.bag "$CFG"
run_one MH_03 /workspace/euroc_bags/MH_03_medium.bag "$CFG"
run_one MH_04 /workspace/euroc_bags/MH_04_difficult.bag "$CFG"
run_one MH_05 /workspace/euroc_bags/MH_05_difficult.bag "$CFG"
run_one V1_01 /workspace/euroc_bags/V1_01_easy.bag "$CFG"
run_one V1_02 /workspace/euroc_bags/V1_02_medium.bag "$CFG"
run_one V1_03 /workspace/euroc_bags/V1_03_difficult.bag "$CFG"
run_one circle /workspace/circle.bag "$KCFG"
run_one infinite /workspace/ovrun/infinite.bag "$KCFG"
echo ALL_DONE
