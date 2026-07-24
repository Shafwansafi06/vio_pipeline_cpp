#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 {python|openvins} BAG OUTPUT_DIR RATE [DURATION]" >&2
  exit 2
fi

implementation=$1
bag_path=$2
output_dir=$3
play_rate=$4
duration=${5:-}

source /opt/ros/noetic/setup.bash
mkdir -p "$output_dir"

node_pid=""
record_pid=""
sampler_pid=""

cleanup() {
  set +e
  [[ -n "$sampler_pid" ]] && kill "$sampler_pid" 2>/dev/null
  [[ -n "$record_pid" ]] && kill -INT "$record_pid" 2>/dev/null
  [[ -n "$record_pid" ]] && wait "$record_pid" 2>/dev/null
  [[ -n "$node_pid" ]] && kill -INT "$node_pid" 2>/dev/null
  [[ -n "$node_pid" ]] && wait "$node_pid" 2>/dev/null
}
trap cleanup EXIT INT TERM

roscore >"$output_dir/roscore.log" 2>&1 &
sleep 3

case "$implementation" in
  python)
    export PYTHONPATH="/workspace:${PYTHONPATH:-}"
    python3 -m vio_pipeline_v2.main \
      _config_path:=/workspace/vio_pipeline_v2/config/kaist_vio \
      _topic_imu:=/mavros/imu/data \
      _topic_cam0:=/camera/infra1/image_rect_raw \
      _topic_cam1:=/camera/infra2/image_rect_raw \
      >"$output_dir/estimator.log" 2>&1 &
    node_pid=$!
    node_name=/openvins_python
    estimate_topic=/openvins_python/odom
    ;;
  openvins)
    source /tmp/openvins_catkin_ws/devel/setup.bash
    roslaunch ov_msckf subscribe.launch \
      config:=kaist_vio \
      config_path:=/workspace/vio_pipeline_v2/config/kaist_vio/estimator_config.yaml \
      dobag:=false \
      >"$output_dir/estimator.log" 2>&1 &
    node_pid=$!
    node_name=/ov_msckf
    estimate_topic=/ov_msckf/poseimu
    ;;
  *)
    echo "unknown implementation: $implementation" >&2
    exit 2
    ;;
esac

for _ in $(seq 1 30); do
  if rosnode list 2>/dev/null | grep -qx "$node_name"; then
    break
  fi
  sleep 1
done
rosnode list | grep -qx "$node_name"

if [[ "$implementation" == openvins ]]; then
  estimator_pid=$(rosnode info "$node_name" | awk '/^Pid:/ {print $2}')
else
  estimator_pid=$node_pid
fi
[[ "$estimator_pid" =~ ^[0-9]+$ ]]

echo "wall_seconds monotonic_ticks user_ticks system_ticks rss_kib hwm_kib threads" \
  >"$output_dir/resources.txt"
(
  start_seconds=$SECONDS
  while [[ -r "/proc/$estimator_pid/stat" ]]; do
    read -r user_ticks system_ticks < <(awk '{print $14, $15}' "/proc/$estimator_pid/stat")
    read -r rss_kib hwm_kib threads < <(
      awk '
        /^VmRSS:/ {rss=$2}
        /^VmHWM:/ {hwm=$2}
        /^Threads:/ {thr=$2}
        END {print rss+0, hwm+0, thr+0}
      ' "/proc/$estimator_pid/status"
    )
    printf '%d %s %s %s %s %s %s\n' \
      "$((SECONDS-start_seconds))" "$(date +%s%N)" "$user_ticks" "$system_ticks" \
      "$rss_kib" "$hwm_kib" "$threads"
    sleep 1
  done
) >>"$output_dir/resources.txt" &
sampler_pid=$!

rosbag record --buffsize=1024 --chunksize=768 \
  -O "$output_dir/output.bag" /pose_transformed "$estimate_topic" \
  >"$output_dir/record.log" 2>&1 &
record_pid=$!
sleep 3

bag_args=(play -r "$play_rate")
if [[ -n "$duration" ]]; then
  bag_args+=(--duration="$duration")
fi
bag_args+=("$bag_path")

start_ns=$(date +%s%N)
rosbag "${bag_args[@]}" >"$output_dir/play.log" 2>&1
end_ns=$(date +%s%N)
printf '%s\n' "$start_ns" >"$output_dir/start_ns.txt"
printf '%s\n' "$end_ns" >"$output_dir/end_ns.txt"

# Allow asynchronous camera workers and rosbag recorder buffers to drain.
sleep 15

kill -INT "$record_pid" 2>/dev/null || true
wait "$record_pid" 2>/dev/null || true
record_pid=""

rosbag info --yaml "$output_dir/output.bag" >"$output_dir/output_info.yaml"
echo "benchmark completed: $implementation"
