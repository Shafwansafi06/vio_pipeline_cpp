#!/usr/bin/env bash
set -eo pipefail

# Run inside the ROS Noetic container under xvfb-run. A private ROS master
# keeps the capture reproducible and avoids interfering with other jobs.
source /opt/ros/noetic/setup.bash
set -u
export PYTHONPATH="/workspace${PYTHONPATH:+:${PYTHONPATH}}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11319}"
export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1

bag_path="${1:-/workspace/KAIST VIO dataset/circle/circle.bag}"
playback_rate="${2:-0.5}"
output_prefix="${3:-/tmp/rviz_python_circle_synced}"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

roscore -p 11319 >"${output_prefix}_roscore.log" 2>&1 &
pids+=("$!")
sleep 3

python3 -m vio_pipeline_v2.main \
  _config_path:=/workspace/vio_pipeline_v2/config/kaist_vio \
  _topic_imu:=/mavros/imu/data \
  _topic_cam0:=/camera/infra1/image_rect_raw \
  _topic_cam1:=/camera/infra2/image_rect_raw \
  >"${output_prefix}_python.log" 2>&1 &
pids+=("$!")

rosrun topic_tools relay /openvins_python/path /ov_msckf/pathimu >/dev/null 2>&1 &
pids+=("$!")
rosrun topic_tools relay /openvins_python/odomimu /ov_msckf/odomimu >/dev/null 2>&1 &
pids+=("$!")
rosrun topic_tools relay /openvins_python/feature_image /ov_msckf/trackhist >/dev/null 2>&1 &
pids+=("$!")

rviz -d /workspace/open_vins_official/ov_msckf/launch/display.rviz \
  >"${output_prefix}_rviz.log" 2>&1 &
pids+=("$!")
sleep 8

# Dismiss the ROS 1 end-of-life dialog and normalize saved window geometry.
xdotool key Return 2>/dev/null || true
sleep 2
window_id="$(xdotool search --name 'display.rviz - RViz' 2>/dev/null | tail -n 1 || true)"
if [[ -n "${window_id}" ]]; then
  xdotool windowmove "${window_id}" 10 10
  xdotool windowsize "${window_id}" 1280 900
fi

rosbag record -O "${output_prefix}_odom.bag" /openvins_python/odomimu \
  >/dev/null 2>&1 &
pids+=("$!")
sleep 2

rosbag play --clock -r "${playback_rate}" --duration=191.5 "${bag_path}" \
  >"${output_prefix}_rosbag.log" 2>&1 &
bag_pid="$!"
pids+=("${bag_pid}")

# At 0.5x playback these correspond to approximately 5 s, 96 s, and 187 s
# of bag time. The endpoint is repeated after queues drain to expose any
# post-input continuation or endpoint movement.
sleep 10
xwd -silent -root -out "${output_prefix}_begin.xwd"
echo CAPTURE_BEGIN
sleep 182
xwd -silent -root -out "${output_prefix}_middle.xwd"
echo CAPTURE_MIDDLE
sleep 182
xwd -silent -root -out "${output_prefix}_near_end.xwd"
echo CAPTURE_NEAR_END

wait "${bag_pid}" || true
sleep 30
xwd -silent -root -out "${output_prefix}_drained_end.xwd"
echo CAPTURE_DRAINED_END
