#!/usr/bin/env bash
# P-02: build both estimators on the Orin inside ov_ros1_20_04 (Ubuntu 20.04
# aarch64, g++ 9.4, OpenCV 4.2.x) — the same toolchain as the x86 container,
# so the only variable vs the x86 numbers is the architecture.
# Run on the Orin host: bash ~/p02/build.sh
set -e
export ROS_DISTRO=noetic
echo "REDACTED" | sudo -S docker run --rm \
    -v "$HOME/p02:/p02" \
    ov_ros1_20_04:latest bash -c '
set -e
cd /p02/dod
source /opt/ros/noetic/setup.bash
mkdir -p build_ros && cd build_ros
cmake -DCMAKE_BUILD_TYPE=Release -DVIO_BUILD_ROS1=ON -DVIO_BUILD_OPENCV_FRONTEND=ON .. > /p02/dod_cmake.log 2>&1
make vio_rosbag_runner_euroc vio_rosbag_runner -j6 > /p02/dod_make.log 2>&1
echo DOD_BUILD_OK
cd /p02
if [ ! -d openvins_ws/devel ]; then
  mkdir -p openvins_ws/src
  ln -sfn /p02/open_vins_official /p02/openvins_ws/src/open_vins
  cp -f /opt/ros/noetic/share/catkin/cmake/toplevel.cmake openvins_ws/src/CMakeLists.txt 2>/dev/null || true
fi
cd /p02/openvins_ws
catkin_make -DCMAKE_BUILD_TYPE=Release -j6 > /p02/ov_make.log 2>&1
echo OV_BUILD_OK
ls devel/lib/ov_msckf/ros1_serial_msckf
' 2>&1 | grep -vE "^\[sudo\]"
