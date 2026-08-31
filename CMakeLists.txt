cmake_minimum_required(VERSION 3.10)
project(vio_pipeline_cpp VERSION 1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Floating-point parity with official Open_VINS.
#
# Official builds with these exact flags (open_vins/ov_msckf/CMakeLists.txt).
# Bit-identical results are impossible unless we compile with the same set:
# -fno-signed-zeros in particular alters FP semantics. -ffp-contract=off is
# ours, and is a strengthening -- it forbids FMA contraction, which otherwise
# depends on the target ISA and would make results machine-dependent.
# Deliberately absent: -ffast-math, -march=native, -Ofast.
# ---------------------------------------------------------------------------
set(VIO_PARITY_FLAGS
    -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno
    -funroll-loops -ffp-contract=off)
option(VIO_PARITY_FP "Compile with official Open_VINS's FP flags" ON)
if(VIO_PARITY_FP)
    add_compile_options(${VIO_PARITY_FLAGS})
endif()

# Increase Eigen's stack allocation limit to 8MB to allow large static covariance matrices on stack
add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=8388608)

# Vendorized Eigen Header Search Path
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Library Sources
set(VIO_LIB_SOURCES
    arena.cpp
    type/quat_ops.cpp
    type/type.cpp
    core/cam.cpp
    core/feature.cpp
    initialize/initialization.cpp
    msckf/state.cpp
    msckf/state_helper.cpp
    msckf/propagator.cpp
    msckf/updaters.cpp
    msckf/updater_mixed.cpp
    msckf/updater_schur.cpp
    msckf/updater_slam_helper.cpp
    msckf/hover_detector.cpp
    msckf/vio_manager.cpp
)

add_library(vio_pipeline STATIC ${VIO_LIB_SOURCES})
target_compile_definitions(vio_pipeline PUBLIC EIGEN_RUNTIME_NO_MALLOC)

# Mathematical equivalence verification test executable
add_executable(verify_math tests/verify_math.cpp)
target_link_libraries(verify_math PRIVATE vio_pipeline)
# NOT registered with ctest, deliberately. Two independent problems:
#  1. Its expected values come from the Python port, which the parity manual
#     documents as never validated against official -- so agreement proves
#     only that both share the same bugs.
#  2. The pipeline does not initialize on its synthetic vectors, so the state
#     stays identity/zero and neither comparison is ever exercised. This was
#     invisible because every check was an assert() and Release defines NDEBUG;
#     the test reported "passed" while verifying nothing. Confirmed present in
#     the pre-parity baseline, so it is long-standing, not a new regression.
# The executable is kept buildable for debugging. Its role is taken over by the
# parity suite (bitdiff_*), which diffs against official rather than Python.

add_executable(verify_noalloc_propagation tests/verify_noalloc_propagation.cpp)
target_link_libraries(verify_noalloc_propagation PRIVATE vio_pipeline)
add_test(NAME verify_noalloc_propagation COMMAND verify_noalloc_propagation)

add_executable(verify_clone_invariants tests/verify_clone_invariants.cpp)
target_link_libraries(verify_clone_invariants PRIVATE vio_pipeline)
add_test(NAME verify_clone_invariants COMMAND verify_clone_invariants)

add_executable(verify_feature_lifetime tests/verify_feature_lifetime.cpp)
target_link_libraries(verify_feature_lifetime PRIVATE vio_pipeline)
add_test(NAME verify_feature_lifetime COMMAND verify_feature_lifetime)

add_executable(verify_slam_jacobians tests/verify_slam_jacobians.cpp)
target_link_libraries(verify_slam_jacobians PRIVATE vio_pipeline)
add_test(NAME verify_slam_jacobians COMMAND verify_slam_jacobians)

add_executable(verify_parallax_noise tests/verify_parallax_noise.cpp)
target_link_libraries(verify_parallax_noise PRIVATE vio_pipeline)
add_test(NAME verify_parallax_noise COMMAND verify_parallax_noise)

add_executable(verify_dynamic_init tests/verify_dynamic_init.cpp)
target_link_libraries(verify_dynamic_init PRIVATE vio_pipeline)
add_test(NAME verify_dynamic_init COMMAND verify_dynamic_init)

# Benchmark executable
add_executable(benchmark tests/benchmark.cpp)
target_link_libraries(benchmark PRIVATE vio_pipeline)

# Open_VINS vs DOD benchmark executable
add_executable(compare_openvins tests/compare_openvins.cpp)
target_link_libraries(compare_openvins PRIVATE vio_pipeline)

# Full VIO Pipeline benchmark executable
add_executable(benchmark_full tests/benchmark_full.cpp)
target_link_libraries(benchmark_full PRIVATE vio_pipeline)

# Optional production image frontend. OpenCV is deliberately isolated from the
# allocation-free estimator library so math-only deployments keep no GUI/vision
# dependency and the ROS transport boundary owns third-party work buffers.
option(VIO_BUILD_OPENCV_FRONTEND "Build the bounded stereo KLT frontend" OFF)
if(VIO_BUILD_OPENCV_FRONTEND)
    find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc video)
    # core/undistort_cv.cpp lives here rather than in vio_pipeline because
    # official implements undistortion with cv::undistortPoints; matching it
    # bit-for-bit means making the same call. See that file's header comment.
    add_library(vio_frontend STATIC core/tracker.cpp core/undistort_cv.cpp)
    target_include_directories(vio_frontend PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(vio_frontend PUBLIC vio_pipeline ${OpenCV_LIBS})
endif()

# ROS-free EuRoC runner over the dataset's ASL folder layout. Same estimator,
# same config (tools/euroc_options.hpp), no ROS -- this is what makes the
# EuRoC ATE loop runnable on a machine without a ROS install.
option(VIO_BUILD_ASL_RUNNER "Build the ROS-free EuRoC ASL runner" OFF)
if(VIO_BUILD_ASL_RUNNER)
    if(NOT VIO_BUILD_OPENCV_FRONTEND)
        message(FATAL_ERROR "VIO_BUILD_ASL_RUNNER requires VIO_BUILD_OPENCV_FRONTEND=ON")
    endif()
    find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs video)
    add_executable(dod_asl_runner tools/dod_asl_runner.cpp)
    target_link_libraries(dod_asl_runner PRIVATE vio_frontend ${OpenCV_LIBS})
    # TUM VI (Schubert et al., IROS 2018) ASL-format runner: independent test
    # dataset, calibrated with the dataset's own pinhole-equi-512 release.
    add_executable(tumvi_asl_runner tools/tumvi_asl_runner.cpp)
    target_link_libraries(tumvi_asl_runner PRIVATE vio_frontend ${OpenCV_LIBS})
endif()

option(VIO_BUILD_ROS1 "Build the ROS1 offline bag runner and RViz publishers" OFF)
if(VIO_BUILD_ROS1)
    if(NOT VIO_BUILD_OPENCV_FRONTEND)
        message(FATAL_ERROR "VIO_BUILD_ROS1 requires VIO_BUILD_OPENCV_FRONTEND=ON")
    endif()
    find_package(catkin REQUIRED COMPONENTS
        roscpp rosbag sensor_msgs geometry_msgs nav_msgs tf2_ros cv_bridge)
    add_executable(vio_rosbag_runner ros/vio_rosbag_runner.cpp)
    target_include_directories(vio_rosbag_runner PRIVATE ${catkin_INCLUDE_DIRS})
    target_link_libraries(vio_rosbag_runner PRIVATE vio_frontend ${catkin_LIBRARIES})

    add_executable(vio_rosbag_runner_euroc ros/vio_rosbag_runner_euroc.cpp)

    # Track-injection variant: DOD's estimator driven by official's dumped
    # feature tracks, to separate frontend error from estimator error.
    add_executable(vio_rosbag_runner_euroc_inject ros/vio_rosbag_runner_euroc_inject.cpp)
    target_include_directories(vio_rosbag_runner_euroc_inject PRIVATE ${catkin_INCLUDE_DIRS})
    target_link_libraries(vio_rosbag_runner_euroc_inject PRIVATE vio_frontend ${catkin_LIBRARIES})
    target_include_directories(vio_rosbag_runner_euroc PRIVATE ${catkin_INCLUDE_DIRS})
    target_link_libraries(vio_rosbag_runner_euroc PRIVATE vio_frontend ${catkin_LIBRARIES})

    # FGI Masala Stereo-Visual-Inertial Dataset 2021 (high-altitude drone).
    add_executable(vio_rosbag_runner_mid_altitude ros/vio_rosbag_runner_mid_altitude.cpp)
    target_include_directories(vio_rosbag_runner_mid_altitude PRIVATE ${catkin_INCLUDE_DIRS})
    target_link_libraries(vio_rosbag_runner_mid_altitude PRIVATE vio_frontend ${catkin_LIBRARIES})
endif()

# ============================================================================
# Parity harness: link official Open_VINS in as a numeric ORACLE.
#
# TEST-ONLY. These targets are never linked into the shipped `vio_pipeline`
# library -- the whole point is to diff against official, not to derive from
# it. Verify with:  nm libvio_pipeline.a | grep ov_  (must be empty).
# ============================================================================
option(VIO_BUILD_PARITY "Build bit-exactness tests against official Open_VINS" OFF)

if(VIO_BUILD_PARITY)
set(OV_BASE "" CACHE PATH "Path to an official Open_VINS checkout")
if(NOT OV_BASE)
    set(OV_BASE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/open_vins_official")
endif()
if(NOT EXISTS "${OV_BASE}/ov_msckf/src/state/State.cpp")
    message(FATAL_ERROR "VIO_BUILD_PARITY=ON requires official Open_VINS sources; "
                        "set -DOV_BASE=/path/to/open_vins (looked in ${OV_BASE})")
endif()
message(STATUS "Parity oracle: ${OV_BASE}")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d)
find_package(Boost REQUIRED COMPONENTS system filesystem date_time)

set(OV_ORACLE_SRCS
    ${OV_BASE}/ov_core/src/feat/Feature.cpp
    ${OV_BASE}/ov_core/src/feat/FeatureDatabase.cpp
    ${OV_BASE}/ov_core/src/feat/FeatureInitializer.cpp
    ${OV_BASE}/ov_core/src/types/Landmark.cpp
    ${OV_BASE}/ov_core/src/utils/print.cpp
    ${OV_BASE}/ov_msckf/src/state/Propagator.cpp
    ${OV_BASE}/ov_msckf/src/state/State.cpp
    ${OV_BASE}/ov_msckf/src/state/StateHelper.cpp
    ${OV_BASE}/ov_msckf/src/update/UpdaterHelper.cpp
    ${OV_BASE}/ov_msckf/src/update/UpdaterMSCKF.cpp
    ${OV_BASE}/ov_msckf/src/update/UpdaterSLAM.cpp
    ${OV_BASE}/ov_msckf/src/update/UpdaterZeroVelocity.cpp
    ${OV_BASE}/ov_init/src/static/StaticInitializer.cpp
)

add_library(ov_oracle STATIC ${OV_ORACLE_SRCS})
# Official is C++14; the oracle must be built exactly as official builds it.
set_target_properties(ov_oracle PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF)
# Same vendored Eigen as DOD -- a different Eigen version means different
# kernels means different bits, which would invalidate every comparison.
target_include_directories(ov_oracle SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${OV_BASE}/ov_core/src
    ${OV_BASE}/ov_msckf/src
    ${OV_BASE}/ov_init/src
)
target_link_libraries(ov_oracle PUBLIC ${OpenCV_LIBS} ${Boost_LIBRARIES})
# Official picks <cassert> up transitively from its ROS/catkin include chain,
# which we do not have. Force-include rather than patch their sources -- the
# oracle must stay byte-for-byte the upstream checkout.
target_compile_options(ov_oracle PRIVATE -include cassert)

# One test per parity stage, so a failure names the function.
foreach(STAGE bitdiff_smoke bitdiff_chi2 bitdiff_cam bitdiff_prop bitdiff_tri bitdiff_update bitdiff_ekf bitdiff_jac)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/${STAGE}.cpp)
        # undistort_cv.cpp is compiled in directly rather than depending on
        # vio_frontend, so parity tests build without VIO_BUILD_OPENCV_FRONTEND.
        # OpenCV is already on the link line via ov_oracle.
        add_executable(${STAGE} tests/${STAGE}.cpp core/undistort_cv.cpp)
        target_link_libraries(${STAGE} PRIVATE vio_pipeline ov_oracle)
        add_test(NAME ${STAGE} COMMAND ${STAGE})
    endif()
endforeach()
endif()
