cmake_minimum_required(VERSION 3.10)
project(vio_pipeline_cpp VERSION 1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    msckf/updater_slam_helper.cpp
    msckf/hover_detector.cpp
    msckf/vio_manager.cpp
)

add_library(vio_pipeline STATIC ${VIO_LIB_SOURCES})
target_compile_definitions(vio_pipeline PUBLIC EIGEN_RUNTIME_NO_MALLOC)

# Mathematical equivalence verification test executable
add_executable(verify_math tests/verify_math.cpp)
target_link_libraries(verify_math PRIVATE vio_pipeline)
add_test(NAME verify_math COMMAND verify_math)

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
    add_library(vio_frontend STATIC core/tracker.cpp)
    target_include_directories(vio_frontend PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(vio_frontend PUBLIC vio_pipeline ${OpenCV_LIBS})
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
    target_include_directories(vio_rosbag_runner_euroc PRIVATE ${catkin_INCLUDE_DIRS})
    target_link_libraries(vio_rosbag_runner_euroc PRIVATE vio_frontend ${catkin_LIBRARIES})
endif()

# ============================================================================
# Optional official Open_VINS comparison (no ROS, no Ceres, no Track)
# ============================================================================
option(VIO_BUILD_OPENVINS_COMPARISON
       "Build the benchmark against a vendored official Open_VINS checkout"
       OFF)

if(VIO_BUILD_OPENVINS_COMPARISON)
set(OV_BASE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/open_vins_official")
if(NOT EXISTS "${OV_BASE}/ov_msckf/src/state/State.cpp")
    message(FATAL_ERROR
        "VIO_BUILD_OPENVINS_COMPARISON=ON requires official Open_VINS sources at ${OV_BASE}")
endif()

# Collect only the non-ROS, non-Ceres, non-sim, non-track source files
set(OV_CORE_SRCS
    ${OV_BASE}/ov_core/src/feat/Feature.cpp
    ${OV_BASE}/ov_core/src/feat/FeatureDatabase.cpp
    ${OV_BASE}/ov_core/src/feat/FeatureInitializer.cpp
    ${OV_BASE}/ov_core/src/types/Landmark.cpp
    ${OV_BASE}/ov_core/src/utils/print.cpp
)
set(OV_MSCKF_SRCS
    ${OV_BASE}/ov_msckf/src/state/Propagator.cpp
    ${OV_BASE}/ov_msckf/src/state/State.cpp
    ${OV_BASE}/ov_msckf/src/state/StateHelper.cpp
    ${OV_BASE}/ov_msckf/src/update/UpdaterHelper.cpp
    ${OV_BASE}/ov_msckf/src/update/UpdaterMSCKF.cpp
)
set(OV_INIT_SRCS
    ${OV_BASE}/ov_init/src/static/StaticInitializer.cpp
    ${OV_BASE}/ov_init/src/init/InertialInitializer.cpp
)

# OpenCV5 include path (system)
set(OV_OPENCV_INC /usr/include/opencv5)

# Boost math headers (header-only chi_squared)
find_package(Boost REQUIRED)

add_library(ov_ekf_core STATIC ${OV_CORE_SRCS} ${OV_MSCKF_SRCS} ${OV_INIT_SRCS})
target_include_directories(ov_ekf_core SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen
    ${OV_OPENCV_INC}
    ${Boost_INCLUDE_DIRS}
    ${OV_BASE}/ov_core/src
    ${OV_BASE}/ov_msckf/src
    ${OV_BASE}/ov_init/src
)
target_compile_options(ov_ekf_core PRIVATE
    -DUSING_ROS_FALSE
    -DNO_CERES_SUPPORT
)

# Inject OpenCV5 compat prefix header into each OV source file via COMPILE_FLAGS
# (avoids -include path with spaces which breaks GCC argument parsing)
set(OV_ALL_SRCS ${OV_CORE_SRCS} ${OV_MSCKF_SRCS} ${OV_INIT_SRCS})
set(OV_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ov_compat_prefix.hpp")
foreach(SRC ${OV_ALL_SRCS})
    set_source_files_properties(${SRC} PROPERTIES
        COMPILE_FLAGS "-include${OV_PREFIX} -include${OV_PREFIX}"
    )
endforeach()

# Link against OpenCV5 core (needed for Mat, etc. in FeatureHelper)
target_link_libraries(ov_ekf_core PUBLIC
    /usr/lib/libopencv_core.so
    /usr/lib/libopencv_calib.so
    /usr/lib/libopencv_imgproc.so
    /usr/lib/libopencv_3d.so
)

# Full production pipeline comparison: Python vs C++ DOD vs real Open_VINS
add_executable(benchmark_pipeline_full tests/benchmark_pipeline_full.cpp)
target_include_directories(benchmark_pipeline_full PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen
    ${OV_OPENCV_INC}
    ${Boost_INCLUDE_DIRS}
    ${OV_BASE}/ov_core/src
    ${OV_BASE}/ov_msckf/src
    ${OV_BASE}/ov_init/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(benchmark_pipeline_full PRIVATE vio_pipeline ov_ekf_core)
endif()
