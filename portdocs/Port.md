# VIO Pipeline C++ Porting Guide

This document acts as the master lookup index mapping the original Python VIO pipeline (`vio_pipeline_v2`) to the high-performance C++17 implementation.

## File Mapping Index

| Python Source File | C++ Header File | C++ Source File | Description |
| :--- | :--- | :--- | :--- |
| `type/quat_ops.py` | [type/quat_ops.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/type/quat_ops.hpp) | [type/quat_ops.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/type/quat_ops.cpp) | JPL quaternion math (skew, multiplication, exp/log maps, SO3/SE3) |
| `type/Variable.py` | [type/type.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/type/type.hpp) | [type/type.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/type/type.cpp) | EKF state variable representations (IMU, Clone, Landmark) |
| `core/cam/Camera.py` | [core/cam.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/core/cam.hpp) | [core/cam.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/core/cam.cpp) | Camera projection models (RADTAN, EQUIDISTANT) |
| `core/feature/FeatureDatabase.py` | [core/feature.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/core/feature.hpp) | [core/feature.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/core/feature.cpp) | Zero-heap flat feature database, Linear DLT / Levenberg-Marquardt triangulation |
| `initialize/static.py` | [initialize/initialization.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/initialize/initialization.hpp) | [initialize/initialization.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/initialize/initialization.cpp) | Static inertial alignment, gravity vector estimation, bias calculation |
| `msckf/State.py` | [msckf/state.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/state.hpp) | [msckf/state.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/state.cpp) | Unified state structure managing active EKF variables |
| `msckf/state_helper.py` | [msckf/state_helper.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/state_helper.hpp) | [msckf/state_helper.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/state_helper.cpp) | Schur complement, covariance propagation, and EKF updates |
| `msckf/Propagator.py` | [msckf/propagator.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/propagator.hpp) | [msckf/propagator.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/propagator.cpp) | RK4 and analytical EKF covariance/mean propagation |
| `msckf/updaters/` | [msckf/updaters.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/updaters.hpp) | [msckf/updaters.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/updaters.cpp) | Visual MSCKF batch updater, SLAM landmark updater, and ZUPT |
| `msckf/VioManager.py` | [msckf/vio_manager.hpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/vio_manager.hpp) | [msckf/vio_manager.cpp](file:///home/nikhil/dev/vins/Open_vins_python/vio_pipeline_cpp/msckf/vio_manager.cpp) | Master VIO pipeline manager handling IMU and camera queues |

---

## Porting Details

### 1. JPL Quaternions & Rotations
Python arrays were replaced with optimized Eigen vectors and matrices.
* All JPL multiplication formulas in `type/quat_ops.py` were translated to `Eigen::Vector4d` operations in `quat_ops.cpp`.
* JPL quaternion layout is `[q_x, q_y, q_z, q_w]`, which maps to `Eigen::Vector4d`.
* Nominal orientations utilize the right-hand JPL convention:
  $$\mathbf{R}(\bar{q}) = (2 q_w^2 - 1)\mathbf{I}_3 - 2 q_w \lfloor \mathbf{q}_\times \rfloor + 2 \mathbf{q}\mathbf{q}^T$$

### 2. State & Variable Porting
Python's class-based dynamic types in `type/Variable.py` were translated to a unified `type::Variable` struct.
* Getter methods (`Rot()`, `pos()`, `vel()`) were implemented to return views of nominal variables using `Eigen::Map` casts on raw double buffers.
* Retraction operations (boxplus/boxminus) utilize JPL quaternion multiplication and vector additions on local tangent coordinates.

### 3. Feature Tracker & Triangulation
Python's `scipy.optimize` Levenberg-Marquardt solver was ported to a lightweight C++ solver in `core/feature.cpp`.
* Zero heap allocation: we use fixed stack matrices `Eigen::Matrix<double, Dynamic, Dynamic>` with small compile-time limits.
* Linear DLT is used for initial landmark triangulation, followed by 1D depth bearing optimization.

### 4. MSCKF Covariance Propagation & Updates
* **Analytical Propagation**: Dispatches transitions using pre-computed Jacobian integrations.
* **Batch Visual Updates**: Nullspace projection ($Q_2^T \mathbf{r}$) is executed using `Eigen::HouseholderQR` to compress the visual residual block, eliminating landmark variables from the EKF state update step.
