#pragma once
#include <Eigen/Core>

namespace core {

struct ImuData {
    double timestamp = 0.0;
    Eigen::Vector3d wm = Eigen::Vector3d::Zero();
    Eigen::Vector3d am = Eigen::Vector3d::Zero();
    
    bool operator<(const ImuData& other) const {
        return timestamp < other.timestamp;
    }
};

struct CameraData {
    double timestamp = 0.0;
    int sensor_ids[2] = {-1, -1};
    int num_sensors = 0;
    // Images and masks are used by front-end tracking, but EKF estimator math uses track coordinates directly.
};

} // namespace core
