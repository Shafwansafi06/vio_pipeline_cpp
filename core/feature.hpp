#pragma once
#include <Eigen/Core>

namespace core {

struct ClonePose {
    Eigen::Matrix3d R;
    Eigen::Vector3d p;
};

struct ClonesCamera {
    int num_clones = 0;
    double timestamps[20];
    ClonePose poses[2][20]; // [camera_id][clone_index]
};

const ClonePose* find_clone_pose(const ClonesCamera& clones, int cam_id, double ts);

struct FeatureMeasurement {
    int cam_id;
    double timestamp;
    Eigen::Vector2d uv;
    Eigen::Vector2d uv_norm;
};

struct Feature {
    int featid = -1;
    bool to_delete = false;
    FeatureMeasurement measurements[48];
    int num_measurements = 0;
    
    // Triangulated coordinates
    Eigen::Vector3d p_FinG = Eigen::Vector3d::Zero();
    int anchor_cam_id = -1;
    double anchor_clone_timestamp = 0.0;
    Eigen::Vector3d p_FinA = Eigen::Vector3d::Zero();
};

struct FeatureDatabase {
    Feature features[2048];
    std::size_t count = 0;
};

// Database functions
int find_feature_index(const FeatureDatabase& db, int feat_id);
Feature* get_feature(FeatureDatabase& db, int feat_id);
bool get_feature_clone(const FeatureDatabase& db, int feat_id, Feature& feat_out);
void update_feature(FeatureDatabase& db, int feat_id, double timestamp, int cam_id, double u, double v, double u_n, double v_n);
void cleanup_db(FeatureDatabase& db);
void cleanup_db_measurements(FeatureDatabase& db, double timestamp);
void cleanup_db_measurements_exact(FeatureDatabase& db, double timestamp);
double get_oldest_db_timestamp(const FeatureDatabase& db);

// Feature measurement filtering
void clean_old_measurements(Feature& feat, const double* valid_times, int num_valid);
void clean_invalid_measurements(Feature& feat, const double* invalid_times, int num_invalid);
void clear_older_measurements(Feature& feat, double timestamp);

// Triangulation Options
struct FeatureInitializerOptions {
    bool triangulate_1d = false;
    bool refine_features = true;
    double max_cond_number = 10000.0;
    double min_dist = 0.25;
    double max_dist = 40.0;
    double max_baseline = 40.0;
    double init_lamda = 1e-3;
    double max_lamda = 1e10;
    int max_runs = 10;
    double min_dx = 1e-6;
    double min_dcost = 1e-6;
    double lam_mult = 10.0;
};

// Triangulation Math
bool single_triangulation(Feature& feat, const ClonesCamera& clones, const FeatureInitializerOptions& options);
bool single_triangulation_1d(Feature& feat, const ClonesCamera& clones, const FeatureInitializerOptions& options);
bool single_gaussnewton(Feature& feat, const ClonesCamera& clones, const FeatureInitializerOptions& options);
double compute_error(const ClonesCamera& clones, const Feature& feat, double alpha, double beta, double rho);

bool compute_disparity(const FeatureDatabase& db, double newest_time, double oldest_time, double& mean, double& var, int& count);
bool compute_disparity_two_frames(const FeatureDatabase& db, double time0, double time1, double& mean, double& var, int& count);

} // namespace core
