/**
 * benchmark_pipeline_full.cpp
 *
 * Production-level end-to-end VIO pipeline comparison:
 *   [A] Our Ported DOD C++17 VioManager  (msckf/vio_manager.hpp)
 *   [B] Official Open_VINS EKF classes   (ov_msckf::State + Propagator + UpdaterMSCKF + InertialInitializer)
 *
 * Both run the SAME synthetic VIO data:
 *   - 61 IMU readings @ 100 Hz over 0.60 s
 *   - 30 features tracked across 7 camera frames @ 10 Hz
 *   - Static IMU initializer with 0.50-second window
 *
 * Timing reported per full run (0.60-second timeline).
 * Python result is printed from an external pre-run (benchmark_full.py already executed).
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

// ---------- Our DOD port ----------
#include "msckf/vio_manager.hpp"
#include "tests/test_data.hpp"

// ---------- Official Open_VINS ----------
#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "init/InertialInitializer.h"
#include "init/InertialInitializerOptions.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "state/StateOptions.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterOptions.h"
#include "utils/NoiseManager.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

// ============================================================
// Helpers to build OV ImuData / Feature from our test data
// ============================================================
static std::vector<ov_core::ImuData> build_ov_imu() {
  std::vector<ov_core::ImuData> readings;
  readings.reserve(tests::NUM_IMU_READINGS);
  for (int i = 0; i < tests::NUM_IMU_READINGS; ++i) {
    ov_core::ImuData d;
    d.timestamp = tests::imu_readings[i].timestamp;
    d.wm << tests::imu_readings[i].wm[0], tests::imu_readings[i].wm[1], tests::imu_readings[i].wm[2];
    d.am << tests::imu_readings[i].am[0], tests::imu_readings[i].am[1], tests::imu_readings[i].am[2];
    readings.push_back(d);
  }
  return readings;
}

// Build per-frame feature observations from our test_data global features
// Returns a map: timestamp -> vector of shared_ptr<ov_core::Feature>
static std::vector<std::pair<double, std::vector<std::shared_ptr<ov_core::Feature>>>> build_ov_frames() {
  std::vector<std::pair<double, std::vector<std::shared_ptr<ov_core::Feature>>>> frames;
  for (int fi = 0; fi < tests::NUM_FRAMES; ++fi) {
    double ts = fi * 0.10;
    std::vector<std::shared_ptr<ov_core::Feature>> frame_feats;
    for (int t = 0; t < tests::NUM_TRACKS; ++t) {
      const core::Feature& gf = tests::features[t];
      auto ov_feat = std::make_shared<ov_core::Feature>();
      ov_feat->featid = gf.featid;
      for (int m = 0; m < gf.num_measurements; ++m) {
        if (gf.measurements[m].timestamp <= ts) {
          int cam_id = gf.measurements[m].cam_id;
          double mts = gf.measurements[m].timestamp;
          Eigen::Vector2f uv(gf.measurements[m].uv[0], gf.measurements[m].uv[1]);
          Eigen::Vector2f uv_norm(gf.measurements[m].uv_norm[0], gf.measurements[m].uv_norm[1]);
          if (ov_feat->timestamps.find(cam_id) == ov_feat->timestamps.end()) {
            ov_feat->timestamps[cam_id] = {};
            ov_feat->uvs[cam_id] = {};
            ov_feat->uvs_norm[cam_id] = {};
          }
          ov_feat->timestamps[cam_id].push_back(mts);
          ov_feat->uvs[cam_id].push_back(uv);
          ov_feat->uvs_norm[cam_id].push_back(uv_norm);
        }
      }
      if (!ov_feat->timestamps.empty()) {
        frame_feats.push_back(ov_feat);
      }
    }
    frames.emplace_back(ts, std::move(frame_feats));
  }
  return frames;
}

// ============================================================
// A: DOD C++ Port benchmark
// ============================================================
static double bench_dod(int num_iters) {
  msckf::VioManagerOptions params;
  params.init_opt.init_window_time = 0.5;
  params.init_opt.init_imu_thresh = 2.0;
  params.init_opt.init_max_disparity = 2.0;
  params.init_opt.gravity_mag = 9.81;
  params.zupt_max_velocity = -1.0;
  params.num_cameras = 1;
  double cam_vals[8] = {460.0, 460.0, 320.0, 240.0, 0.01, -0.02, 0.001, -0.002};
  core::init_camera(params.cam_models[0], core::CameraModelType::RADTAN, 640, 480, cam_vals);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int run = 0; run < num_iters; ++run) {
    msckf::VioManagerData vio;
    msckf::init_vio_manager(vio, params);
    int next_imu_idx = 0;
    int next_frame_idx = 0;
    double next_frame_time = 0.0;
    for (int step = 0; step <= 60; ++step) {
      double ct = step * 0.01;
      while (next_imu_idx < tests::NUM_IMU_READINGS && tests::imu_readings[next_imu_idx].timestamp <= ct) {
        msckf::feed_measurement_imu(vio, tests::imu_readings[next_imu_idx]);
        next_imu_idx++;
      }
      if (std::abs(ct - next_frame_time) < 1e-5 && next_frame_idx < tests::NUM_FRAMES) {
        core::Feature frame_tracks[tests::NUM_TRACKS];
        int track_count = 0;
        for (int t = 0; t < tests::NUM_TRACKS; ++t) {
          const core::Feature& gf = tests::features[t];
          core::Feature af = gf;
          af.num_measurements = 0;
          for (int m = 0; m < gf.num_measurements; ++m) {
            if (gf.measurements[m].timestamp <= ct)
              af.measurements[af.num_measurements++] = gf.measurements[m];
          }
          if (af.num_measurements > 0)
            frame_tracks[track_count++] = af;
        }
        msckf::feed_measurement_camera_tracks(vio, ct, frame_tracks, track_count);
        next_frame_idx++;
        next_frame_time += 0.10;
      }
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

// ============================================================
// B: Official Open_VINS EKF benchmark
// ============================================================
static double bench_openvins(int num_iters) {
  // Pre-build imu data and frame observations once
  const auto ov_imu_readings = build_ov_imu();
  const auto ov_frames = build_ov_frames();

  // State options
  ov_msckf::StateOptions state_opt;
  state_opt.do_fej = true;
  state_opt.max_clone_size = 11;
  state_opt.num_cameras = 1;
  state_opt.imu_avg = false;
  state_opt.use_rk4_integration = true;

  // Noise
  ov_msckf::NoiseManager noises;
  noises.sigma_w = 0.005;
  noises.sigma_a = 0.010;
  noises.sigma_wb = 0.001;
  noises.sigma_ab = 0.002;
  noises.sigma_w_2 = noises.sigma_w * noises.sigma_w;
  noises.sigma_a_2 = noises.sigma_a * noises.sigma_a;
  noises.sigma_wb_2 = noises.sigma_wb * noises.sigma_wb;
  noises.sigma_ab_2 = noises.sigma_ab * noises.sigma_ab;

  // Camera intrinsics value vector
  Eigen::Matrix<double, 8, 1> cam_calib;
  cam_calib << 460.0, 460.0, 320.0, 240.0, 0.01, -0.02, 0.001, -0.002;

  // Updater options
  ov_msckf::UpdaterOptions upd_opt;
  upd_opt.sigma_pix = 1.0;
  upd_opt.chi2_multipler = 5.0;

  // Initializer options
  ov_init::InertialInitializerOptions init_opt;
  init_opt.init_window_time = 0.50;
  init_opt.init_imu_thresh = 2.0;
  init_opt.init_max_disparity = 2.0;
  init_opt.gravity_mag = 9.81;
  init_opt.init_dyn_use = false;

  // Silence Open_VINS output during benchmark
  ov_core::Printer::setPrintLevel(ov_core::Printer::PrintLevel::SILENT);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int run = 0; run < num_iters; ++run) {
    // Fresh state each run
    auto state = std::make_shared<ov_msckf::State>(state_opt);

    // Set camera calibration on state
    auto cam_model = std::make_shared<ov_core::CamRadtan>(640, 480);
    cam_model->set_value(cam_calib);
    state->_cam_intrinsics_cameras[0] = cam_model;
    state->_cam_intrinsics[0] = std::make_shared<ov_type::Vec>(8);
    state->_cam_intrinsics[0]->set_value(cam_calib);

    // Propagator
    auto propagator = std::make_shared<ov_msckf::Propagator>(noises, init_opt.gravity_mag);

    // Feature database (shared between initializer and updater)
    auto db = std::make_shared<ov_core::FeatureDatabase>();

    // Initializer
    auto initializer = std::make_shared<ov_init::InertialInitializer>(init_opt, db);

    // MSCKF Updater
    ov_msckf::UpdaterMSCKF updater_msckf(upd_opt);

    bool initialized = false;

    int next_imu_idx = 0;
    int next_frame_idx = 0;
    double next_frame_time = 0.0;

    for (int step = 0; step <= 60; ++step) {
      double ct = step * 0.01;

      // Feed IMU
      while (next_imu_idx < (int)ov_imu_readings.size() && ov_imu_readings[next_imu_idx].timestamp <= ct) {
        const auto& imu = ov_imu_readings[next_imu_idx];
        if (!initialized) {
          initializer->feed_imu(imu);
          propagator->feed_imu(imu, -1.0);
        } else {
          propagator->feed_imu(imu, state->margtimestep() - 0.50);
        }
        next_imu_idx++;
      }

      // Feed camera frame
      if (std::abs(ct - next_frame_time) < 1e-5 && next_frame_idx < (int)ov_frames.size()) {
        const auto& [frame_ts, frame_feats] = ov_frames[next_frame_idx];

        // Populate feature database with this frame's observations
        for (auto& feat : frame_feats) {
          for (auto& [cam_id, times] : feat->timestamps) {
            for (int mi = 0; mi < (int)times.size(); ++mi) {
              db->update_feature(feat->featid, times[mi], cam_id,
                                 feat->uvs[cam_id][mi](0), feat->uvs[cam_id][mi](1),
                                 feat->uvs_norm[cam_id][mi](0), feat->uvs_norm[cam_id][mi](1));
            }
          }
        }

        if (!initialized) {
          // Try initialization
          double init_ts = 0.0;
          Eigen::MatrixXd init_cov;
          std::vector<std::shared_ptr<ov_type::Type>> init_order;
          bool success = initializer->initialize(init_ts, init_cov, init_order, state->_imu, false);
          if (success) {
            ov_msckf::StateHelper::set_initial_covariance(state, init_cov, init_order);
            state->_timestamp = init_ts;
            initialized = true;
            db->cleanup_measurements(init_ts);
          }
        } else {
          // Propagate + clone
          propagator->propagate_and_clone(state, ct);

          // Collect lost features for MSCKF update
          std::vector<std::shared_ptr<ov_core::Feature>> update_feats;
          for (auto& feat : frame_feats) {
            bool seen_now = false;
            auto it = feat->timestamps.find(0);
            if (it != feat->timestamps.end()) {
              for (auto t : it->second) {
                if (std::abs(t - ct) < 1e-6) { seen_now = true; break; }
              }
            }
            if (!seen_now) {
              // Feature is lost — get from database for update
              auto db_feat = db->get_feature(feat->featid, false);
              if (db_feat && db_feat->timestamps[0].size() >= 2)
                update_feats.push_back(db_feat);
            }
          }

          if (!update_feats.empty())
            updater_msckf.update(state, update_feats);

          // Marginalize oldest clone
          if ((int)state->_clones_IMU.size() > state->_options.max_clone_size)
            ov_msckf::StateHelper::marginalize_old_clone(state);

          // Clean DB
          db->cleanup_measurements(state->margtimestep());
        }
        next_frame_idx++;
        next_frame_time += 0.10;
      }
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

// ============================================================
// main
// ============================================================
int main() {
  std::cout << "=======================================================\n";
  std::cout << "  Production VIO Pipeline Benchmark\n";
  std::cout << "  Same 0.60s synthetic dataset: 61 IMU + 7 cam frames\n";
  std::cout << "=======================================================\n\n";

  // Warm up caches
  bench_dod(5);

  constexpr int N_DOD = 10000;
  constexpr int N_OV = 2000;

  std::cout << "[1/3] Python vio_pipeline_v2 (pre-measured via uv run benchmark_full.py):\n";
  std::cout << "      Avg: 55.46 ms/run  (100 runs)\n\n";

  std::cout << "[2/3] Our C++ DOD VioManager (" << N_DOD << " runs)...\n";
  double dod_total = bench_dod(N_DOD);
  double dod_avg_ms = (dod_total / N_DOD) * 1000.0;
  std::cout << "      Total: " << dod_total << " s | Avg: " << dod_avg_ms << " ms/run\n\n";

  std::cout << "[3/3] Official Open_VINS EKF (State+Propagator+UpdaterMSCKF+InertialInitializer, " << N_OV << " runs)...\n";
  double ov_total = bench_openvins(N_OV);
  double ov_avg_ms = (ov_total / N_OV) * 1000.0;
  std::cout << "      Total: " << ov_total << " s | Avg: " << ov_avg_ms << " ms/run\n\n";

  constexpr double python_ms = 55.46;
  std::cout << "=======================================================\n";
  std::cout << "  RESULTS SUMMARY\n";
  std::cout << "=======================================================\n";
  std::printf("  %-40s %10s  %8s\n", "Implementation", "Avg (ms/run)", "Speedup");
  std::printf("  %-40s %10s  %8s\n", "--------------------------------------", "----------", "-------");
  std::printf("  %-40s %10.3f  %8s\n", "Python vio_pipeline_v2 (baseline)", python_ms, "1.0x");
  std::printf("  %-40s %10.3f  %8.1fx\n", "Official Open_VINS (real EKF classes)", ov_avg_ms, python_ms / ov_avg_ms);
  std::printf("  %-40s %10.3f  %8.1fx\n", "Our C++ DOD VioManager", dod_avg_ms, python_ms / dod_avg_ms);
  std::printf("\n  DOD vs Open_VINS:  %.1fx %s\n",
              ov_avg_ms / dod_avg_ms,
              dod_avg_ms < ov_avg_ms ? "faster (DOD wins)" : "slower");
  std::cout << "=======================================================\n";

  return 0;
}
