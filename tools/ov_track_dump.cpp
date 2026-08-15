// Official OpenVINS counterpart of tools/dod_track_dump.cpp: feeds the same
// EuRoC ASL stereo images through ov_core's TrackKLT and dumps the surviving
// cam0 track ids each frame. Official's source is NOT modified -- TrackBase
// exposes get_last_ids() publicly, which is the exact state DOD's dump reads.
//
// Tracker parameters below mirror open_vins/config/euroc_mav/estimator_config.yaml.
//
//   ./ov_track_dump <mav0_dir> <out.txt> [max_frames]

#include "cam/CamRadtan.h"
#include "track/TrackKLT.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kImageWidth = 752;
constexpr int kImageHeight = 480;

std::vector<std::pair<double, std::string>> read_index(const std::string &cam_dir) {
  std::vector<std::pair<double, std::string>> out;
  std::FILE *file = std::fopen((cam_dir + "/data.csv").c_str(), "r");
  if (file == nullptr)
    return out;
  char line[512];
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    if (line[0] == '#')
      continue;
    long long ns = 0;
    char name[256] = {0};
    if (std::sscanf(line, "%lld,%255[^\r\n]", &ns, name) != 2)
      continue;
    out.emplace_back(double(ns) * 1e-9, cam_dir + "/data/" + name);
  }
  std::fclose(file);
  return out;
}

std::shared_ptr<ov_core::CamBase> make_camera(const double *calib) {
  auto camera = std::make_shared<ov_core::CamRadtan>(kImageWidth, kImageHeight);
  Eigen::Matrix<double, 8, 1> values;
  for (int i = 0; i < 8; i++)
    values(i) = calib[i];
  camera->set_value(values);
  return camera;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s MAV0_DIR OUT_TXT [MAX_FRAMES]\n", argv[0]);
    return 2;
  }
  const std::string mav0 = argv[1];
  const long max_frames = argc > 3 ? std::atol(argv[3]) : -1;
  std::FILE *dump = std::fopen(argv[2], "w");
  if (dump == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", argv[2]);
    return 3;
  }

  const double cam0_calib[8] = {458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
  const double cam1_calib[8] = {457.587, 456.134, 379.999, 255.238, -0.28368365, 0.07451284, -0.00010473, -3.55590700e-05};
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras;
  cameras[0] = make_camera(cam0_calib);
  cameras[1] = make_camera(cam1_calib);

  // config/euroc_mav/estimator_config.yaml: num_pts 200, fast 20, grid 5x5,
  // min_px_dist 10, HISTOGRAM, use_stereo true. numaruco is OpenVINS's default
  // max_aruco_features (1024), matching TrackerOptions::num_aruco on DOD's side.
  auto tracker = std::make_shared<ov_core::TrackKLT>(cameras, 200, 1024, true, ov_core::TrackBase::HISTOGRAM, 20, 5, 5, 10);

  const auto left_index = read_index(mav0 + "/cam0");
  const auto right_index = read_index(mav0 + "/cam1");
  if (left_index.empty() || left_index.size() != right_index.size()) {
    std::fprintf(stderr, "cam0/cam1 index mismatch (%zu vs %zu)\n", left_index.size(), right_index.size());
    return 4;
  }

  cv::setNumThreads(4);
  cv::setRNGSeed(0);
  long frames = 0;
  double track_seconds = 0.0;
  for (size_t i = 0; i < left_index.size(); i++) {
    if (max_frames > 0 && frames >= max_frames)
      break;
    ov_core::CameraData message;
    message.timestamp = left_index[i].first;
    message.sensor_ids.push_back(0);
    message.sensor_ids.push_back(1);
    message.images.push_back(cv::imread(left_index[i].second, cv::IMREAD_GRAYSCALE));
    message.images.push_back(cv::imread(right_index[i].second, cv::IMREAD_GRAYSCALE));
    if (message.images[0].empty() || message.images[1].empty()) {
      std::fprintf(stderr, "failed to read frame %zu\n", i);
      return 5;
    }
    message.masks.push_back(cv::Mat::zeros(kImageHeight, kImageWidth, CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(kImageHeight, kImageWidth, CV_8UC1));
    const auto track_start = std::chrono::steady_clock::now();
    tracker->feed_new_camera(message);
    track_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - track_start).count();

    const auto ids = tracker->get_last_ids();
    const auto obs = tracker->get_last_obs();
    std::fprintf(dump, "F %.9f\n", message.timestamp);
    for (size_t cam = 0; cam < 2; cam++) {
      auto id_it = ids.find(cam);
      auto pt_it = obs.find(cam);
      if (id_it == ids.end() || pt_it == obs.end())
        continue;
      for (size_t k = 0; k < id_it->second.size() && k < pt_it->second.size(); k++) {
        std::fprintf(dump, "%zu %zu %.4f %.4f\n", cam, id_it->second[k], pt_it->second[k].pt.x, pt_it->second[k].pt.y);
      }
    }
    ++frames;
  }
  std::fflush(dump);
  std::fprintf(stderr, "official: %ld frames, tracker mean %.3f ms/frame\n",
               frames, frames ? 1000.0 * track_seconds / double(frames) : 0.0);
  return 0;
}
