#ifndef HOTARUBI_TOOLS_RUNTIME_SPARSEDRIVE_PS_HPP_
#define HOTARUBI_TOOLS_RUNTIME_SPARSEDRIVE_PS_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hotarubi/driver.hpp"
#include "hotarubi/runtime.hpp"
#include "hotarubi/safetensors.hpp"
#include "hotarubi/sparsedrive_layout.hpp"

namespace hotarubi::sparsedrive {

inline constexpr uint32_t kDetections = 900;
inline constexpr uint32_t kClasses = 10;
inline constexpr uint32_t kAnchorWidth = 11;
inline constexpr uint32_t kQualityWidth = 2;
inline constexpr uint32_t kMapInstances = 100;
inline constexpr uint32_t kMapAnchorWidth = 40;
inline constexpr uint32_t kMapClasses = 3;
inline constexpr uint32_t kFeatureWidth = 256;
inline constexpr uint32_t kRowFloats = 32;
inline constexpr uint32_t kMapAnchorRowFloats = 64;
inline constexpr uint32_t kModes = 6;
inline constexpr uint32_t kMotionSteps = 12;
inline constexpr uint32_t kPlanSteps = 6;
inline constexpr uint32_t kCommands = 3;
inline constexpr uint32_t kPlanQueries = kCommands * kModes;
inline constexpr uint32_t kPlanStatusWidth = 10;
inline constexpr uint32_t kStateRows = kDetections + 1;
inline constexpr uint32_t kStateWidth = 3;
inline constexpr uint32_t kImageValues =
    kImageChannels * kImageRows * kImageColumns;

struct Frame {
  std::vector<float> images;
  std::vector<float> projection;
  std::vector<float> image_wh;
  std::vector<float> t_global;
  std::vector<float> t_global_inv;
  int64_t timestamp_us = 0;
  std::vector<float> command;
  std::string token;
  std::string scene;
};

struct PlanningInputs {
  std::span<const float> det_cls;
  std::span<const float> det_anchor;
  std::span<const float> motion_cls;
  std::span<const float> motion_reg;
  std::span<const float> plan_cls;
  std::span<const float> plan_reg;
  std::span<const float> command;
};

struct Planning {
  std::vector<float> score;
  std::vector<float> trajectories;
  std::vector<float> final_trajectory;
  uint32_t command = 0;
  uint32_t mode = 0;
};

struct Result {
  Planning planning;
  std::vector<float> det_feat;
  std::vector<float> det_anchor;
  std::vector<float> det_cls;
  std::vector<float> det_quality;
  std::vector<float> map_feat;
  std::vector<float> map_anchor;
  std::vector<float> map_cls;
  std::vector<float> motion_cls;
  std::vector<float> motion_reg;
  std::vector<float> plan_cls;
  std::vector<float> plan_reg;
  std::vector<float> plan_status;
  std::vector<float> state;
  std::vector<float> det_prevconf;
  std::vector<uint32_t> det_keep;
  std::vector<float> map_prevconf;
  std::vector<uint32_t> map_keep;
  int64_t det_x0 = 0;
  int64_t map_x0 = 0;
  uint32_t fs_flag = 0;
  Counters counters;
};

class SceneTracker {
 public:
  std::string_view Stream(std::string_view scene);

 private:
  std::optional<std::string> scene_;
};

std::vector<std::byte> StemImage(std::span<const float> image);
std::vector<std::byte> CameraRows(std::span<const float> projection);
void PutKeypointConstants(std::span<std::byte> kp_const,
                          std::span<const float> projection,
                          std::span<const float> image_wh);
std::vector<std::byte> FrameMeta(int64_t timestamp_us,
                                 std::span<const float> t_global);
void PutInverse(std::span<std::byte> tinv, std::span<const float> t_global_inv);
float ExpFloat(float x);
float SigmoidFloat(float x);
Planning DecodePlanning(const PlanningInputs& inputs);

std::expected<Frame, std::string> ReadFrame(const SafeTensors& file);
std::expected<Result, std::string> RunFrame(Runtime& runtime,
                                            std::string_view stream,
                                            const Frame& frame,
                                            std::chrono::milliseconds timeout);
std::vector<std::pair<std::string, Tensor>> OutputTensors(const Frame& frame,
                                                          const Result& result);

}  // namespace hotarubi::sparsedrive

#endif  // HOTARUBI_TOOLS_RUNTIME_SPARSEDRIVE_PS_HPP_
