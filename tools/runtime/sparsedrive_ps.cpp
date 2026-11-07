#include "sparsedrive_ps.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "frame_files.hpp"
#include "hotarubi/driver.hpp"
#include "hotarubi/exact.hpp"
#include "hotarubi/require.hpp"
#include "hotarubi/runtime.hpp"
#include "hotarubi/safetensors.hpp"
#include "hotarubi/sparsedrive_layout.hpp"

namespace hotarubi::sparsedrive {
namespace {

static_assert(std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<double>::is_iec559);
static_assert(FLT_EVAL_METHOD == 0);

constexpr uint32_t kFloatBytes = 4;
constexpr uint32_t kWordFloats = 32;
constexpr uint32_t kTokenFloats = 8;
constexpr uint32_t kSide = 4;
constexpr uint32_t kProjectionRows = 3;
constexpr uint32_t kFrameWords = 2;
constexpr uint32_t kInverseWords = 4;
constexpr uint32_t kAnchorX = 0;
constexpr uint32_t kAnchorY = 1;
constexpr uint32_t kAnchorW = 3;
constexpr uint32_t kAnchorL = 4;
constexpr uint32_t kAnchorSin = 6;
constexpr uint32_t kAnchorCos = 7;
constexpr double kStaticDistance = 0.5;
constexpr double kEgoOffset = 0.5;
constexpr float kCollisionPenalty = -999.0F;
constexpr float kExpOverflow = 89.0F;
constexpr float kSaturation = 104.0F;
constexpr int64_t kMinNormalExponent = -126;
constexpr int64_t kMaxExponent = 127;
constexpr int64_t kSignificandBits = 24;
constexpr int64_t kSubnormalScale = 149;
constexpr uint32_t kSignBit = 0x80000000U;
constexpr uint32_t kInfinityBits = 0x7F800000U;

struct Heading {
  double c = 1;
  double s = 0;
};

struct Box {
  double x = 0;
  double y = 0;
  double w = 0;
  double l = 0;
  Heading heading;
};

using Point = std::array<float, 2>;
using Trajectory = std::array<Point, kPlanSteps + 1>;
using Boxes = std::array<Box, kPlanSteps>;

void PutWord(std::span<std::byte> bytes, size_t index, uint32_t value) {
  for (size_t i = 0; i < kFloatBytes; ++i) {
    bytes[(index * kFloatBytes) + i] = static_cast<std::byte>(value >> (8 * i));
  }
}

void PutFloat(std::span<std::byte> bytes, size_t index, float value) {
  PutWord(bytes, index, std::bit_cast<uint32_t>(value));
}

uint32_t GetWord(std::span<const std::byte> bytes, size_t index) {
  uint32_t value = 0;
  for (size_t i = 0; i < kFloatBytes; ++i) {
    value |= static_cast<uint32_t>(bytes[(index * kFloatBytes) + i]) << (8 * i);
  }
  return value;
}

float GetFloat(std::span<const std::byte> bytes, size_t index) {
  return std::bit_cast<float>(GetWord(bytes, index));
}

float ToFloat(const exact::Real& value) {
  if (value.IsZero()) {
    return 0.0F;
  }
  const uint32_t sign = value.Negative() ? kSignBit : 0U;
  const exact::Real magnitude = value.Negative() ? value.Negated() : value;
  const int64_t top = magnitude.TopBit();
  if (top > kMaxExponent ||
      (top == kMaxExponent &&
       magnitude.Scaled(kSignificandBits - 1 - kMaxExponent).Nearest() >=
           (int64_t{1} << kSignificandBits))) {
    return std::bit_cast<float>(sign | kInfinityBits);
  }
  if (top < kMinNormalExponent) {
    return std::bit_cast<float>(
        sign |
        static_cast<uint32_t>(magnitude.Scaled(kSubnormalScale).Nearest()));
  }
  return std::bit_cast<float>(value.ToBinary32());
}

exact::Real Exact(float value) {
  return exact::Real::Binary32(std::bit_cast<uint32_t>(value));
}

float Fp32OfFp64(int64_t numerator, int64_t denominator) {
  return std::bit_cast<float>(
      exact::Real::Binary64(
          exact::Real::Ratio(numerator, denominator).ToBinary64())
          .ToBinary32());
}

float EgoDimension(int64_t hundredths) {
  return Fp32OfFp64(hundredths, 100) * Fp32OfFp64(11, 10);
}

Heading Normalize(float dx, float dy) {
  const auto x = static_cast<double>(dx);
  const auto y = static_cast<double>(dy);
  const double norm = std::sqrt((x * x) + (y * y));
  if (!(norm > 0)) {
    return {};
  }
  return {.c = x / norm, .s = y / norm};
}

double Distance(const Point& a, const Point& b) {
  const auto dx = static_cast<double>(a[0] - b[0]);
  const auto dy = static_cast<double>(a[1] - b[1]);
  return std::sqrt((dx * dx) + (dy * dy));
}

std::array<Heading, kPlanSteps + 1> Headings(const Trajectory& trajectory,
                                             Heading start) {
  std::array<Heading, kPlanSteps + 1> headings;
  headings.fill(start);
  if (Distance(trajectory[kPlanSteps], trajectory[0]) < kStaticDistance) {
    return headings;
  }
  for (uint32_t t = 1; t < kPlanSteps; ++t) {
    headings.at(t) =
        Normalize(trajectory.at(t + 1)[0] - trajectory.at(t - 1)[0],
                  trajectory.at(t + 1)[1] - trajectory.at(t - 1)[1]);
  }
  headings[kPlanSteps] =
      Normalize(trajectory[kPlanSteps][0] - trajectory[kPlanSteps - 1][0],
                trajectory[kPlanSteps][1] - trajectory[kPlanSteps - 1][1]);
  return headings;
}

bool CornersInside(const Box& outer, const Box& inner) {
  const double dx = inner.x - outer.x;
  const double dy = inner.y - outer.y;
  const double cx = (dx * outer.heading.c) + (dy * outer.heading.s);
  const double cy = (dy * outer.heading.c) - (dx * outer.heading.s);
  const double c =
      (inner.heading.c * outer.heading.c) + (inner.heading.s * outer.heading.s);
  const double s =
      (inner.heading.s * outer.heading.c) - (inner.heading.c * outer.heading.s);
  const double half_w = outer.w / 2;
  const double half_l = outer.l / 2;
  constexpr std::array<std::array<double, 2>, 4> kCorners = {
      {{-0.5, -0.5}, {-0.5, 0.5}, {0.5, 0.5}, {0.5, -0.5}}};
  return std::ranges::any_of(
      kCorners, [&](const std::array<double, 2>& corner) {
        const double ox = corner[0] * inner.w;
        const double oy = corner[1] * inner.l;
        const double px = cx + ((c * ox) - (s * oy));
        const double py = cy + ((s * ox) + (c * oy));
        return px <= half_w && px >= -half_w && py <= half_l && py >= -half_l;
      });
}

bool Collide(const Box& a, const Box& b) {
  return CornersInside(a, b) || CornersInside(b, a);
}

uint32_t ArgMax(std::span<const float> values) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < values.size(); ++i) {
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return best;
}

Boxes ToBoxes(const Trajectory& trajectory, Heading start, double w, double l,
              double offset) {
  const auto headings = Headings(trajectory, start);
  Boxes boxes;
  for (uint32_t t = 1; t <= kPlanSteps; ++t) {
    const Heading heading = headings.at(t);
    boxes.at(t - 1) = {
        .x = static_cast<double>(trajectory.at(t)[0]) + (offset * heading.c),
        .y = static_cast<double>(trajectory.at(t)[1]) + (offset * heading.s),
        .w = w,
        .l = l,
        .heading = heading};
  }
  return boxes;
}

Boxes AgentBoxes(const PlanningInputs& inputs, uint32_t agent) {
  const std::span<const float> anchor = inputs.det_anchor.subspan(
      static_cast<size_t>(agent) * kAnchorWidth, kAnchorWidth);
  const uint32_t mode = ArgMax(
      inputs.motion_cls.subspan(static_cast<size_t>(agent) * kModes, kModes));
  const std::span<const float> reg = inputs.motion_reg.subspan(
      ((static_cast<size_t>(agent) * kModes) + mode) * kMotionSteps * 2,
      static_cast<size_t>(kMotionSteps) * 2);
  Trajectory trajectory;
  trajectory[0] = {anchor[kAnchorX], anchor[kAnchorY]};
  Point sum = {reg[0], reg[1]};
  for (uint32_t t = 0; t < kPlanSteps; ++t) {
    if (t > 0) {
      sum = {sum[0] + reg[size_t{2} * t], sum[1] + reg[(size_t{2} * t) + 1]};
    }
    trajectory.at(t + 1) = {sum[0] + anchor[kAnchorX],
                            sum[1] + anchor[kAnchorY]};
  }
  return ToBoxes(trajectory, Normalize(anchor[kAnchorCos], anchor[kAnchorSin]),
                 ExpFloat(anchor[kAnchorW]), ExpFloat(anchor[kAnchorL]), 0);
}

Boxes EgoBoxes(std::span<const float> trajectories, uint32_t mode) {
  Trajectory trajectory;
  trajectory[0] = {0.0F, 0.0F};
  for (uint32_t t = 0; t < kPlanSteps; ++t) {
    const size_t at = ((static_cast<size_t>(mode) * kPlanSteps) + t) * 2;
    trajectory.at(t + 1) = {trajectories[at], trajectories[at + 1]};
  }
  return ToBoxes(trajectory, {.c = 0, .s = 1}, EgoDimension(408),
                 EgoDimension(173), kEgoOffset);
}

bool Collides(const Boxes& ego, const std::vector<Boxes>& agents) {
  return std::ranges::any_of(agents, [&](const Boxes& agent) {
    for (uint32_t t = 0; t < kPlanSteps; ++t) {
      if (Collide(ego.at(t), agent.at(t))) {
        return true;
      }
    }
    return false;
  });
}

std::array<float, kModes> Rescore(std::array<float, kModes> scores,
                                  std::span<const float> trajectories,
                                  const PlanningInputs& inputs) {
  std::vector<Boxes> agents;
  for (uint32_t a = 0; a < kDetections; ++a) {
    const std::span<const float> logits =
        inputs.det_cls.subspan(static_cast<size_t>(a) * kClasses, kClasses);
    if (*std::ranges::max_element(logits) >= 0) {
      agents.push_back(AgentBoxes(inputs, a));
    }
  }
  if (agents.empty()) {
    return scores;
  }
  std::array<bool, kModes> collides{};
  for (uint32_t m = 0; m < kModes; ++m) {
    collides.at(m) = Collides(EgoBoxes(trajectories, m), agents);
  }
  if (std::ranges::all_of(collides, std::identity())) {
    return scores;
  }
  for (uint32_t m = 0; m < kModes; ++m) {
    if (collides.at(m)) {
      scores.at(m) = scores.at(m) + kCollisionPenalty;
    }
  }
  return scores;
}

std::expected<std::vector<std::byte>, std::string> ReadRegion(
    const Runtime& runtime, std::string_view name) {
  const auto region = runtime.FindRegion(name);
  if (!region) {
    return std::unexpected(region.error());
  }
  std::vector<std::byte> bytes(region->size);
  if (auto read = runtime.Read(name, bytes); !read) {
    return std::unexpected(read.error());
  }
  return bytes;
}

std::expected<std::vector<float>, std::string> ReadRows(const Runtime& runtime,
                                                        std::string_view name,
                                                        uint32_t row_floats,
                                                        uint32_t rows,
                                                        uint32_t cols) {
  const auto bytes = ReadRegion(runtime, name);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  if (cols > row_floats ||
      bytes->size() < static_cast<size_t>(rows) * row_floats * kFloatBytes) {
    return std::unexpected(
        std::format("region {} of {} bytes does not hold {} rows of {} values",
                    name, bytes->size(), rows, row_floats));
  }
  std::vector<float> values;
  values.reserve(static_cast<size_t>(rows) * cols);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      values.push_back(
          GetFloat(*bytes, (static_cast<size_t>(r) * row_floats) + c));
    }
  }
  return values;
}

std::expected<std::vector<uint32_t>, std::string> ReadWords(
    const Runtime& runtime, std::string_view name) {
  const auto bytes = ReadRegion(runtime, name);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  std::vector<uint32_t> words(bytes->size() / kFloatBytes);
  for (size_t i = 0; i < words.size(); ++i) {
    words[i] = GetWord(*bytes, i);
  }
  return words;
}

std::vector<float> AsFloats(const std::vector<uint32_t>& words) {
  std::vector<float> values(words.size());
  std::ranges::transform(words, values.begin(), [](uint32_t word) {
    return std::bit_cast<float>(word);
  });
  return values;
}

std::expected<void, std::string> Update(
    Runtime& runtime, std::string_view name, size_t size,
    const std::function<void(std::span<std::byte>)>& change) {
  auto bytes = ReadRegion(runtime, name);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  if (bytes->size() != size) {
    return std::unexpected(std::format("region {} has {} bytes, not {}", name,
                                       bytes->size(), size));
  }
  change(*bytes);
  return runtime.Write(name, *bytes);
}

std::expected<void, std::string> PlaceInputs(Runtime& runtime,
                                             const Frame& frame) {
  for (uint32_t c = 0; c < kCameras; ++c) {
    const std::string name =
        c == 0 ? std::string("stem.in") : std::format("stem.in.cam{}", c);
    const std::span<const float> image =
        std::span<const float>(frame.images)
            .subspan(static_cast<size_t>(c) * kImageValues, kImageValues);
    if (auto written = runtime.Write(name, StemImage(image)); !written) {
      return written;
    }
  }
  for (const std::string_view head : {"det", "map"}) {
    if (auto written = runtime.Write(std::format("{}.cam_in", head),
                                     CameraRows(frame.projection));
        !written) {
      return written;
    }
    if (auto updated = Update(
            runtime, std::format("{}.kp_const", head),
            static_cast<size_t>(kCameras) * kWordFloats * kFloatBytes,
            [&](std::span<std::byte> bytes) {
              PutKeypointConstants(bytes, frame.projection, frame.image_wh);
            });
        !updated) {
      return updated;
    }
  }
  if (auto written = runtime.Write(
          "fs.meta", FrameMeta(frame.timestamp_us, frame.t_global));
      !written) {
    return written;
  }
  return Update(
      runtime, "fs.tinv",
      static_cast<size_t>(kInverseWords + 1) * kWordFloats * kFloatBytes,
      [&](std::span<std::byte> bytes) {
        PutInverse(bytes, frame.t_global_inv);
      });
}

std::expected<void, std::string> ReadPlanningInputs(const Runtime& runtime,
                                                    std::string_view stream,
                                                    Result& result) {
  struct Source {
    std::string name;
    uint32_t rows;
    uint32_t cols;
    std::vector<float>* out;
  };
  const std::array<Source, 6> sources = {
      {{.name = "det.cls",
        .rows = kDetections,
        .cols = kClasses,
        .out = &result.det_cls},
       {.name = std::format("{}.det.anchor", stream),
        .rows = kDetections,
        .cols = kAnchorWidth,
        .out = &result.det_anchor},
       {.name = "motion.motion_cls",
        .rows = kDetections * kModes,
        .cols = 1,
        .out = &result.motion_cls},
       {.name = "motion.motion_reg",
        .rows = kDetections * kModes,
        .cols = kMotionSteps * 2,
        .out = &result.motion_reg},
       {.name = "motion.plan_cls",
        .rows = kPlanQueries,
        .cols = 1,
        .out = &result.plan_cls},
       {.name = "motion.plan_reg",
        .rows = kPlanQueries,
        .cols = kPlanSteps * 2,
        .out = &result.plan_reg}}};
  for (const Source& source : sources) {
    auto values =
        ReadRows(runtime, source.name, kRowFloats, source.rows, source.cols);
    if (!values) {
      return std::unexpected(values.error());
    }
    *source.out = std::move(*values);
  }
  return {};
}

std::expected<void, std::string> ReadOutputs(const Runtime& runtime,
                                             std::string_view stream,
                                             Result& result) {
  struct Source {
    std::string name;
    uint32_t row_floats;
    uint32_t rows;
    uint32_t cols;
    std::vector<float>* out;
  };
  const std::array<Source, 7> sources = {
      {{.name = std::format("{}.det.x", stream),
        .row_floats = kFeatureWidth,
        .rows = kDetections,
        .cols = kFeatureWidth,
        .out = &result.det_feat},
       {.name = "det.qt",
        .row_floats = kRowFloats,
        .rows = kDetections,
        .cols = kQualityWidth,
        .out = &result.det_quality},
       {.name = std::format("{}.map.x", stream),
        .row_floats = kFeatureWidth,
        .rows = kMapInstances,
        .cols = kFeatureWidth,
        .out = &result.map_feat},
       {.name = std::format("{}.map.anchor", stream),
        .row_floats = kMapAnchorRowFloats,
        .rows = kMapInstances,
        .cols = kMapAnchorWidth,
        .out = &result.map_anchor},
       {.name = "map.cls",
        .row_floats = kRowFloats,
        .rows = kMapInstances,
        .cols = kMapClasses,
        .out = &result.map_cls},
       {.name = "motion.plan_status",
        .row_floats = kRowFloats,
        .rows = 1,
        .cols = kPlanStatusWidth,
        .out = &result.plan_status},
       {.name = "motion.state",
        .row_floats = kRowFloats,
        .rows = kStateRows,
        .cols = kStateWidth,
        .out = &result.state}}};
  for (const Source& source : sources) {
    auto values = ReadRows(runtime, source.name, source.row_floats, source.rows,
                           source.cols);
    if (!values) {
      return std::unexpected(values.error());
    }
    *source.out = std::move(*values);
  }
  const auto det_prevconf = ReadWords(runtime, "det.prevconf");
  const auto det_keep = ReadWords(runtime, "det.keep");
  const auto map_prevconf = ReadWords(runtime, "map.prevconf");
  const auto map_keep = ReadWords(runtime, "map.keep");
  const auto flag = ReadWords(runtime, "fs.flag");
  const auto det_x = runtime.FindRegion(std::format("{}.det.x", stream));
  const auto map_x = runtime.FindRegion(std::format("{}.map.x", stream));
  for (const auto* words :
       {&det_prevconf, &det_keep, &map_prevconf, &map_keep, &flag}) {
    if (!*words) {
      return std::unexpected(words->error());
    }
  }
  if (!det_x || !map_x || flag->empty()) {
    return std::unexpected("the final feature regions or the flag are missing");
  }
  result.det_prevconf = AsFloats(*det_prevconf);
  result.det_keep = *det_keep;
  result.map_prevconf = AsFloats(*map_prevconf);
  result.map_keep = *map_keep;
  result.det_x0 = det_x->offset;
  result.map_x0 = map_x->offset;
  result.fs_flag = flag->front();
  return {};
}

std::vector<uint64_t> Shape(std::initializer_list<uint32_t> dims) {
  std::vector<uint64_t> shape;
  shape.reserve(dims.size());
  for (const uint32_t dim : dims) {
    shape.push_back(dim);
  }
  return shape;
}

}  // namespace

std::string_view SceneTracker::Stream(std::string_view scene) {
  const bool continues = scene_.has_value() && *scene_ == scene;
  scene_ = std::string(scene);
  return continues ? "continue" : "start";
}

std::vector<std::byte> StemImage(std::span<const float> image) {
  Require(image.size() == kImageValues,
          "the camera image is not 3 x 256 x 704");
  std::vector<std::byte> out(kStemImageBytes);
  for (uint32_t p = 0; p < kStemPhases; ++p) {
    for (uint32_t c = 0; c < kImageChannels; ++c) {
      for (uint32_t y = 0; y < kImageRows; ++y) {
        const size_t row =
            ((static_cast<size_t>((p * kImageChannels) + c) * kImageRows) + y) *
            kStemRowFloats;
        const size_t from =
            ((static_cast<size_t>(c) * kImageRows) + y) * kImageColumns;
        for (uint32_t x = 0; x < kImageColumns; ++x) {
          PutFloat(out, row + kStemPadding + (size_t{2} * p) + x,
                   image[from + x]);
        }
      }
    }
  }
  return out;
}

std::vector<std::byte> CameraRows(std::span<const float> projection) {
  Require(projection.size() == static_cast<size_t>(kCameras) * kSide * kSide,
          "the projection matrices are not 6 x 4 x 4");
  std::vector<std::byte> out(static_cast<size_t>(kWordFloats) * kWordFloats *
                             kFloatBytes);
  for (uint32_t c = 0; c < kCameras; ++c) {
    for (uint32_t i = 0; i < kProjectionRows * kSide; ++i) {
      PutFloat(out, (static_cast<size_t>(c) * kWordFloats) + i,
               projection[(static_cast<size_t>(c) * kSide * kSide) + i]);
    }
  }
  return out;
}

void PutKeypointConstants(std::span<std::byte> kp_const,
                          std::span<const float> projection,
                          std::span<const float> image_wh) {
  Require(
      kp_const.size() ==
              static_cast<size_t>(kCameras) * kWordFloats * kFloatBytes &&
          projection.size() == static_cast<size_t>(kCameras) * kSide * kSide &&
          image_wh.size() == static_cast<size_t>(kCameras) * 2,
      "the keypoint constants do not match the cameras");
  for (uint32_t c = 0; c < kCameras; ++c) {
    const size_t word = static_cast<size_t>(c) * kWordFloats;
    PutFloat(kp_const, word + kKpWidthLane, image_wh[size_t{2} * c]);
    PutFloat(kp_const, word + kKpHeightLane, image_wh[(size_t{2} * c) + 1]);
    for (uint32_t i = 0; i < kProjectionRows * kSide; ++i) {
      PutFloat(kp_const, word + kKpProjectionLane + i,
               projection[(static_cast<size_t>(c) * kSide * kSide) + i]);
    }
  }
}

std::vector<std::byte> FrameMeta(int64_t timestamp_us,
                                 std::span<const float> t_global) {
  Require(t_global.size() == static_cast<size_t>(kSide) * kSide,
          "T_global is not 4 x 4");
  std::vector<std::byte> out(static_cast<size_t>(kFrameWords) * kWordFloats *
                             kFloatBytes);
  const auto timestamp = static_cast<uint64_t>(timestamp_us);
  PutWord(out, 0, static_cast<uint32_t>(timestamp));
  PutWord(out, 1, static_cast<uint32_t>(timestamp >> 32));
  for (uint32_t j = 0; j < kSide; ++j) {
    for (uint32_t i = 0; i < kSide; ++i) {
      PutFloat(out, kWordFloats + (j * kTokenFloats) + i,
               t_global[(i * kSide) + j]);
    }
  }
  return out;
}

void PutInverse(std::span<std::byte> tinv,
                std::span<const float> t_global_inv) {
  Require(tinv.size() >= static_cast<size_t>(kInverseWords) * kWordFloats *
                             kFloatBytes &&
              t_global_inv.size() == static_cast<size_t>(kSide) * kSide,
          "T_global_inv does not fit the inverse words");
  std::fill_n(tinv.begin(),
              static_cast<size_t>(kInverseWords) * kWordFloats * kFloatBytes,
              std::byte{0});
  for (uint32_t k = 0; k < kInverseWords; ++k) {
    for (uint32_t i = 0; i < kSide; ++i) {
      PutFloat(tinv, (k * kWordFloats) + i, t_global_inv[(i * kSide) + k]);
    }
  }
}

float ExpFloat(float x) {
  if (std::isnan(x)) {
    return x;
  }
  if (x >= kExpOverflow) {
    return std::numeric_limits<float>::infinity();
  }
  if (x <= -kSaturation) {
    return 0.0F;
  }
  return ToFloat(exact::Exp(Exact(x)));
}

float SigmoidFloat(float x) {
  if (std::isnan(x)) {
    return x;
  }
  if (x >= kSaturation) {
    return 1.0F;
  }
  if (x <= -kSaturation) {
    return 0.0F;
  }
  const exact::Real one = exact::Real::Integer(1);
  return ToFloat(one / (one + exact::Exp(Exact(x).Negated())));
}

Planning DecodePlanning(const PlanningInputs& inputs) {
  Require(
      inputs.det_cls.size() == static_cast<size_t>(kDetections) * kClasses &&
          inputs.det_anchor.size() ==
              static_cast<size_t>(kDetections) * kAnchorWidth &&
          inputs.motion_cls.size() ==
              static_cast<size_t>(kDetections) * kModes &&
          inputs.motion_reg.size() ==
              static_cast<size_t>(kDetections) * kModes * kMotionSteps * 2 &&
          inputs.plan_cls.size() == kPlanQueries &&
          inputs.plan_reg.size() ==
              static_cast<size_t>(kPlanQueries) * kPlanSteps * 2 &&
          inputs.command.size() == kCommands,
      "the planning inputs have the wrong sizes");
  Planning planning;
  planning.trajectories.resize(static_cast<size_t>(kPlanQueries) * kPlanSteps *
                               2);
  for (uint32_t q = 0; q < kPlanQueries; ++q) {
    for (uint32_t xy = 0; xy < 2; ++xy) {
      float sum = 0;
      for (uint32_t t = 0; t < kPlanSteps; ++t) {
        const size_t at =
            (((static_cast<size_t>(q) * kPlanSteps) + t) * 2) + xy;
        sum = t == 0 ? inputs.plan_reg[at] : sum + inputs.plan_reg[at];
        planning.trajectories[at] = sum;
      }
    }
  }
  planning.command = ArgMax(inputs.command);
  const size_t first = static_cast<size_t>(planning.command) * kModes;
  const std::span<const float> trajectories =
      std::span<const float>(planning.trajectories)
          .subspan(first * kPlanSteps * 2,
                   static_cast<size_t>(kModes) * kPlanSteps * 2);
  std::array<float, kModes> scores{};
  std::ranges::copy(inputs.plan_cls.subspan(first, kModes), scores.begin());
  scores = Rescore(scores, trajectories, inputs);
  planning.mode = ArgMax(scores);
  planning.score.assign(inputs.plan_cls.begin(), inputs.plan_cls.end());
  std::ranges::copy(
      scores, planning.score.begin() + static_cast<std::ptrdiff_t>(first));
  std::ranges::transform(planning.score, planning.score.begin(), SigmoidFloat);
  const auto final_trajectory =
      trajectories.subspan(static_cast<size_t>(planning.mode) * kPlanSteps * 2,
                           static_cast<size_t>(kPlanSteps) * 2);
  planning.final_trajectory.assign(final_trajectory.begin(),
                                   final_trajectory.end());
  return planning;
}

std::expected<Frame, std::string> ReadFrame(const SafeTensors& file) {
  Frame frame;
  struct Field {
    std::string_view name;
    std::vector<uint64_t> shape;
    std::vector<float>* out;
  };
  const std::array<Field, 6> fields = {
      {{.name = "img",
        .shape = Shape({kCameras, kImageChannels, kImageRows, kImageColumns}),
        .out = &frame.images},
       {.name = "projection_mat",
        .shape = Shape({kCameras, kSide, kSide}),
        .out = &frame.projection},
       {.name = "image_wh",
        .shape = Shape({kCameras, 2}),
        .out = &frame.image_wh},
       {.name = "T_global",
        .shape = Shape({kSide, kSide}),
        .out = &frame.t_global},
       {.name = "T_global_inv",
        .shape = Shape({kSide, kSide}),
        .out = &frame.t_global_inv},
       {.name = "cmd", .shape = Shape({kCommands}), .out = &frame.command}}};
  for (const Field& field : fields) {
    auto values = ReadFloats(file, field.name, field.shape);
    if (!values) {
      return std::unexpected(values.error());
    }
    *field.out = std::move(*values);
  }
  if (!file.Contains("timestamp_us") ||
      file.Info("timestamp_us").dtype != DType::kI64 ||
      file.Int64s("timestamp_us").size() != 1) {
    return std::unexpected("tensor timestamp_us is not a single I64");
  }
  frame.timestamp_us = file.Int64s("timestamp_us").front();
  auto token = ReadText(file, "token");
  auto scene = ReadText(file, "scene_token");
  if (!token || !scene) {
    return std::unexpected(token ? scene.error() : token.error());
  }
  frame.token = std::move(*token);
  frame.scene = std::move(*scene);
  return frame;
}

std::expected<Result, std::string> RunFrame(Runtime& runtime,
                                            std::string_view stream,
                                            const Frame& frame,
                                            std::chrono::milliseconds timeout,
                                            const TraceRequest& trace) {
  Timing timing;
  auto clock = std::chrono::steady_clock::now();
  const auto lap = [&clock] {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - clock;
    clock = now;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
  };
  if (auto placed = PlaceInputs(runtime, frame); !placed) {
    return std::unexpected(placed.error());
  }
  timing.inputs = lap();
  Result result;
  std::optional<std::string> failure;
  bool decoded = false;
  const auto on_signal = [&](const Signal& signal) {
    const auto begin = std::chrono::steady_clock::now();
    if (decoded || signal.id != kMotionDoneSignal ||
        signal.value != kMotionDoneSignal) {
      failure = std::format("unexpected signal {} with value {}", signal.id,
                            signal.value);
      return;
    }
    if (auto read = ReadPlanningInputs(runtime, stream, result); read) {
      result.planning = DecodePlanning({.det_cls = result.det_cls,
                                        .det_anchor = result.det_anchor,
                                        .motion_cls = result.motion_cls,
                                        .motion_reg = result.motion_reg,
                                        .plan_cls = result.plan_cls,
                                        .plan_reg = result.plan_reg,
                                        .command = frame.command});
      decoded = true;
    } else {
      failure = read.error();
    }
    timing.signals += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin);
  };
  auto counters = runtime.Run(stream, timeout, on_signal, trace);
  timing.run = lap() - timing.signals;
  if (!counters) {
    return std::unexpected(counters.error());
  }
  if (failure) {
    return std::unexpected(*failure);
  }
  if (!decoded) {
    return std::unexpected(
        "the instruction stream did not signal the end of "
        "the motion head");
  }
  result.counters = *counters;
  if (auto read = ReadOutputs(runtime, stream, result); !read) {
    return std::unexpected(read.error());
  }
  timing.outputs = lap();
  result.timing = timing;
  return result;
}

std::vector<std::pair<std::string, Tensor>> OutputTensors(
    const Frame& frame, const Result& result) {
  const Planning& planning = result.planning;
  const std::array<int64_t, 3> integers = {planning.mode, result.det_x0,
                                           result.map_x0};
  std::vector<std::pair<std::string, Tensor>> tensors;
  tensors.emplace_back(
      "final_planning",
      FloatTensor(Shape({kPlanSteps, 2}), planning.final_trajectory));
  tensors.emplace_back("planning",
                       FloatTensor(Shape({kCommands, kModes, kPlanSteps, 2}),
                                   planning.trajectories));
  tensors.emplace_back("planning_score",
                       FloatTensor(Shape({kCommands, kModes}), planning.score));
  tensors.emplace_back("mode", Int64Tensor({}, std::span(integers).first(1)));
  tensors.emplace_back("cmd", FloatTensor(Shape({kCommands}), frame.command));
  tensors.emplace_back(
      "det_feat",
      FloatTensor(Shape({kDetections, kFeatureWidth}), result.det_feat));
  tensors.emplace_back(
      "det_anchor",
      FloatTensor(Shape({kDetections, kAnchorWidth}), result.det_anchor));
  tensors.emplace_back(
      "det_cls", FloatTensor(Shape({kDetections, kClasses}), result.det_cls));
  tensors.emplace_back(
      "det_quality",
      FloatTensor(Shape({kDetections, kQualityWidth}), result.det_quality));
  tensors.emplace_back(
      "map_feat",
      FloatTensor(Shape({kMapInstances, kFeatureWidth}), result.map_feat));
  tensors.emplace_back(
      "map_anchor",
      FloatTensor(Shape({kMapInstances, kMapAnchorWidth}), result.map_anchor));
  tensors.emplace_back(
      "map_cls",
      FloatTensor(Shape({kMapInstances, kMapClasses}), result.map_cls));
  tensors.emplace_back("motion_cls", FloatTensor(Shape({kDetections, kModes}),
                                                 result.motion_cls));
  tensors.emplace_back(
      "motion_reg", FloatTensor(Shape({kDetections, kModes, kMotionSteps, 2}),
                                result.motion_reg));
  tensors.emplace_back("plan_cls",
                       FloatTensor(Shape({kPlanQueries}), result.plan_cls));
  tensors.emplace_back(
      "plan_reg",
      FloatTensor(Shape({kPlanQueries, kPlanSteps, 2}), result.plan_reg));
  tensors.emplace_back("plan_status", FloatTensor(Shape({kPlanStatusWidth}),
                                                  result.plan_status));
  tensors.emplace_back(
      "st", FloatTensor(Shape({kStateRows, kStateWidth}), result.state));
  tensors.emplace_back("det_prevconf", FloatTensor({result.det_prevconf.size()},
                                                   result.det_prevconf));
  tensors.emplace_back("det_keep",
                       Uint32Tensor({result.det_keep.size()}, result.det_keep));
  tensors.emplace_back("det_x0",
                       Int64Tensor({}, std::span(integers).subspan(1, 1)));
  tensors.emplace_back("map_prevconf", FloatTensor({result.map_prevconf.size()},
                                                   result.map_prevconf));
  tensors.emplace_back("map_keep",
                       Uint32Tensor({result.map_keep.size()}, result.map_keep));
  tensors.emplace_back("map_x0",
                       Int64Tensor({}, std::span(integers).subspan(2, 1)));
  tensors.emplace_back("fs_flag",
                       Uint32Tensor({}, std::span(&result.fs_flag, 1)));
  return tensors;
}

}  // namespace hotarubi::sparsedrive
