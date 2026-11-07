#include "sparsedrive_ps.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hotarubi/driver.hpp"
#include "hotarubi/runtime.hpp"
#include "hotarubi/safetensors.hpp"
#include "hotarubi/sparsedrive_layout.hpp"
#include "scripted_bus.hpp"

namespace hotarubi::sparsedrive {
namespace {

using std::chrono_literals::operator""ms;

constexpr uint64_t kBase = 0x38700000;
constexpr uint32_t kWordBytes = 128;

float FloatAt(std::span<const std::byte> bytes, size_t index) {
  uint32_t bits = 0;
  for (size_t i = 0; i < 4; ++i) {
    bits |= static_cast<uint32_t>(bytes[(index * 4) + i]) << (8 * i);
  }
  return std::bit_cast<float>(bits);
}

uint32_t WordAt(std::span<const std::byte> bytes, size_t index) {
  return std::bit_cast<uint32_t>(FloatAt(bytes, index));
}

void SetFloat(std::span<std::byte> bytes, size_t index, float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  for (size_t i = 0; i < 4; ++i) {
    bytes[(index * 4) + i] = static_cast<std::byte>(bits >> (8 * i));
  }
}

std::vector<float> Iota(size_t size, float scale) {
  std::vector<float> values(size);
  for (size_t i = 0; i < size; ++i) {
    values[i] = static_cast<float>(i) * scale;
  }
  return values;
}

struct Scene {
  std::vector<float> det_cls =
      std::vector<float>(static_cast<size_t>(kDetections) * kClasses, -5.0F);
  std::vector<float> det_anchor =
      std::vector<float>(static_cast<size_t>(kDetections) * kAnchorWidth, 0.0F);
  std::vector<float> motion_cls =
      std::vector<float>(static_cast<size_t>(kDetections) * kModes, 0.0F);
  std::vector<float> motion_reg = std::vector<float>(
      static_cast<size_t>(kDetections) * kModes * kMotionSteps * 2, 0.0F);
  std::vector<float> plan_cls = std::vector<float>(kPlanQueries, 0.0F);
  std::vector<float> plan_reg = std::vector<float>(
      static_cast<size_t>(kPlanQueries) * kPlanSteps * 2, 0.0F);
  std::vector<float> command = {1.0F, 0.0F, 0.0F};

  void AddAgent(float x, float y, float logit) {
    det_cls[0] = logit;
    det_anchor[0] = x;
    det_anchor[1] = y;
    det_anchor[7] = 1.0F;
  }

  void Steer(uint32_t mode, float dx, float dy) {
    for (uint32_t t = 0; t < kPlanSteps; ++t) {
      plan_reg[((static_cast<size_t>(mode) * kPlanSteps) + t) * 2] = dx;
      plan_reg[(((static_cast<size_t>(mode) * kPlanSteps) + t) * 2) + 1] = dy;
    }
  }

  [[nodiscard]] PlanningInputs Inputs() const {
    return {.det_cls = det_cls,
            .det_anchor = det_anchor,
            .motion_cls = motion_cls,
            .motion_reg = motion_reg,
            .plan_cls = plan_cls,
            .plan_reg = plan_reg,
            .command = command};
  }
};

Scene CrossingScene(float logit) {
  Scene scene;
  scene.AddAgent(0.0F, 3.0F, logit);
  for (uint32_t m = 0; m < 3; ++m) {
    scene.Steer(m, 0.0F, 1.0F);
  }
  for (uint32_t m = 3; m < kModes; ++m) {
    scene.Steer(m, 10.0F, 0.0F);
  }
  scene.plan_cls[0] = 5.0F;
  scene.plan_cls[3] = 4.0F;
  return scene;
}

TEST(SparseDriveTest, StemImagePlacesEachPhaseAfterThePadding) {
  const std::vector<float> image = Iota(kImageValues, 1.0F);
  const std::vector<std::byte> bytes = StemImage(image);
  ASSERT_EQ(bytes.size(), kStemImageBytes);
  const auto at = [](uint32_t p, uint32_t c, uint32_t y, uint32_t column) {
    return (((((static_cast<size_t>(p) * kImageChannels) + c) * kImageRows) +
             y) *
            kStemRowFloats) +
           column;
  };
  const auto pixel = [](uint32_t c, uint32_t y, uint32_t x) {
    return static_cast<float>((((c * kImageRows) + y) * kImageColumns) + x);
  };
  EXPECT_EQ(FloatAt(bytes, at(0, 0, 0, 3)), pixel(0, 0, 0));
  EXPECT_EQ(FloatAt(bytes, at(1, 0, 0, 5)), pixel(0, 0, 0));
  EXPECT_EQ(FloatAt(bytes, at(1, 2, 255, 5 + 703)), pixel(2, 255, 703));
  EXPECT_EQ(FloatAt(bytes, at(0, 1, 7, 2)), 0.0F);
  EXPECT_EQ(FloatAt(bytes, at(0, 1, 7, 3 + 704)), 0.0F);
  EXPECT_EQ(FloatAt(bytes, at(1, 1, 7, 4)), 0.0F);
}

TEST(SparseDriveTest, CameraRowsHoldTheFirstThreeRowsOfEachProjection) {
  const std::vector<float> projection =
      Iota(static_cast<size_t>(kCameras) * 16, 1.0F);
  const std::vector<std::byte> bytes = CameraRows(projection);
  ASSERT_EQ(bytes.size(), size_t{32} * kWordBytes);
  EXPECT_EQ(FloatAt(bytes, (2 * 32) + 11), projection[(2 * 16) + 11]);
  EXPECT_EQ(FloatAt(bytes, (5 * 32) + 0), projection[size_t{5} * 16]);
  EXPECT_EQ(FloatAt(bytes, (2 * 32) + 12), 0.0F);
  EXPECT_EQ(FloatAt(bytes, size_t{6} * 32), 0.0F);
}

TEST(SparseDriveTest, KeypointConstantsKeepTheOtherLanes) {
  std::vector<std::byte> bytes(static_cast<size_t>(kCameras) * kWordBytes,
                               std::byte{0x11});
  const std::vector<float> projection =
      Iota(static_cast<size_t>(kCameras) * 16, 1.0F);
  const std::vector<float> wh = {704, 256, 704, 256, 704, 256,
                                 704, 256, 704, 256, 704, 256};
  PutKeypointConstants(bytes, projection, wh);
  EXPECT_EQ(FloatAt(bytes, (3 * 32) + kKpWidthLane), 704.0F);
  EXPECT_EQ(FloatAt(bytes, (3 * 32) + kKpHeightLane), 256.0F);
  EXPECT_EQ(FloatAt(bytes, (3 * 32) + kKpProjectionLane + 11),
            projection[(3 * 16) + 11]);
  EXPECT_EQ(WordAt(bytes, (3 * 32) + 11), 0x11111111U);
  EXPECT_EQ(WordAt(bytes, (3 * 32) + kKpProjectionLane + 12), 0x11111111U);
}

TEST(SparseDriveTest, FrameMetaHoldsTheTimestampAndTheTransposedPose) {
  const std::vector<float> pose = Iota(16, 1.0F);
  const int64_t timestamp = (int64_t{0x12345} << 32) | 0x6789ABCD;
  const std::vector<std::byte> bytes = FrameMeta(timestamp, pose);
  ASSERT_EQ(bytes.size(), size_t{2} * kWordBytes);
  EXPECT_EQ(WordAt(bytes, 0), 0x6789ABCDU);
  EXPECT_EQ(WordAt(bytes, 1), 0x12345U);
  EXPECT_EQ(FloatAt(bytes, 32 + (2 * 8) + 1), pose[(1 * 4) + 2]);
  EXPECT_EQ(FloatAt(bytes, 32 + (3 * 8) + 0), pose[3]);
  EXPECT_EQ(FloatAt(bytes, 32 + (3 * 8) + 4), 0.0F);
}

TEST(SparseDriveTest, InverseFillsFourWordsAndKeepsTheFifth) {
  std::vector<std::byte> bytes(size_t{5} * kWordBytes, std::byte{0x22});
  const std::vector<float> inverse = Iota(16, 0.5F);
  PutInverse(bytes, inverse);
  EXPECT_EQ(FloatAt(bytes, (2 * 32) + 1), inverse[(1 * 4) + 2]);
  EXPECT_EQ(FloatAt(bytes, (2 * 32) + 4), 0.0F);
  EXPECT_EQ(WordAt(bytes, size_t{4} * 32), 0x22222222U);
}

TEST(SparseDriveTest, SceneTrackerStartsEachScene) {
  SceneTracker tracker;
  EXPECT_EQ(tracker.Stream("a"), "start");
  EXPECT_EQ(tracker.Stream("a"), "continue");
  EXPECT_EQ(tracker.Stream("b"), "start");
  EXPECT_EQ(tracker.Stream("a"), "start");
}

TEST(SparseDriveTest, ExpAndSigmoidAreCorrectlyRounded) {
  EXPECT_EQ(ExpFloat(0.0F), 1.0F);
  EXPECT_EQ(std::bit_cast<uint32_t>(ExpFloat(1.0F)), 0x402DF854U);
  EXPECT_EQ(std::bit_cast<uint32_t>(ExpFloat(-100.0F)), 27U);
  EXPECT_EQ(ExpFloat(89.0F), std::numeric_limits<float>::infinity());
  EXPECT_EQ(ExpFloat(-104.0F), 0.0F);
  EXPECT_EQ(SigmoidFloat(0.0F), 0.5F);
  EXPECT_EQ(std::bit_cast<uint32_t>(SigmoidFloat(-100.0F)), 27U);
  EXPECT_EQ(SigmoidFloat(-1000.0F), 0.0F);
  EXPECT_EQ(SigmoidFloat(104.0F), 1.0F);
  EXPECT_TRUE(
      std::isnan(SigmoidFloat(std::numeric_limits<float>::quiet_NaN())));
}

TEST(SparseDriveTest, PlanningFollowsTheCommandWithoutConfidentAgents) {
  Scene scene;
  scene.command = {0.0F, 1.0F, 0.0F};
  scene.plan_cls = Iota(kPlanQueries, 0.0F);
  scene.plan_cls[8] = 3.0F;
  scene.plan_reg =
      Iota(static_cast<size_t>(kPlanQueries) * kPlanSteps * 2, 0.25F);
  const Planning planning = DecodePlanning(scene.Inputs());
  EXPECT_EQ(planning.command, 1U);
  EXPECT_EQ(planning.mode, 2U);
  ASSERT_EQ(planning.trajectories.size(),
            size_t{kPlanQueries} * kPlanSteps * 2);
  const size_t first = ((size_t{8} * kPlanSteps) + 0) * 2;
  EXPECT_EQ(planning.trajectories[first], scene.plan_reg[first]);
  EXPECT_EQ(planning.trajectories[first + 2],
            scene.plan_reg[first] + scene.plan_reg[first + 2]);
  ASSERT_EQ(planning.final_trajectory.size(), size_t{kPlanSteps} * 2);
  EXPECT_EQ(planning.final_trajectory[2], planning.trajectories[first + 2]);
  ASSERT_EQ(planning.score.size(), kPlanQueries);
  EXPECT_EQ(planning.score[8], SigmoidFloat(3.0F));
  EXPECT_EQ(planning.score[0], 0.5F);
}

TEST(SparseDriveTest, RescoringDropsTheModesThatHitAnAgent) {
  const Planning planning = DecodePlanning(CrossingScene(1.0F).Inputs());
  EXPECT_EQ(planning.command, 0U);
  EXPECT_EQ(planning.mode, 3U);
  EXPECT_EQ(planning.score[0], SigmoidFloat(5.0F - 999.0F));
  EXPECT_EQ(planning.score[1], SigmoidFloat(-999.0F));
  EXPECT_EQ(planning.score[3], SigmoidFloat(4.0F));
  EXPECT_EQ(planning.final_trajectory[0], 10.0F);
  EXPECT_EQ(planning.final_trajectory[10], 60.0F);
}

TEST(SparseDriveTest, RescoringIgnoresAgentsBelowHalfConfidence) {
  const Planning planning = DecodePlanning(CrossingScene(-0.1F).Inputs());
  EXPECT_EQ(planning.mode, 0U);
  EXPECT_EQ(planning.score[0], SigmoidFloat(5.0F));
}

TEST(SparseDriveTest, RescoringKeepsTheScoresWhenEveryModeCollides) {
  Scene scene = CrossingScene(1.0F);
  for (uint32_t m = 0; m < kModes; ++m) {
    scene.Steer(m, 0.0F, 1.0F);
  }
  const Planning planning = DecodePlanning(scene.Inputs());
  EXPECT_EQ(planning.mode, 0U);
  EXPECT_EQ(planning.score[0], SigmoidFloat(5.0F));
}

TEST(SparseDriveTest, RescoringMovesTheAgentAlongItsBestMode) {
  Scene scene = CrossingScene(1.0F);
  scene.det_anchor[0] = 20.0F;
  scene.motion_cls[1] = 2.0F;
  for (uint32_t t = 0; t < kPlanSteps; ++t) {
    scene.motion_reg[(((size_t{1} * kMotionSteps) + t) * 2)] = -10.0F;
  }
  const Planning planning = DecodePlanning(scene.Inputs());
  EXPECT_EQ(planning.mode, 3U);
  EXPECT_EQ(planning.score[0], SigmoidFloat(5.0F - 999.0F));
}

std::filesystem::path TempPath(const std::string& name) {
  return std::filesystem::path(testing::TempDir()) /
         (name + "_" + std::to_string(getpid()) + ".safetensors");
}

TEST(SparseDriveTest, ReadsAFrame) {
  const std::filesystem::path path = TempPath("sparsedrive_frame");
  const std::vector<float> images =
      Iota(size_t{kCameras} * kImageValues, 0.001F);
  const std::vector<float> projection = Iota(size_t{kCameras} * 16, 1.0F);
  const std::vector<float> wh(size_t{kCameras} * 2, 704.0F);
  const std::vector<float> pose = Iota(16, 2.0F);
  const std::vector<float> command = {0.0F, 0.0F, 1.0F};
  const std::vector<int64_t> timestamp = {1533151603547590};
  ASSERT_TRUE(
      WriteSafeTensors(
          path,
          {{"img",
            FloatTensor({kCameras, kImageChannels, kImageRows, kImageColumns},
                        images)},
           {"projection_mat", FloatTensor({kCameras, 4, 4}, projection)},
           {"image_wh", FloatTensor({kCameras, 2}, wh)},
           {"T_global", FloatTensor({4, 4}, pose)},
           {"T_global_inv", FloatTensor({4, 4}, pose)},
           {"timestamp_us", Int64Tensor({}, timestamp)},
           {"cmd", FloatTensor({3}, command)}},
          {{"token", "t0"}, {"scene_token", "s0"}})
          .has_value());
  const auto file = SafeTensors::Open(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(file.has_value()) << file.error();
  const auto frame = ReadFrame(*file);
  ASSERT_TRUE(frame.has_value()) << frame.error();
  EXPECT_EQ(frame->images, images);
  EXPECT_EQ(frame->projection, projection);
  EXPECT_EQ(frame->timestamp_us, timestamp[0]);
  EXPECT_EQ(frame->command, command);
  EXPECT_EQ(frame->token, "t0");
  EXPECT_EQ(frame->scene, "s0");
}

class SparseDriveRunTest : public testing::Test {
 protected:
  struct Layout {
    std::map<std::string, Region, std::less<>> regions;
    uint32_t end = 4096;

    void Add(const std::string& name, uint32_t size) {
      regions[name] = {.offset = end, .size = size};
      end = (end + size + 4095) / 4096 * 4096;
    }
  };

  static Layout MakeLayout() {
    Layout layout;
    layout.Add("stem.in", static_cast<uint32_t>(kStemImageBytes));
    for (uint32_t c = 1; c < kCameras; ++c) {
      layout.Add(std::format("stem.in.cam{}", c),
                 static_cast<uint32_t>(kStemImageBytes));
    }
    for (const std::string_view head : {"det", "map"}) {
      layout.Add(std::format("{}.cam_in", head), 32 * kWordBytes);
      layout.Add(std::format("{}.kp_const", head), kCameras * kWordBytes);
    }
    layout.Add("fs.meta", 2 * kWordBytes);
    layout.Add("fs.tinv", 5 * kWordBytes);
    layout.Add("fs.flag", kWordBytes);
    const uint32_t rows = (928 + 128) * kWordBytes;
    for (const char* name : {"det.cls", "det.qt", "start.det.anchor"}) {
      layout.Add(name, rows);
    }
    layout.Add("start.det.x", (928 + 128) * 1024);
    layout.Add("start.map.x", 256 * 1024);
    layout.Add("start.map.anchor", 256 * 256);
    layout.Add("map.cls", 256 * kWordBytes);
    for (const char* name : {"motion.motion_cls", "motion.motion_reg"}) {
      layout.Add(name, 5408 * kWordBytes);
    }
    for (const char* name :
         {"motion.plan_cls", "motion.plan_reg", "motion.plan_status"}) {
      layout.Add(name, 32 * kWordBytes);
    }
    layout.Add("motion.state", 928 * kWordBytes);
    layout.Add("det.prevconf", 3712);
    layout.Add("det.keep", 2432);
    layout.Add("map.prevconf", 512);
    layout.Add("map.keep", 256);
    return layout;
  }

  void LoadProgram(ScriptedBus& bus) {
    device_ = std::make_unique<Device>(bus);
    runtime_ = std::make_unique<Runtime>(*device_, memory_, kBase);
    Program program{.base = kBase,
                    .image = std::vector<std::byte>(4096),
                    .instruction_offset = 0,
                    .instruction_count = 1,
                    .streams = {{"start", {.offset = 0, .count = 1}}},
                    .listings = {},
                    .regions = layout_.regions,
                    .parameters = {{"cameras", kCameras}}};
    ASSERT_TRUE(runtime_->Load(std::move(program)).has_value());
  }

  std::span<std::byte> Bytes(const std::string& name) {
    const Region& region = layout_.regions.at(name);
    return std::span<std::byte>(memory_).subspan(region.offset, region.size);
  }

  void PutRows(const std::string& name, uint32_t row_floats, uint32_t cols,
               const std::vector<float>& values) {
    const std::span<std::byte> bytes = Bytes(name);
    for (size_t i = 0; i < values.size(); ++i) {
      SetFloat(bytes, ((i / cols) * row_floats) + (i % cols), values[i]);
    }
  }

  void PutScene(const Scene& scene) {
    PutRows("det.cls", kRowFloats, kClasses, scene.det_cls);
    PutRows("start.det.anchor", kRowFloats, kAnchorWidth, scene.det_anchor);
    PutRows("motion.motion_cls", kRowFloats, 1, scene.motion_cls);
    PutRows("motion.motion_reg", kRowFloats, kMotionSteps * 2,
            scene.motion_reg);
    PutRows("motion.plan_cls", kRowFloats, 1, scene.plan_cls);
    PutRows("motion.plan_reg", kRowFloats, kPlanSteps * 2, scene.plan_reg);
    SetFloat(Bytes("fs.flag"), 0, std::bit_cast<float>(uint32_t{1}));
    SetFloat(Bytes("det.keep"), 607, std::bit_cast<float>(uint32_t{0xABCD}));
  }

  static Frame MakeFrame(const Scene& scene) {
    Frame frame{.images = Iota(size_t{kCameras} * kImageValues, 1.0F),
                .projection = Iota(size_t{kCameras} * 16, 1.0F),
                .image_wh = std::vector<float>(size_t{kCameras} * 2, 64.0F),
                .t_global = Iota(16, 1.0F),
                .t_global_inv = Iota(16, -1.0F),
                .timestamp_us = 7,
                .command = scene.command,
                .token = "t",
                .scene = "s"};
    return frame;
  }

  Layout layout_ = MakeLayout();
  std::vector<std::byte> memory_ = std::vector<std::byte>(layout_.end);
  std::unique_ptr<Device> device_;
  std::unique_ptr<Runtime> runtime_;
};

TEST_F(SparseDriveRunTest, PlacesTheInputsAndDecodesThePlanAtTheSignal) {
  ScriptedBus bus({{.id = kMotionDoneSignal, .value = kMotionDoneSignal}});
  LoadProgram(bus);
  const Scene scene = CrossingScene(1.0F);
  PutScene(scene);
  const Frame frame = MakeFrame(scene);
  const auto result = RunFrame(*runtime_, "start", frame, 1000ms);
  ASSERT_TRUE(result.has_value()) << result.error();
  EXPECT_EQ(bus.HostGo(), std::vector<uint32_t>{kMotionDoneSignal});
  EXPECT_EQ(result->planning.mode, 3U);
  EXPECT_EQ(result->det_x0, layout_.regions.at("start.det.x").offset);
  EXPECT_EQ(result->fs_flag, 1U);
  ASSERT_EQ(result->det_keep.size(), 608U);
  EXPECT_EQ(result->det_keep.back(), 0xABCDU);
  EXPECT_EQ(result->det_cls[0], 1.0F);
  EXPECT_EQ(result->state.size(), size_t{kStateRows} * kStateWidth);
  const auto stem = Bytes("stem.in.cam3");
  EXPECT_EQ(FloatAt(stem, kStemPadding),
            frame.images[size_t{3} * kImageValues]);
  EXPECT_EQ(FloatAt(Bytes("map.kp_const"), (5 * 32) + kKpWidthLane), 64.0F);
  EXPECT_EQ(WordAt(Bytes("fs.meta"), 0), 7U);
  EXPECT_EQ(FloatAt(Bytes("fs.tinv"), (1 * 32) + 2),
            frame.t_global_inv[(2 * 4) + 1]);
  EXPECT_EQ(FloatAt(Bytes("det.cam_in"), (4 * 32) + 3),
            frame.projection[(4 * 16) + 3]);
  const auto tensors = OutputTensors(frame, *result);
  ASSERT_EQ(tensors.size(), 25U);
  EXPECT_EQ(tensors[0].first, "final_planning");
}

TEST_F(SparseDriveRunTest, ReportsAStreamThatDoesNotSignal) {
  ScriptedBus bus({});
  LoadProgram(bus);
  const Scene scene;
  PutScene(scene);
  const auto result = RunFrame(*runtime_, "start", MakeFrame(scene), 1000ms);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("did not signal"), std::string::npos);
}

TEST_F(SparseDriveRunTest, ReportsAnUnexpectedSignal) {
  ScriptedBus bus({{.id = 2, .value = 5}});
  LoadProgram(bus);
  const Scene scene;
  PutScene(scene);
  const auto result = RunFrame(*runtime_, "start", MakeFrame(scene), 1000ms);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("unexpected signal 2"), std::string::npos);
}

}  // namespace
}  // namespace hotarubi::sparsedrive
