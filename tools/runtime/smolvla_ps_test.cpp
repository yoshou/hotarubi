#include "smolvla_ps.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "hotarubi/driver.hpp"
#include "hotarubi/formats.hpp"
#include "hotarubi/runtime.hpp"
#include "hotarubi/safetensors.hpp"
#include "scripted_bus.hpp"

namespace hotarubi::smolvla {
namespace {

using std::chrono_literals::operator""ms;

constexpr uint64_t kBase = 0x38700000;
constexpr uint32_t kLanguageRows = 12;
constexpr uint32_t kLmRows = 192;
constexpr uint32_t kActionRows = 64;

Image Uniform(uint32_t height, uint32_t width, uint8_t value) {
  return {.height = height,
          .width = width,
          .rgb = std::vector<uint8_t>(
              static_cast<size_t>(height) * width * kChannels, value)};
}

float Pixel(const std::vector<float>& pixels, uint32_t c, uint32_t y,
            uint32_t x) {
  return pixels.at((((static_cast<size_t>(c) * kImageSize) + y) * kImageSize) +
                   x);
}

Matrix Counting(uint32_t rows, uint32_t cols, float scale) {
  Matrix matrix{
      .rows = rows,
      .cols = cols,
      .bits = std::vector<uint32_t>(static_cast<size_t>(rows) * cols)};
  for (size_t i = 0; i < matrix.bits.size(); ++i) {
    matrix.bits[i] = std::bit_cast<uint32_t>(static_cast<float>(i) * scale);
  }
  return matrix;
}

std::filesystem::path TempPath(const std::string& name) {
  return std::filesystem::path(testing::TempDir()) /
         (name + "_" + std::to_string(getpid()) + ".safetensors");
}

TEST(SmolVlaTest, ScalesPixelsFromZeroOneToMinusOneOne) {
  Image image = Uniform(kImageSize, kImageSize, 0);
  image.rgb[0] = 0;
  image.rgb[1] = 255;
  image.rgb[2] = 51;
  const std::vector<float> pixels = Preprocess(image);
  ASSERT_EQ(pixels.size(),
            static_cast<size_t>(kChannels) * kImageSize * kImageSize);
  EXPECT_EQ(Pixel(pixels, 0, 0, 0), -1.0F);
  EXPECT_EQ(Pixel(pixels, 1, 0, 0), 1.0F);
  EXPECT_EQ(Pixel(pixels, 2, 0, 0), ((51.0F / 255.0F) * 2.0F) - 1.0F);
  EXPECT_EQ(Pixel(pixels, 1, 0, 1), -1.0F);
}

TEST(SmolVlaTest, PadsAtTheTopLeftAfterAnAspectPreservingResize) {
  const std::vector<float> wide = Preprocess(Uniform(480, 640, 255));
  for (uint32_t c = 0; c < kChannels; ++c) {
    EXPECT_EQ(Pixel(wide, c, 0, 0), -1.0F);
    EXPECT_EQ(Pixel(wide, c, 127, 511), -1.0F);
    EXPECT_EQ(Pixel(wide, c, 128, 0), 1.0F);
    EXPECT_EQ(Pixel(wide, c, 511, 511), 1.0F);
  }
  const std::vector<float> tall = Preprocess(Uniform(640, 480, 255));
  EXPECT_EQ(Pixel(tall, 0, 511, 127), -1.0F);
  EXPECT_EQ(Pixel(tall, 0, 0, 128), 1.0F);
}

TEST(SmolVlaTest, ResizesWithHalfPixelCentres) {
  Image image = Uniform(256, 256, 255);
  for (uint32_t y = 0; y < 256; ++y) {
    for (uint32_t c = 0; c < kChannels; ++c) {
      image.rgb[(static_cast<size_t>(y) * 256 * kChannels) + c] = 0;
    }
  }
  const std::vector<float> pixels = Preprocess(image);
  for (const uint32_t y : {0U, 100U, 511U}) {
    EXPECT_EQ(Pixel(pixels, 1, y, 0), -1.0F);
    EXPECT_EQ(Pixel(pixels, 1, y, 1), -0.5F);
    EXPECT_EQ(Pixel(pixels, 1, y, 2), 0.5F);
    EXPECT_EQ(Pixel(pixels, 1, y, 3), 1.0F);
  }
}

TEST(SmolVlaTest, DownscalingByTwoAveragesEachBlock) {
  Image image = Uniform(1024, 1024, 0);
  for (uint32_t y = 0; y < 1024; ++y) {
    for (uint32_t x = 0; x < 1024; ++x) {
      if ((x + y) % 2 == 1) {
        for (uint32_t c = 0; c < kChannels; ++c) {
          image.rgb[(((static_cast<size_t>(y) * 1024) + x) * kChannels) + c] =
              255;
        }
      }
    }
  }
  const std::vector<float> pixels = Preprocess(image);
  EXPECT_EQ(Pixel(pixels, 0, 0, 0), 0.0F);
  EXPECT_EQ(Pixel(pixels, 2, 300, 17), 0.0F);
}

TEST(SmolVlaTest, PatchRowsFollowTheDesignOrder) {
  std::vector<float> pixels(static_cast<size_t>(kChannels) * kImageSize *
                            kImageSize);
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<float>(i);
  }
  const Matrix rows = PatchRows(pixels);
  ASSERT_EQ(rows.rows, kPatchTokens);
  ASSERT_EQ(rows.cols, kPatchValues);
  const auto value = [](uint32_t c, uint32_t py, uint32_t px, uint32_t ky,
                        uint32_t kx) {
    return std::bit_cast<uint32_t>(static_cast<float>(
        (c * kImageSize * kImageSize) + (((py * kPatch) + ky) * kImageSize) +
        (px * kPatch) + kx));
  };
  const uint32_t p = (((1 * 4) + 2) * 64) + (3 * 8) + 5;
  const uint32_t s = (((4 * 3) + 1) * 32) + (4 * 5) + 2;
  EXPECT_EQ(rows.At(p, 0), value(0, s / 32, s % 32, 0, 0));
  EXPECT_EQ(rows.At(p, (2 * 256) + (15 * 16) + 15),
            value(2, s / 32, s % 32, 15, 15));
  EXPECT_EQ(rows.At(p, (1 * 256) + (7 * 16) + 9),
            value(1, s / 32, s % 32, 7, 9));
  EXPECT_EQ(rows.At(0, 17), value(0, 0, 0, 1, 1));
  EXPECT_EQ(rows.At(1023, 767), value(2, 31, 31, 15, 15));
}

TEST(SmolVlaTest, PatchInputHoldsEachCameraAsMxint8Words) {
  const std::vector<Image> cameras = {Uniform(kImageSize, kImageSize, 255),
                                      Uniform(kImageSize, kImageSize, 0)};
  const std::vector<std::byte> bytes = PatchInput(cameras);
  const size_t camera_bytes = static_cast<size_t>(kPatchTokens / 4) *
                              (kPatchValues / kBlock) * kWordAbufBytes;
  ASSERT_EQ(bytes.size(), 2 * camera_bytes);
  EXPECT_EQ(bytes[0], std::byte{0x40});
  EXPECT_EQ(bytes[127], std::byte{0x40});
  EXPECT_EQ(bytes[128], std::byte{127});
  EXPECT_EQ(bytes[131], std::byte{127});
  EXPECT_EQ(bytes[camera_bytes], std::byte{0xC0});
  EXPECT_EQ(bytes[camera_bytes + 128], std::byte{127});
}

TEST(SmolVlaTest, LanguageRowsFollowTheImageTokens) {
  const Matrix language = Counting(kLanguageRows, kLanguageWidth, 1.0F);
  const Matrix input = LanguageInput(kLmRows, 128, language);
  ASSERT_EQ(input.rows, kLmRows);
  EXPECT_EQ(input.At(128, 0), language.At(0, 0));
  EXPECT_EQ(input.At(139, 959), language.At(11, 959));
  EXPECT_EQ(input.At(127, 959), 0U);
  EXPECT_EQ(input.At(140, 0), 0U);
}

TEST(SmolVlaTest, StateIsZeroPaddedInTheFirstRow) {
  const std::vector<float> state = {0.5F, -1.5F};
  const Matrix input = StateInput(state);
  ASSERT_EQ(input.rows, kStateRows);
  ASSERT_EQ(input.cols, kStateWidth);
  EXPECT_EQ(input.At(0, 0), std::bit_cast<uint32_t>(0.5F));
  EXPECT_EQ(input.At(0, 1), std::bit_cast<uint32_t>(-1.5F));
  EXPECT_EQ(input.At(0, 2), 0U);
  EXPECT_EQ(input.At(1, 0), 0U);
}

TEST(SmolVlaTest, NoiseFillsTheFirstRows) {
  const Matrix noise = Counting(kChunk, kActionWidth, 0.25F);
  const Matrix input = NoiseInput(kActionRows, noise);
  ASSERT_EQ(input.rows, kActionRows);
  EXPECT_EQ(input.At(49, 31), noise.At(49, 31));
  EXPECT_EQ(input.At(50, 0), 0U);
}

TEST(SmolVlaTest, GeneratedNoiseIsReproducibleAndStandardNormal) {
  std::mt19937_64 first(7);
  std::mt19937_64 second(7);
  const Matrix a = GenerateNoise(first);
  const Matrix b = GenerateNoise(second);
  EXPECT_EQ(a.bits, b.bits);
  double sum = 0;
  double squares = 0;
  for (const uint32_t bits : a.bits) {
    const auto value = static_cast<double>(std::bit_cast<float>(bits));
    sum += value;
    squares += value * value;
  }
  const auto n = static_cast<double>(a.bits.size());
  EXPECT_NEAR(sum / n, 0.0, 0.1);
  EXPECT_NEAR(squares / n, 1.0, 0.15);
  EXPECT_NE(GenerateNoise(first).bits, a.bits);
}

TEST(SmolVlaTest, ActionsAreTheFirstSixColumnsOfTheChunk) {
  const Matrix x_t = Counting(kActionRows, kActionWidth, 1.0F);
  const Matrix actions = Actions(x_t);
  ASSERT_EQ(actions.rows, kChunk);
  ASSERT_EQ(actions.cols, kActionDims);
  EXPECT_EQ(actions.At(0, 5), x_t.At(0, 5));
  EXPECT_EQ(actions.At(49, 0), x_t.At(49, 0));
}

TEST(SmolVlaTest, LanguageTableLooksUpTheRowsOfATask) {
  const std::filesystem::path path = TempPath("language");
  std::vector<float> rows(static_cast<size_t>(kLanguageRows) * kLanguageWidth);
  for (size_t i = 0; i < rows.size(); ++i) {
    rows[i] = static_cast<float>(i);
  }
  ASSERT_TRUE(
      WriteSafeTensors(path,
                       {{"pick up the cube",
                         FloatTensor({kLanguageRows, kLanguageWidth}, rows)}},
                       {})
          .has_value());
  const auto table = LanguageTable::Open(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(table.has_value()) << table.error();
  const auto found = table->Rows("pick up the cube");
  ASSERT_TRUE(found.has_value()) << found.error();
  EXPECT_EQ(found->rows, kLanguageRows);
  EXPECT_EQ(found->At(11, 959), std::bit_cast<uint32_t>(rows.back()));
  EXPECT_FALSE(table->Rows("stack the cubes").has_value());
}

TEST(SmolVlaTest, ReadsAFrameAndGeneratesMissingNoise) {
  const std::filesystem::path path = TempPath("frame");
  const std::vector<float> state = {1, 2, 3, 4, 5, 6};
  const auto image = [](uint8_t value) {
    return Tensor{.dtype = DType::kU8,
                  .shape = {4, 8, kChannels},
                  .data = std::vector<std::byte>(size_t{4} * 8 * kChannels,
                                                 std::byte{value})};
  };
  ASSERT_TRUE(WriteSafeTensors(path,
                               {{"camera1", image(10)},
                                {"camera2", image(20)},
                                {"state", FloatTensor({6}, state)}},
                               {{"task", "pick"}})
                  .has_value());
  const auto file = SafeTensors::Open(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(file.has_value()) << file.error();
  std::mt19937_64 generator(3);
  const auto frame = ReadFrame(*file, generator);
  ASSERT_TRUE(frame.has_value()) << frame.error();
  EXPECT_EQ(frame->cameras[0].height, 4U);
  EXPECT_EQ(frame->cameras[1].width, 8U);
  EXPECT_EQ(frame->cameras[1].rgb[5], 20);
  EXPECT_EQ(frame->state, state);
  EXPECT_EQ(frame->task, "pick");
  EXPECT_TRUE(frame->generated_noise);
  std::mt19937_64 same(3);
  EXPECT_EQ(frame->noise.bits, GenerateNoise(same).bits);
}

TEST(SmolVlaTest, RejectsAFrameWithoutItsSecondCamera) {
  const std::filesystem::path path = TempPath("camera");
  const std::vector<float> state = {1};
  ASSERT_TRUE(
      WriteSafeTensors(path,
                       {{"camera1", Tensor{.dtype = DType::kU8,
                                           .shape = {1, 1, kChannels},
                                           .data = std::vector<std::byte>(3)}},
                        {"state", FloatTensor({1}, state)}},
                       {{"task", "pick"}})
          .has_value());
  const auto file = SafeTensors::Open(path);
  std::filesystem::remove(path);
  ASSERT_TRUE(file.has_value()) << file.error();
  std::mt19937_64 generator(0);
  const auto frame = ReadFrame(*file, generator);
  ASSERT_FALSE(frame.has_value());
  EXPECT_NE(frame.error().find("camera2"), std::string::npos);
}

class SmolVlaRunTest : public testing::Test {
 protected:
  static constexpr uint32_t kPatchOffset = 0x10000;
  static constexpr uint32_t kLanguageOffset = 0x200000;
  static constexpr uint32_t kStateOffset = 0x300000;
  static constexpr uint32_t kNoiseOffset = 0x310000;
  static constexpr uint32_t kOutputOffset = 0x320000;

  static Program MakeProgram(int64_t cameras) {
    const auto patch_bytes = static_cast<uint32_t>(
        2 * (kPatchTokens / 4) * (kPatchValues / kBlock) * kWordAbufBytes);
    return {
        .base = kBase,
        .image = std::vector<std::byte>(4096),
        .instruction_offset = 0,
        .instruction_count = 1,
        .streams = {},
        .regions = {{"patch.in", {.offset = kPatchOffset, .size = patch_bytes}},
                    {"xl.0",
                     {.offset = kLanguageOffset,
                      .size = kLmRows * kLanguageWidth * 4}},
                    {"state.in", {.offset = kStateOffset, .size = 4096}},
                    {"x_t.0", {.offset = kNoiseOffset, .size = 8192}},
                    {"x_t.10", {.offset = kOutputOffset, .size = 8192}}},
        .parameters = {
            {"cameras", cameras}, {"n_lang", kLanguageRows}, {"steps", 10}}};
  }

  static Frame MakeFrame() {
    Frame frame{
        .cameras = {Uniform(kImageSize, kImageSize, 255), Uniform(480, 640, 0)},
        .state = {0.25F, 0.5F},
        .noise = Counting(kChunk, kActionWidth, 0.5F),
        .generated_noise = false,
        .task = "pick"};
    return frame;
  }

  [[nodiscard]] std::span<const std::byte> At(uint32_t offset,
                                              size_t size) const {
    return std::span<const std::byte>(memory_).subspan(offset, size);
  }

  ScriptedBus bus_{{}};
  Device device_{bus_};
  std::vector<std::byte> memory_ = std::vector<std::byte>(size_t{4} << 20);
  Runtime runtime_{device_, memory_, kBase};
};

TEST_F(SmolVlaRunTest, WritesTheInputsRunsAndReadsTheActions) {
  ASSERT_TRUE(runtime_.Load(MakeProgram(2)).has_value());
  const Matrix x_t = Counting(kActionRows, kActionWidth, 2.0F);
  const std::vector<std::byte> packed = PackFbuf(x_t);
  std::ranges::copy(packed, memory_.begin() + kOutputOffset);
  const Frame frame = MakeFrame();
  const Matrix language = Counting(kLanguageRows, kLanguageWidth, 3.0F);
  const auto result = RunFrame(runtime_, frame, language, 1000ms);
  ASSERT_TRUE(result.has_value()) << result.error();
  EXPECT_EQ(bus_.Starts(), 1U);
  EXPECT_EQ(result->actions.bits, Actions(x_t).bits);
  EXPECT_EQ(result->x_t.bits, x_t.bits);
  const std::vector<std::byte> patches = PatchInput(frame.cameras);
  EXPECT_TRUE(std::ranges::equal(At(kPatchOffset, patches.size()), patches));
  const std::vector<std::byte> lm =
      PackFbuf(LanguageInput(kLmRows, 2 * kTokensPerCamera, language));
  EXPECT_TRUE(std::ranges::equal(At(kLanguageOffset, lm.size()), lm));
  const std::vector<std::byte> state = PackFbuf(StateInput(frame.state));
  EXPECT_TRUE(std::ranges::equal(At(kStateOffset, state.size()), state));
  const std::vector<std::byte> noise =
      PackFbuf(NoiseInput(kActionRows, frame.noise));
  EXPECT_TRUE(std::ranges::equal(At(kNoiseOffset, noise.size()), noise));
}

TEST_F(SmolVlaRunTest, RejectsAProgramForAnotherNumberOfCameras) {
  ASSERT_TRUE(runtime_.Load(MakeProgram(1)).has_value());
  const auto result =
      RunFrame(runtime_, MakeFrame(),
               Counting(kLanguageRows, kLanguageWidth, 1.0F), 1000ms);
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().find("cameras"), std::string::npos);
  EXPECT_EQ(bus_.Starts(), 0U);
}

}  // namespace
}  // namespace hotarubi::smolvla
