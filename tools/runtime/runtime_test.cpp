#include "hotarubi/runtime.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "bench_bus.hpp"
#include "hotarubi/driver.hpp"
#include "npu_bench.hpp"
#include "npu_isa.hpp"

namespace hotarubi {
namespace {

using std::chrono_literals::operator""ms;

constexpr uint32_t kInputOffset = 0x1000;
constexpr uint32_t kOutputOffset = 0x2000;
constexpr uint32_t kWordBytes = 128;
constexpr uint32_t kScale = 4;
constexpr uint32_t kTwo = 0x40000000U;

Program DoubleProgram() {
  const std::vector<tb::Instruction> instructions = {
      tb::DmaLoad({.ddr_addr = tb::Ddr::kBase + kInputOffset,
                   .mem = tb::Mem::kFbuf,
                   .onchip = 0,
                   .rows = 1,
                   .row_words = 1,
                   .ddr_stride = kWordBytes}),
      tb::Vec({.op = kScale,
               .src = 0,
               .dst = 1,
               .rows_q = 1,
               .n8 = 1,
               .imm = kTwo,
               .wait = tb::kWaitDma}),
      tb::DmaStore({.ddr_addr = tb::Ddr::kBase + kOutputOffset,
                    .mem = tb::Mem::kFbuf,
                    .onchip = 1,
                    .rows = 1,
                    .row_words = 1,
                    .ddr_stride = kWordBytes,
                    .wait = tb::kWaitVec}),
      tb::Halt()};
  return {
      .base = tb::Ddr::kBase,
      .image = Serialize(instructions),
      .instruction_offset = 0,
      .instruction_count = static_cast<uint32_t>(instructions.size()),
      .streams = {{"double",
                   {.offset = 0,
                    .count = static_cast<uint32_t>(instructions.size())}}},
      .regions = {{"input", {.offset = kInputOffset, .size = kWordBytes}},
                  {"output", {.offset = kOutputOffset, .size = kWordBytes}}},
      .parameters = {{"scale", 2}}};
}

class RuntimeTest : public testing::Test {
 protected:
  tb::NpuBench bench_;
  BenchBus bus_{bench_};
  Device device_{bus_};
  Runtime runtime_{device_, BenchMemory(bench_), tb::Ddr::kBase};
};

TEST_F(RuntimeTest, RunsAProgramWithInputsAndOutputs) {
  const auto loaded = runtime_.Load(DoubleProgram());
  ASSERT_TRUE(loaded.has_value()) << loaded.error();
  for (const float offset : {-7.5F, 100.25F}) {
    std::array<float, 32> input{};
    std::array<float, 32> doubled{};
    for (uint32_t i = 0; i < input.size(); ++i) {
      input.at(i) = static_cast<float>(i) + offset;
      doubled.at(i) = 2.0F * input.at(i);
    }
    const auto written = runtime_.Write("input", FloatBytes(input));
    ASSERT_TRUE(written.has_value()) << written.error();
    const auto counters = runtime_.Run(1000ms);
    ASSERT_TRUE(counters.has_value()) << counters.error();
    EXPECT_EQ(counters->issued, 4U);
    std::vector<std::byte> output(kWordBytes);
    const auto read = runtime_.Read("output", output);
    ASSERT_TRUE(read.has_value()) << read.error();
    EXPECT_EQ(output, FloatBytes(doubled));
  }
}

TEST_F(RuntimeTest, RunsANamedStreamAndReportsAnUnknownOne) {
  ASSERT_TRUE(runtime_.Load(DoubleProgram()).has_value());
  std::array<float, 32> input{};
  input.fill(3.0F);
  std::array<float, 32> doubled{};
  doubled.fill(6.0F);
  ASSERT_TRUE(runtime_.Write("input", FloatBytes(input)).has_value());
  const auto counters = runtime_.Run("double", 1000ms);
  ASSERT_TRUE(counters.has_value()) << counters.error();
  EXPECT_EQ(counters->issued, 4U);
  std::vector<std::byte> output(kWordBytes);
  ASSERT_TRUE(runtime_.Read("output", output).has_value());
  EXPECT_EQ(output, FloatBytes(doubled));
  const auto missing = runtime_.Run("triple", 1000ms);
  ASSERT_FALSE(missing.has_value());
  EXPECT_NE(missing.error().find("triple"), std::string::npos);
}

TEST_F(RuntimeTest, FindsRegionsAndParameters) {
  EXPECT_FALSE(runtime_.FindRegion("input").has_value());
  ASSERT_TRUE(runtime_.Load(DoubleProgram()).has_value());
  const auto input = runtime_.FindRegion("input");
  ASSERT_TRUE(input.has_value()) << input.error();
  EXPECT_EQ(input->offset, kInputOffset);
  EXPECT_EQ(input->size, kWordBytes);
  EXPECT_FALSE(runtime_.FindRegion("weights").has_value());
  EXPECT_EQ(runtime_.Parameter("scale"), 2);
  EXPECT_FALSE(runtime_.Parameter("steps").has_value());
}

TEST_F(RuntimeTest, RejectsAProgramBuiltForAnotherAddress) {
  Program program = DoubleProgram();
  program.base += 0x100000;
  const auto loaded = runtime_.Load(program);
  ASSERT_FALSE(loaded.has_value());
  EXPECT_NE(loaded.error().find("built for"), std::string::npos);
}

TEST_F(RuntimeTest, RejectsAnImageLargerThanTheMemory) {
  Program program = DoubleProgram();
  program.image.resize(tb::Ddr::kSize + 16);
  EXPECT_FALSE(runtime_.Load(program).has_value());
}

TEST_F(RuntimeTest, RejectsAMisplacedInstructionStream) {
  Program program = DoubleProgram();
  program.instruction_offset = 16;
  EXPECT_FALSE(runtime_.Load(program).has_value());
  program.instruction_offset = 0;
  program.instruction_count = 100;
  EXPECT_FALSE(runtime_.Load(program).has_value());
  program = DoubleProgram();
  program.streams["double"].offset = 16;
  EXPECT_FALSE(runtime_.Load(program).has_value());
  program.streams["double"] = {.offset = 0, .count = 0};
  EXPECT_FALSE(runtime_.Load(program).has_value());
}

TEST_F(RuntimeTest, RejectsARegionOutsideTheMemoryOrMisaligned) {
  Program program = DoubleProgram();
  program.regions["output"].offset = tb::Ddr::kSize - 64;
  EXPECT_FALSE(runtime_.Load(program).has_value());
  program.regions["output"].offset = kOutputOffset + 8;
  EXPECT_FALSE(runtime_.Load(program).has_value());
}

TEST_F(RuntimeTest, RejectsUnknownRegionsAndSizes) {
  ASSERT_TRUE(runtime_.Load(DoubleProgram()).has_value());
  const std::vector<std::byte> half(kWordBytes / 2);
  EXPECT_FALSE(runtime_.Write("weights", half).has_value());
  EXPECT_FALSE(runtime_.Write("input", half).has_value());
}

TEST_F(RuntimeTest, RejectsUseBeforeAProgramIsLoaded) {
  const std::vector<std::byte> word(kWordBytes);
  EXPECT_FALSE(runtime_.Write("input", word).has_value());
  EXPECT_FALSE(runtime_.Run(1000ms).has_value());
}

class ProgramFiles {
 public:
  ProgramFiles()
      : directory_(std::filesystem::path(testing::TempDir()) /
                   (std::string(testing::UnitTest::GetInstance()
                                    ->current_test_info()
                                    ->name()) +
                    "_" + std::to_string(getpid()))) {
    std::filesystem::create_directories(directory_);
  }
  ProgramFiles(const ProgramFiles&) = delete;
  ProgramFiles& operator=(const ProgramFiles&) = delete;
  ProgramFiles(ProgramFiles&&) = delete;
  ProgramFiles& operator=(ProgramFiles&&) = delete;
  ~ProgramFiles() { std::filesystem::remove_all(directory_); }

  void Write(const Program& program, uint64_t image_bytes) const {
    std::ofstream image(directory_ / "image.bin", std::ios::binary);
    for (const std::byte value : program.image) {
      image.put(static_cast<char>(value));
    }
    std::ofstream(directory_ / "meta.json") << std::format(
        R"({{"base": {}, "n_instr": {}, "instr_off": {}, "image_bytes": {},)"
        R"( "steps": 10, "streams": {{"double": [{}, {}]}},)"
        R"( "regions": {{"input": [{}, {}], "output": [{}, {}]}},)"
        R"( "outputs": ["output"]}})",
        program.base, program.instruction_count, program.instruction_offset,
        image_bytes, program.instruction_offset, program.instruction_count,
        kInputOffset, kWordBytes, kOutputOffset, kWordBytes);
  }

  [[nodiscard]] const std::filesystem::path& Directory() const {
    return directory_;
  }

 private:
  std::filesystem::path directory_;
};

TEST_F(RuntimeTest, RunsAProgramLoadedFromItsFiles) {
  const Program compiled = DoubleProgram();
  const ProgramFiles files;
  files.Write(compiled, compiled.image.size());
  auto program = LoadProgram(files.Directory());
  ASSERT_TRUE(program.has_value()) << program.error();
  EXPECT_EQ(program->base, compiled.base);
  EXPECT_EQ(program->instruction_count, compiled.instruction_count);
  EXPECT_EQ(program->image, compiled.image);
  EXPECT_EQ(program->regions.at("output").offset, kOutputOffset);
  EXPECT_EQ(program->streams.at("double").count, compiled.instruction_count);
  ASSERT_EQ(program->parameters.size(), 1U);
  EXPECT_EQ(program->parameters.at("steps"), 10);
  ASSERT_TRUE(runtime_.Load(std::move(*program)).has_value());
  std::array<float, 32> input{};
  std::array<float, 32> doubled{};
  for (uint32_t i = 0; i < input.size(); ++i) {
    input.at(i) = 0.5F * static_cast<float>(i);
    doubled.at(i) = static_cast<float>(i);
  }
  ASSERT_TRUE(runtime_.Write("input", FloatBytes(input)).has_value());
  ASSERT_TRUE(runtime_.Run(1000ms).has_value());
  std::vector<std::byte> output(kWordBytes);
  ASSERT_TRUE(runtime_.Read("output", output).has_value());
  EXPECT_EQ(output, FloatBytes(doubled));
}

TEST(LoadProgramTest, RejectsAnImageOfTheWrongSizeOrMissingFiles) {
  const Program compiled = DoubleProgram();
  const ProgramFiles files;
  EXPECT_FALSE(LoadProgram(files.Directory()).has_value());
  files.Write(compiled, compiled.image.size() + 16);
  const auto program = LoadProgram(files.Directory());
  ASSERT_FALSE(program.has_value());
  EXPECT_NE(program.error().find("image.bin"), std::string::npos);
}

}  // namespace
}  // namespace hotarubi
