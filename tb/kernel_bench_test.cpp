#include "kernel_bench.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "analysis.hpp"
#include "ddr_image.hpp"
#include "input.hpp"
#include "isa.hpp"
#include "npu_bench.hpp"
#include "output.hpp"
#include "predict.hpp"
#include "program.hpp"
#include "trace_record.hpp"

namespace hotarubi::tb {
namespace {

constexpr uint64_t kRowBytes = 1024;

class KernelBenchTest : public testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::path(testing::TempDir()) /
            ("kernel_" + std::to_string(getpid()));
    std::filesystem::remove_all(root_);
    compiler::DdrImage image(uint64_t{1} << 20);
    compiler::Program program(Ddr::kBase);
    const uint64_t instructions = image.Alloc("instr", 4096);
    const uint64_t input = image.Alloc("in", kRowBytes);
    const uint64_t output = image.Alloc("out", kRowBytes);
    std::vector<std::byte> values(kRowBytes);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<std::byte>(i % 7);
    }
    image.Put(input, values);
    const compiler::Tile in = program.Alloc(compiler::Mem::kFbuf, 8);
    const compiler::Tile out = program.Alloc(compiler::Mem::kFbuf, 8);
    {
      const compiler::Program::Scope scope(program, "setup");
      program.Load({.ddr_offset = input,
                    .mem = compiler::Mem::kFbuf,
                    .onchip = in.At(),
                    .rows = 1,
                    .row_words = 8,
                    .ddr_stride = 0},
                   "ld in");
    }
    {
      const compiler::Program::Scope scope(program, "kernel");
      program.Vec({.op = compiler::VecOp::kRelu,
                   .src = in.At(),
                   .dst = out.At(),
                   .rows_q = 8,
                   .n8 = 1},
                  "relu");
      program.Store({.ddr_offset = output,
                     .mem = compiler::Mem::kFbuf,
                     .onchip = out.At(),
                     .rows = 1,
                     .row_words = 8,
                     .ddr_stride = 0},
                    "st out");
    }
    program.Halt();
    program.Finalize({.window = 1});
    image.Put(instructions, program.Bytes());
    count_ = static_cast<uint32_t>(program.Size());
    std::vector<compiler::Stream> streams;
    streams.push_back({.name = "inference",
                       .program = std::move(program),
                       .offset = instructions});
    compiler::ReserveTraceRegion(image, streams);
    const compiler::Compiled compiled{.streams = std::move(streams),
                                      .image = std::move(image),
                                      .parameters = {}};
    ASSERT_TRUE(
        compiler::WriteCompiled(root_ / "compiled", compiled).has_value());
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  std::filesystem::path root_;
  uint32_t count_ = 0;
};

TEST_F(KernelBenchTest, RunsAWindowOnTheRtlAndWritesAComparableDirectory) {
  const KernelOptions options{.program = root_ / "compiled",
                              .stream = "inference",
                              .first = 1,
                              .last = 2,
                              .preamble = {0},
                              .statistics = true,
                              .out = root_ / "kernel",
                              .max_cycles = 2000000,
                              .seed = 3};
  const auto counters = RunKernel(options);
  ASSERT_TRUE(counters.has_value()) << counters.error();
  EXPECT_EQ(counters->issued, 4U);
  EXPECT_EQ(counters->trace_count, 4U + 3U);
  const auto loaded = trace::LoadCompiled(root_ / "kernel");
  ASSERT_TRUE(loaded.has_value()) << loaded.error();
  const auto stream = trace::LoadStream(*loaded, std::string(kKernelStream));
  ASSERT_TRUE(stream.has_value()) << stream.error();
  ASSERT_EQ(stream->count, 4U);
  EXPECT_EQ(stream->instructions[0].tag, "setup");
  EXPECT_EQ(stream->instructions[1].tag, "kernel");
  EXPECT_EQ(stream->instructions[1].note, "relu");
  EXPECT_EQ(stream->instructions[3].opcode, 7U);
  const auto records =
      trace::LoadRecords(root_ / "kernel" / "kernel.trace.bin");
  ASSERT_TRUE(records.has_value()) << records.error();
  const auto counters_file =
      trace::LoadCounters(root_ / "kernel" / "kernel.counters.json");
  ASSERT_TRUE(counters_file.has_value()) << counters_file.error();
  EXPECT_TRUE(counters_file->statistics);
  const auto analysis = trace::Analyze(*stream, *records, *counters_file);
  ASSERT_TRUE(analysis.has_value()) << analysis.error();
  EXPECT_EQ(analysis->events.size(), 4U);
  EXPECT_TRUE(analysis->has_statistics);
  EXPECT_EQ(analysis->categories.at("LD FBUF").statistics.a, 64U);
  EXPECT_EQ(analysis->categories.at("ST FBUF").statistics.b, 64U);
  EXPECT_EQ(analysis->categories.at("VEC RELU").statistics.d, 1U);
  const auto predicted =
      trace::Predict(*loaded, std::string(kKernelStream), true);
  ASSERT_TRUE(predicted.has_value()) << predicted.error();
  EXPECT_EQ(predicted->counters.issued, 4U);
  EXPECT_EQ(predicted->counters.trace_count, 7U);
  const std::vector<iss::TraceRecord> predicted_records =
      trace::DecodeRecords(predicted->records);
  const auto prediction =
      trace::Analyze(*stream, predicted_records, std::nullopt);
  ASSERT_TRUE(prediction.has_value()) << prediction.error();
  EXPECT_EQ(prediction->categories.at("LD FBUF").statistics.a, 64U);
  EXPECT_EQ(prediction->categories.at("VEC RELU").statistics.d, 1U);
}

TEST_F(KernelBenchTest, RejectsWindowsOutsideTheStream) {
  KernelOptions options{.program = root_ / "compiled",
                        .stream = "inference",
                        .first = 2,
                        .last = count_,
                        .preamble = {},
                        .statistics = false,
                        .out = root_ / "kernel",
                        .max_cycles = 1000,
                        .seed = 1};
  EXPECT_FALSE(RunKernel(options).has_value());
  options.last = 1;
  EXPECT_FALSE(RunKernel(options).has_value());
  options.first = 0;
  options.preamble = {count_};
  EXPECT_FALSE(RunKernel(options).has_value());
  options.preamble = {};
  options.stream = "missing";
  EXPECT_FALSE(RunKernel(options).has_value());
}

}  // namespace
}  // namespace hotarubi::tb
