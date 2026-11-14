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
#include "hotarubi/json.hpp"
#include "input.hpp"
#include "isa.hpp"
#include "output.hpp"
#include "predict.hpp"
#include "program.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "trace_record.hpp"

namespace hotarubi::trace {
namespace {

constexpr uint64_t kBase = 0x30000000;
constexpr uint64_t kWordA = 144;
constexpr uint64_t kWordF = 128;

class TraceTest : public testing::Test {
 protected:
  void SetUp() override {
    directory_ = std::filesystem::path(testing::TempDir()) /
                 ("trace_" + std::to_string(getpid()));
    std::filesystem::remove_all(directory_);
    compiler::DdrImage image(uint64_t{4} << 20);
    compiler::Program program(kBase);
    const uint64_t instructions = image.Alloc("instr", 4096);
    const uint64_t weights = image.Alloc("w", kWordA * 8);
    const uint64_t activations = image.Alloc("a", kWordA * 64);
    const uint64_t output = image.Alloc("out", kWordF * 64);
    const compiler::Tile w = program.Alloc(compiler::Mem::kWbuf, 8);
    const compiler::Tile a = program.Alloc(compiler::Mem::kAbuf, 64);
    const compiler::Tile c = program.Alloc(compiler::Mem::kFbuf, 8);
    {
      const compiler::Program::Scope scope(program, "layer/load");
      program.Load({.ddr_offset = weights,
                    .mem = compiler::Mem::kWbuf,
                    .onchip = w.At(),
                    .rows = 1,
                    .row_words = 8,
                    .ddr_stride = 0},
                   "ld W");
      program.Load({.ddr_offset = activations,
                    .mem = compiler::Mem::kAbuf,
                    .onchip = a.At(),
                    .rows = 8,
                    .row_words = 8,
                    .ddr_stride = 144 * 8},
                   "ld A");
    }
    {
      const compiler::Program::Scope scope(program, "layer/gemm");
      program.Gemm({.a_base = a.At(),
                    .a_stride = 8,
                    .kb = 8,
                    .m = 32,
                    .w_base = w.At(),
                    .w_stride = 8,
                    .n8 = 1,
                    .c_base = c.At()},
                   "gemm");
    }
    {
      const compiler::Program::Scope scope(program, "layer/post");
      program.Vec({.op = compiler::VecOp::kRelu,
                   .src = c.At(),
                   .dst = c.At(),
                   .rows_q = 8,
                   .n8 = 1},
                  "relu");
      program.Store({.ddr_offset = output,
                     .mem = compiler::Mem::kFbuf,
                     .onchip = c.At(),
                     .rows = 8,
                     .row_words = 1,
                     .ddr_stride = 128},
                    "st out");
    }
    program.Signal(1, 1, "signal");
    program.WaitHost(1, "wait");
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
    ASSERT_TRUE(compiler::WriteCompiled(directory_, compiled).has_value());
  }

  void TearDown() override { std::filesystem::remove_all(directory_); }

  [[nodiscard]] LoadedProgram Loaded() const {
    auto loaded = LoadCompiled(directory_);
    EXPECT_TRUE(loaded.has_value()) << loaded.error();
    return std::move(*loaded);
  }

  std::filesystem::path directory_;
  uint32_t count_ = 0;
};

TEST_F(TraceTest, LoadsTheStreamWithTagsAndNotes) {
  const LoadedProgram loaded = Loaded();
  const auto stream = LoadStream(loaded, "inference");
  ASSERT_TRUE(stream.has_value()) << stream.error();
  ASSERT_EQ(stream->count, count_);
  EXPECT_EQ(stream->instructions[0].tag, "layer/load");
  EXPECT_EQ(stream->instructions[0].note, "ld W");
  EXPECT_EQ(stream->instructions[2].source, iss::Source::kGemm);
  EXPECT_EQ(stream->instructions[3].tag, "layer/post");
  EXPECT_EQ(stream->instructions[5].tag, "");
  EXPECT_EQ(Category(stream->instructions[0]), "LD WBUF");
  EXPECT_EQ(Category(stream->instructions[1]), "LD ABUF");
  EXPECT_EQ(Category(stream->instructions[2]), "GEMM FP4");
  EXPECT_EQ(Category(stream->instructions[3]), "VEC RELU");
  EXPECT_EQ(Category(stream->instructions[4]), "ST FBUF");
  EXPECT_EQ(Category(stream->instructions[5]), "SIGNAL");
  EXPECT_EQ(Category(stream->instructions[6]), "WAIT_HOST");
  EXPECT_EQ(Category(stream->instructions[7]), "HALT");
  EXPECT_EQ(Streaming(stream->instructions[0]), 72U);
  EXPECT_EQ(Streaming(stream->instructions[2]), 64U);
  EXPECT_EQ(Streaming(stream->instructions[3]), 32U);
  EXPECT_EQ(Streaming(stream->instructions[4]), 64U);
  EXPECT_FALSE(LoadStream(loaded, "missing").has_value());
  ASSERT_EQ(stream->instructions[2].reads.size(), 2U);
  EXPECT_EQ(stream->instructions[2].reads[0].mem, iss::Mem::kAbuf);
  EXPECT_EQ(stream->instructions[2].reads[0].hi, 64U);
  ASSERT_EQ(stream->instructions[2].writes.size(), 1U);
  EXPECT_EQ(stream->instructions[2].writes[0].mem, iss::Mem::kFbuf);
  EXPECT_EQ(stream->instructions[2].writes[0].hi, 8U);
  EXPECT_TRUE(stream->instructions[5].reads.empty());
}

TEST_F(TraceTest, PredictChecksTheListedRangesAndDumpsTheAccesses) {
  const auto prediction = Predict(Loaded(), "inference", false, {}, true);
  ASSERT_TRUE(prediction.has_value()) << prediction.error();
  const std::string& accesses = prediction->accesses;
  EXPECT_NE(accesses.find(R"({"index":2,"reads":[["ABUF",0,64],["WBUF",0,8]],)"
                          R"("writes":[["FBUF",0,8]]})"),
            std::string::npos);
  EXPECT_NE(accesses.find(R"({"index":7,"reads":[],"writes":[]})"),
            std::string::npos);
  const std::filesystem::path listing =
      directory_ / compiler::ListingFileName("inference");
  auto text = ReadText(listing);
  ASSERT_TRUE(text.has_value());
  const size_t position = text->find(R"(["WBUF",0,8])");
  ASSERT_NE(position, std::string::npos);
  text->replace(position, 12, R"(["WBUF",0,4])");
  ASSERT_TRUE(WriteText(listing, *text).has_value());
  const auto narrowed = Predict(Loaded(), "inference", false);
  ASSERT_FALSE(narrowed.has_value());
  EXPECT_NE(narrowed.error().find(
                "instruction 0 writes WBUF word 4 outside its listed ranges"),
            std::string::npos)
      << narrowed.error();
}

TEST_F(TraceTest, PredictsAndAnalyzesAStream) {
  const LoadedProgram loaded = Loaded();
  const auto prediction = Predict(loaded, "inference", true);
  ASSERT_TRUE(prediction.has_value()) << prediction.error();
  EXPECT_EQ(prediction->counters.issued, count_);
  EXPECT_EQ(prediction->counters.trace_count, count_ + 5);
  const auto stream = LoadStream(loaded, "inference");
  ASSERT_TRUE(stream.has_value());
  const std::vector<iss::TraceRecord> records =
      DecodeRecords(prediction->records);
  ASSERT_EQ(records.size(), count_ + 5);
  const CountersFile counters{.stream = "inference",
                              .statistics = true,
                              .counters = prediction->counters};
  const auto analysis = Analyze(*stream, records, counters);
  ASSERT_TRUE(analysis.has_value()) << analysis.error();
  EXPECT_EQ(analysis->events.size(), count_);
  EXPECT_EQ(analysis->total_cycles, prediction->counters.cycles);
  EXPECT_TRUE(analysis->has_statistics);
  uint64_t occupancy = 0;
  for (const auto& [mask, cycles] : analysis->occupancy) {
    occupancy += cycles;
  }
  EXPECT_EQ(occupancy, analysis->total_cycles);
  EXPECT_EQ(analysis->busy.at(1), analysis->categories.at("GEMM FP4").span);
  EXPECT_EQ(analysis->categories.at("GEMM FP4").count, 1U);
  EXPECT_EQ(analysis->categories.at("GEMM FP4").span, 64U + 51);
  EXPECT_EQ(analysis->categories.at("GEMM FP4").statistics.a, 64U);
  EXPECT_EQ(analysis->categories.at("LD ABUF").statistics.a, 8U * 8 * 9);
  EXPECT_EQ(analysis->categories.at("ST FBUF").statistics.b, 64U);
  EXPECT_EQ(analysis->categories.at("VEC RELU").statistics.d, 1U);
  EXPECT_EQ(analysis->events[2].blocker, "DMA");
  EXPECT_GT(analysis->events[2].gap, 0U);
  EXPECT_EQ(analysis->events[3].blocker, "GEMM");
  const std::map<std::string, Aggregate> depth1 = ByTag(*analysis, 1);
  ASSERT_TRUE(depth1.contains("layer"));
  EXPECT_EQ(depth1.at("layer").count, 5U);
  EXPECT_EQ(depth1.at("").count, 3U);
  const std::map<std::string, Aggregate> depth2 = ByTag(*analysis, 2);
  EXPECT_EQ(depth2.at("layer/load").count, 2U);
  EXPECT_EQ(depth2.at("layer/post").count, 2U);
  const ReportOptions options{.depth = 2, .clock_hz = 250000000, .gaps = 5};
  const std::string text = ReportText(*analysis, options);
  EXPECT_NE(text.find("GEMM FP4"), std::string::npos);
  EXPECT_NE(text.find("layer/gemm"), std::string::npos);
  const auto json = Json::Parse(ReportJson(*analysis, options).Dump());
  ASSERT_TRUE(json.has_value()) << json.error();
  EXPECT_EQ(json->Find("instructions")->Integer(), count_);
  const auto perfetto = Json::Parse(Perfetto(*analysis, 250000000));
  ASSERT_TRUE(perfetto.has_value()) << perfetto.error();
  EXPECT_EQ(perfetto->Find("traceEvents")->Items().size(), 4U + count_);
  EXPECT_NE(KernelsText(*analysis, options).find("layer/post"),
            std::string::npos);
  EXPECT_NE(DiffText(*analysis, *analysis, options).find("+0.000"),
            std::string::npos);
  const Calibration calibration =
      Calibrate({{.analysis = &*analysis, .stream = &*stream}});
  EXPECT_EQ(calibration.parameters.gemm_fixed_cycles, 51U);
  EXPECT_EQ(calibration.parameters.start_cycles, analysis->first_issue);
}

TEST_F(TraceTest, CheckRejectsInconsistentRecords) {
  const LoadedProgram loaded = Loaded();
  const auto prediction = Predict(loaded, "inference", false);
  ASSERT_TRUE(prediction.has_value()) << prediction.error();
  const auto stream = LoadStream(loaded, "inference");
  ASSERT_TRUE(stream.has_value());
  const std::vector<iss::TraceRecord> records =
      DecodeRecords(prediction->records);
  ASSERT_TRUE(Check(*stream, records, std::nullopt).has_value());
  std::vector<iss::TraceRecord> missing = records;
  missing.pop_back();
  EXPECT_NE(Check(*stream, missing, std::nullopt)
                .error()
                .find("no completion record"),
            std::string::npos);
  std::vector<iss::TraceRecord> twice = records;
  twice.push_back(records[0]);
  EXPECT_NE(Check(*stream, twice, std::nullopt).error().find("two completion"),
            std::string::npos);
  std::vector<iss::TraceRecord> wrong = records;
  wrong[2].opcode = 5;
  EXPECT_NE(Check(*stream, wrong, std::nullopt).error().find("does not match"),
            std::string::npos);
  std::vector<iss::TraceRecord> beyond = records;
  beyond[0].index = count_;
  EXPECT_NE(Check(*stream, beyond, std::nullopt).error().find("beyond"),
            std::string::npos);
  std::vector<iss::TraceRecord> overlap = records;
  overlap[1].issue = records[0].issue;
  EXPECT_NE(
      Check(*stream, overlap, std::nullopt).error().find("before or with"),
      std::string::npos);
  CountersFile counters{.stream = "inference",
                        .statistics = false,
                        .counters = prediction->counters};
  ASSERT_TRUE(Check(*stream, records, counters).has_value());
  counters.counters.cycles += 1;
  EXPECT_NE(Check(*stream, records, counters).error().find("cycle counter"),
            std::string::npos);
  counters.counters.cycles -= 1;
  counters.counters.trace_count += 1;
  EXPECT_NE(Check(*stream, records, counters).error().find("trace records"),
            std::string::npos);
  counters.counters.trace_count -= 1;
  counters.stream = "other";
  EXPECT_NE(Check(*stream, records, counters).error().find("belong to"),
            std::string::npos);
}

TEST_F(TraceTest, ReadsAndWritesCountersFiles) {
  const LoadedProgram loaded = Loaded();
  const auto prediction = Predict(loaded, "inference", true);
  ASSERT_TRUE(prediction.has_value()) << prediction.error();
  const std::filesystem::path path = directory_ / "inference.counters.json";
  ASSERT_TRUE(WriteText(path, CountersJson("inference", true,
                                           prediction->counters, "predicted"))
                  .has_value());
  const auto counters = LoadCounters(path);
  ASSERT_TRUE(counters.has_value()) << counters.error();
  EXPECT_EQ(counters->stream, "inference");
  EXPECT_TRUE(counters->statistics);
  EXPECT_EQ(counters->counters.cycles, prediction->counters.cycles);
  EXPECT_EQ(counters->counters.trace_count, prediction->counters.trace_count);
  EXPECT_FALSE(LoadCounters(directory_ / "none.json").has_value());
}

TEST(TagPrefixTest, CutsTagsAtTheRequestedDepth) {
  EXPECT_EQ(TagPrefix("a/b/c", 1), "a");
  EXPECT_EQ(TagPrefix("a/b/c", 2), "a/b");
  EXPECT_EQ(TagPrefix("a/b/c", 5), "a/b/c");
  EXPECT_EQ(TagPrefix("", 1), "");
  EXPECT_EQ(TagPrefix("a", 0), "");
}

}  // namespace
}  // namespace hotarubi::trace
