#include "timing.hpp"

#include <gtest/gtest.h>

#include <cstdint>

#include "dma_engine.hpp"
#include "gemm_array.hpp"
#include "timeline.hpp"
#include "trace_record.hpp"
#include "vec_engine.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

TEST(TraceRecordTest, CompletionRecordsRoundTrip) {
  const TraceRecord record{.index = 0x12345678,
                           .issue = 0xFFFFFFF0,
                           .completion = 0x00000010,
                           .source = Source::kVec,
                           .opcode = 5,
                           .wait = 0xD,
                           .flags = 0x1,
                           .type = RecordType::kCompletion,
                           .statistics = {}};
  const Word128 word = EncodeRecord(record);
  EXPECT_EQ(word.Field(0, 32), 0x12345678U);
  EXPECT_EQ(word.Field(32, 32), 0xFFFFFFF0U);
  EXPECT_EQ(word.Field(64, 32), 0x10U);
  EXPECT_EQ(word.Field(96, 4), 2U);
  EXPECT_EQ(word.Field(100, 8), 5U);
  EXPECT_EQ(word.Field(108, 4), 0xDU);
  EXPECT_EQ(word.Field(112, 4), 1U);
  EXPECT_EQ(word.Field(116, 12), 0U);
  EXPECT_EQ(DecodeRecord(word), record);
}

TEST(TraceRecordTest, StatisticsRecordsRoundTripAndSaturate) {
  TraceRecord record{.index = 7,
                     .issue = 0,
                     .completion = 0,
                     .source = Source::kDma,
                     .opcode = 0,
                     .wait = 0,
                     .flags = 0,
                     .type = RecordType::kStatistics,
                     .statistics = {.a = 0x123456, .b = 1, .c = 2, .d = 3}};
  const Word128 word = EncodeRecord(record);
  EXPECT_EQ(word.Field(0, 32), 7U);
  EXPECT_EQ(word.Field(32, 4), 0U);
  EXPECT_EQ(word.Field(36, 4), 0U);
  EXPECT_EQ(word.Field(40, 24), 0x123456U);
  EXPECT_EQ(word.Field(64, 24), 1U);
  EXPECT_EQ(word.Field(88, 24), 2U);
  EXPECT_EQ(word.Field(112, 4), 0U);
  EXPECT_EQ(word.Field(116, 4), 1U);
  EXPECT_EQ(word.Field(120, 8), 3U);
  EXPECT_EQ(DecodeRecord(word), record);
  record.statistics = {.a = 0x5000000, .b = 0xFFFFFFFF, .c = 0, .d = 300};
  const TraceRecord saturated = DecodeRecord(EncodeRecord(record));
  EXPECT_EQ(saturated.statistics.a, 0xFFFFFFU);
  EXPECT_EQ(saturated.statistics.b, 0xFFFFFFU);
  EXPECT_EQ(saturated.statistics.c, 0U);
  EXPECT_EQ(saturated.statistics.d, 255U);
}

TEST(TraceRecordTest, SourcesFollowTheOpcode) {
  EXPECT_EQ(SourceOf(1), Source::kDma);
  EXPECT_EQ(SourceOf(2), Source::kDma);
  EXPECT_EQ(SourceOf(4), Source::kGemm);
  EXPECT_EQ(SourceOf(5), Source::kVec);
  for (const uint32_t opcode : {0U, 3U, 6U, 7U, 8U, 200U}) {
    EXPECT_EQ(SourceOf(opcode), Source::kSequencer) << opcode;
  }
}

TEST(CostModelTest, GemmCostIsTheArrayTimePlusTheFixedCost) {
  const CostModel model;
  const GemmCommand command{.a_base = 0,
                            .a_stride = 24,
                            .kb = 24,
                            .m = 128,
                            .w_base = 0,
                            .w_stride = 24,
                            .n8 = 32,
                            .c_base = 0,
                            .wmode = 0,
                            .acc = false};
  EXPECT_EQ(GemmArrayCycles(command), 32U * 4 * 24 * 8);
  const Cost cost = model.Gemm(command);
  EXPECT_EQ(cost.cycles, (32U * 4 * 24 * 8) + 51);
  EXPECT_EQ(cost.statistics.a, 32U * 4 * 24 * 8);
  EXPECT_EQ(cost.statistics.b, 0U);
  EXPECT_EQ(cost.statistics.d, 0U);
}

TEST(CostModelTest, DmaCostScalesWithBeatsAndBursts) {
  TimingParameters parameters;
  parameters.dma_fixed_cycles = 10;
  parameters.dma_read_cycles_per_kilobeat = 2048;
  parameters.dma_read_burst_cycles = 5;
  parameters.dma_write_cycles_per_kilobeat = 1024;
  parameters.dma_write_burst_cycles = 7;
  parameters.dma_gather_entry_cycles = 3;
  parameters.dma_gather_row_cycles = 11;
  const CostModel model(parameters);
  const DmaActivity activity{.read_beats = 512,
                             .write_beats = 100,
                             .read_bursts = 32,
                             .write_bursts = 8,
                             .null_entries = 5,
                             .gather_entries = 7,
                             .gather_rows = 2};
  const Cost cost = model.Dma(activity);
  EXPECT_EQ(cost.cycles, 10U + 1024 + 160 + 100 + 56 + 21 + 22);
  EXPECT_EQ(cost.statistics.a, 512U);
  EXPECT_EQ(cost.statistics.b, 100U);
  EXPECT_EQ(cost.statistics.c, 0U);
  EXPECT_EQ(cost.statistics.d, 5U);
}

TEST(CostModelTest, VecCostFollowsTheStructuralModel) {
  TimingParameters parameters;
  parameters.vec_fixed_cycles = 1;
  parameters.vec_pass_cycles = 2;
  parameters.vec_word_cycles = 4;
  parameters.vec_issue_tail_cycles = 3;
  parameters.vec_drain_cycles = 10;
  parameters.vec_drain_reduction_cycles = 5;
  parameters.vec_drain_quantized_cycles = 6;
  parameters.vec_scalar_arithmetic_cycles = 9;
  parameters.vec_scalar_immediate_cycles = 2;
  parameters.vec_scalar_start_cycles = 2;
  parameters.vec_salu_cycles = 3;
  parameters.vec_smem_cycles = 5;
  parameters.vec_branch_cycles = 7;
  const CostModel model(parameters);
  VecActivity activity;
  activity.passes = 3;
  activity.words = 20;
  activity.drains = 3;
  activity.drains_reduction = 2;
  activity.drains_quantized = 1;
  activity.overlaps = 1;
  activity.overlap_stall_words = 5;
  activity.scalars.at(1) = 1;
  activity.scalars.at(4) = 1;
  activity.rn = 1;
  activity.salu = 4;
  activity.smem = 2;
  activity.branches = 1;
  const Cost cost = model.Vec(activity);
  EXPECT_EQ(cost.statistics.a, 100U - 9);
  EXPECT_EQ(cost.statistics.b, 30U + 10 + 6);
  EXPECT_EQ(cost.statistics.c,
            ((8U + 19 + 12) * 9) + ((1U + 2 + 2) * 2) + (3U * 2));
  EXPECT_EQ(cost.statistics.d, 3U);
  EXPECT_EQ(cost.cycles, 1U + 6 + cost.statistics.a + cost.statistics.b +
                             cost.statistics.c + 12 + 10 + 7);
}

TEST(CostModelTest, ScalarProgramsMatchTheBoard) {
  const CostModel model;
  VecActivity layernorm;
  layernorm.scalars.at(1) = 8;
  layernorm.scalars.at(2) = 8;
  layernorm.rn = 1;
  EXPECT_EQ(model.ScalarCycles(layernorm), 2858U);
  VecActivity softmax;
  softmax.scalars.at(3) = 8;
  softmax.scalars.at(4) = 8;
  EXPECT_EQ(model.ScalarCycles(softmax), 1448U);
  VecActivity rmsnorm;
  rmsnorm.scalars.at(2) = 8;
  rmsnorm.rn = 1;
  EXPECT_EQ(model.ScalarCycles(rmsnorm), 2250U);
}

TEST(EstimateTest, RawLoadsCountBeatsPerWord) {
  const DmaActivity fbuf = EstimateDma({.store = false,
                                        .ddr_addr = 0,
                                        .mem = 2,
                                        .onchip = 0,
                                        .rows = 3,
                                        .row_words = 5,
                                        .ddr_stride = 0,
                                        .mode = 0,
                                        .flags = 0,
                                        .tok = 0});
  EXPECT_EQ(fbuf.read_beats, 3U * 5 * 8);
  EXPECT_EQ(fbuf.write_beats, 0U);
  EXPECT_EQ(fbuf.read_bursts, 3U * 3);
  EXPECT_EQ(fbuf.write_bursts, 0U);
  const DmaActivity abuf_store = EstimateDma({.store = true,
                                              .ddr_addr = 0,
                                              .mem = 0,
                                              .onchip = 0,
                                              .rows = 2,
                                              .row_words = 2,
                                              .ddr_stride = 0,
                                              .mode = 0,
                                              .flags = 0,
                                              .tok = 0});
  EXPECT_EQ(abuf_store.write_beats, 2U * 2 * 9);
  EXPECT_EQ(abuf_store.write_bursts, 2U * 2);
  EXPECT_EQ(abuf_store.read_beats, 0U);
}

TEST(EstimateTest, IndexedAndRowFormatTransfersIncludeTheList) {
  const DmaActivity indexed = EstimateDma({.store = false,
                                           .ddr_addr = 0,
                                           .mem = 2,
                                           .onchip = 0,
                                           .rows = 1,
                                           .row_words = 9,
                                           .ddr_stride = 0,
                                           .mode = 0,
                                           .flags = 1,
                                           .tok = 0});
  EXPECT_EQ(indexed.read_beats, 3U + (9U * 8));
  const DmaActivity rows = EstimateDma({.store = false,
                                        .ddr_addr = 0,
                                        .mem = 0,
                                        .onchip = 0,
                                        .rows = 2,
                                        .row_words = 2,
                                        .ddr_stride = 0,
                                        .mode = 0,
                                        .flags = 2,
                                        .tok = 24});
  EXPECT_EQ(rows.read_beats, 4U * 50);
  const DmaActivity split = EstimateDma({.store = false,
                                         .ddr_addr = 0,
                                         .mem = 1,
                                         .onchip = 0,
                                         .rows = 4,
                                         .row_words = 3,
                                         .ddr_stride = 0,
                                         .mode = 1,
                                         .flags = 0,
                                         .tok = 0});
  EXPECT_EQ(split.read_beats, 12U * 9);
}

TEST(EstimateTest, GatherCountsListCoefficientsAndRows) {
  const DmaActivity gather = EstimateDma({.store = false,
                                          .ddr_addr = 0,
                                          .mem = 2,
                                          .onchip = 0,
                                          .rows = 9,
                                          .row_words = 8,
                                          .ddr_stride = 0,
                                          .mode = 3,
                                          .flags = 2,
                                          .tok = 0});
  EXPECT_EQ(gather.gather_entries, 9U * 16);
  EXPECT_EQ(gather.gather_rows, 9U * 16);
  EXPECT_EQ(gather.read_beats, 1U + 72 + (2U * 8 * 8) + (144U * 17));
}

TEST(EstimateTest, DdrExtentsCoverEveryRowOfATransfer) {
  const DmaCommand raw{.ddr_addr = 0x100000,
                       .mem = 2,
                       .rows = 4,
                       .row_words = 2,
                       .ddr_stride = 0x1000};
  EXPECT_EQ(EstimateDdrExtent(raw),
            (DdrExtent{.lo = 0x100000, .hi = 0x100000 + (3 * 0x1000) + 256}));
  const DmaCommand abuf_store{
      .store = true, .ddr_addr = 0x2000, .mem = 0, .rows = 1, .row_words = 3};
  EXPECT_EQ(EstimateDdrExtent(abuf_store),
            (DdrExtent{.lo = 0x2000, .hi = 0x2000 + (3 * 9 * 16)}));
  const DmaCommand split{
      .ddr_addr = 0x3000, .mem = 1, .rows = 2, .row_words = 4, .mode = 1};
  EXPECT_EQ(EstimateDdrExtent(split),
            (DdrExtent{.lo = 0x3000, .hi = 0x3000 + (4 * 9 * 16)}));
  const DmaCommand rows{.ddr_addr = 0x4000,
                        .mem = 2,
                        .rows = 1,
                        .row_words = 4,
                        .flags = 2,
                        .tok = 8};
  EXPECT_EQ(EstimateDdrExtent(rows),
            (DdrExtent{.lo = 0x4000, .hi = 0x4000 + (4 * 8 * 8 * 16)}));
  const DmaCommand indexed{.ddr_addr = 0x5000, .rows = 1, .flags = 1};
  EXPECT_TRUE(EstimateDdrExtent(indexed).whole);
  const DmaCommand gather{.ddr_addr = 0x6000, .rows = 1, .mode = 3};
  EXPECT_TRUE(EstimateDdrExtent(gather).whole);
  const DmaCommand empty{.ddr_addr = 0x7000, .rows = 0, .row_words = 1};
  EXPECT_EQ(EstimateDdrExtent(empty), (DdrExtent{.lo = 0x7000, .hi = 0x7000}));
}

TEST(EstimateTest, FixedOperationActivityExecutesTheRomProgram) {
  const VecActivity add = FixedOperationActivity({.op = 0,
                                                  .quantize = false,
                                                  .src = 0,
                                                  .src2 = 0,
                                                  .dst = 0,
                                                  .rows_q = 32,
                                                  .n8 = 32,
                                                  .p_base = 0,
                                                  .aux = 0,
                                                  .imm = 0});
  EXPECT_EQ(add.passes, 1U);
  EXPECT_EQ(add.words, 1024U);
  EXPECT_EQ(add.drains, 1U);
  EXPECT_EQ(add.drains_reduction, 0U);
  EXPECT_EQ(add.drains_quantized, 0U);
  EXPECT_EQ(add.overlaps, 0U);
  const VecActivity softmax = FixedOperationActivity({.op = 11,
                                                      .quantize = true,
                                                      .src = 0,
                                                      .src2 = 0,
                                                      .dst = 0,
                                                      .rows_q = 8,
                                                      .n8 = 128,
                                                      .p_base = 0,
                                                      .aux = 1024,
                                                      .imm = 2048});
  EXPECT_EQ(softmax.passes, 24U);
  EXPECT_EQ(softmax.words, 24U * 128);
  EXPECT_EQ(softmax.drains_reduction, 16U);
  EXPECT_EQ(softmax.drains_quantized, 8U);
  EXPECT_EQ(softmax.drains, 24U);
  EXPECT_EQ(softmax.overlaps, 0U);
  EXPECT_EQ(softmax.scalars.at(3), 8U);
  EXPECT_EQ(softmax.scalars.at(4), 8U);
  const VecActivity layernorm = FixedOperationActivity({.op = 6,
                                                        .quantize = true,
                                                        .src = 0,
                                                        .src2 = 1536,
                                                        .dst = 0,
                                                        .rows_q = 8,
                                                        .n8 = 96,
                                                        .p_base = 0,
                                                        .aux = 0,
                                                        .imm = 0x358637BD});
  EXPECT_EQ(layernorm.passes, 40U);
  EXPECT_EQ(layernorm.drains, 32U);
  EXPECT_EQ(layernorm.overlaps, 8U);
  EXPECT_EQ(layernorm.overlap_stall_words, 0U);
  EXPECT_EQ(layernorm.scalars.at(1), 8U);
  EXPECT_EQ(layernorm.scalars.at(2), 8U);
  EXPECT_EQ(layernorm.rn, 1U);
  const VecActivity run = FixedOperationActivity({.op = 16,
                                                  .quantize = false,
                                                  .src = 0,
                                                  .src2 = 0,
                                                  .dst = 0,
                                                  .rows_q = 8,
                                                  .n8 = 96,
                                                  .p_base = 0,
                                                  .aux = 300,
                                                  .imm = 0});
  EXPECT_EQ(run.passes, 0U);
}

TEST(ParseTimingTest, AppliesAssignmentsAndRejectsUnknownNames) {
  const auto parsed = ParseTiming("gemm_fixed_cycles=7,vec_smem_cycles=4");
  ASSERT_TRUE(parsed.has_value()) << parsed.error();
  EXPECT_EQ(parsed->gemm_fixed_cycles, 7U);
  EXPECT_EQ(parsed->vec_smem_cycles, 4U);
  EXPECT_EQ(parsed->dma_fixed_cycles, TimingParameters{}.dma_fixed_cycles);
  EXPECT_FALSE(ParseTiming("nothing=1").has_value());
  EXPECT_FALSE(ParseTiming("gemm_fixed_cycles").has_value());
  EXPECT_FALSE(ParseTiming("gemm_fixed_cycles=x").has_value());
  EXPECT_TRUE(ParseTiming("").has_value());
}

TEST(TimelineTest, IssuesInOrderAndHonoursWaitBits) {
  TimingParameters parameters;
  parameters.start_cycles = 10;
  Timeline timeline(parameters);
  timeline.Start();
  const Span dma = timeline.Issue(Source::kDma, 0, 100);
  EXPECT_EQ(dma.issue, 10U);
  EXPECT_EQ(dma.completion, 110U);
  const Span gemm = timeline.Issue(Source::kGemm, 0, 50);
  EXPECT_EQ(gemm.issue, 11U);
  EXPECT_EQ(gemm.completion, 61U);
  const Span vec = timeline.Issue(Source::kVec, 0x1, 20);
  EXPECT_EQ(vec.issue, 111U);
  EXPECT_EQ(vec.completion, 131U);
  const Span gemm2 = timeline.Issue(Source::kGemm, 0, 5);
  EXPECT_EQ(gemm2.issue, 112U);
  const Span halt = timeline.Issue(Source::kSequencer, 0xD, 0);
  EXPECT_EQ(halt.issue, 132U);
  EXPECT_EQ(halt.completion, 132U);
  EXPECT_EQ(timeline.Cycles(), 133U);
  EXPECT_EQ(timeline.Issued(), 5U);
  EXPECT_EQ(timeline.Busy(Source::kDma), 100U);
  EXPECT_EQ(timeline.Busy(Source::kGemm), 55U);
  EXPECT_EQ(timeline.Busy(Source::kVec), 20U);
  EXPECT_EQ(timeline.Starve(), 10U);
  EXPECT_EQ(timeline.Stall(), 133U - 10 - 5);
}

TEST(TimelineTest, EarliestStartDelaysTheDataPhaseButNotTheIssue) {
  Timeline timeline;
  timeline.Start();
  const Span gemm = timeline.Issue(Source::kGemm, 0, 1000);
  const Span store =
      timeline.Issue(Source::kDma, 0, 30, timeline.EngineFree(Source::kGemm));
  EXPECT_EQ(store.issue, gemm.issue + 1);
  EXPECT_EQ(store.start, gemm.completion);
  EXPECT_EQ(store.completion, gemm.completion + 30);
  EXPECT_EQ(timeline.Busy(Source::kDma), store.completion - store.issue);
}

}  // namespace
}  // namespace hotarubi::iss
