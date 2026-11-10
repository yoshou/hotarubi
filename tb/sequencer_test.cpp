#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "npu_bench.hpp"
#include "npu_isa.hpp"

namespace hotarubi::tb {
namespace {

constexpr uint32_t kRegCtrl = 0x00;
constexpr uint32_t kRegStatus = 0x04;
constexpr uint32_t kRegInstrBase = 0x08;
constexpr uint32_t kRegInstrCount = 0x0C;
constexpr uint32_t kRegIrqStatus = 0x10;
constexpr uint32_t kRegIrqEnable = 0x14;
constexpr uint32_t kRegSignal = 0x18;
constexpr uint32_t kRegSignalValue = 0x1C;
constexpr uint32_t kRegIssued = 0x20;
constexpr uint32_t kRegCycles = 0x24;
constexpr uint32_t kRegGemmCycles = 0x28;
constexpr uint32_t kRegVecCycles = 0x2C;
constexpr uint32_t kRegDmaCycles = 0x30;
constexpr uint32_t kRegTraceBase = 0x3C;
constexpr uint32_t kRegTraceCtrl = 0x40;
constexpr uint32_t kRegTraceCount = 0x44;
constexpr uint32_t kRegHostGo = 0x48;

constexpr uint32_t kStatusRunning = 0x01;
constexpr uint32_t kStatusHalted = 0x02;
constexpr uint32_t kStatusError = 0x04;
constexpr uint32_t kStatusWaiting = 0x08;
constexpr uint32_t kStatusIbufEmpty = 0x80;

constexpr uint32_t kIrqSignal = 0x1;
constexpr uint32_t kIrqHalt = 0x2;
constexpr uint32_t kIrqIllegal = 0x4;

constexpr uint32_t kTraceArea = Ddr::kBase + 0x200000U;

Instruction Illegal(uint32_t opcode) { return Header(opcode, 0, 0); }

TEST(SequencerTest, RegistersReadBack) {
  NpuBench bench(61);
  EXPECT_EQ(bench.ReadReg(kRegStatus), kStatusIbufEmpty);
  bench.WriteReg(kRegInstrBase, 0x12345600U);
  bench.WriteReg(kRegInstrCount, 77);
  bench.WriteReg(kRegIrqEnable, 0x5);
  bench.WriteReg(kRegTraceBase, 0x40000010U);
  bench.WriteReg(kRegTraceCtrl, 1);
  bench.WriteReg(kRegHostGo, 0xCAFE);
  EXPECT_EQ(bench.ReadReg(kRegInstrBase), 0x12345600U);
  EXPECT_EQ(bench.ReadReg(kRegInstrCount), 77U);
  EXPECT_EQ(bench.ReadReg(kRegIrqEnable), 0x5U);
  EXPECT_EQ(bench.ReadReg(kRegTraceBase), 0x40000010U);
  EXPECT_EQ(bench.ReadReg(kRegTraceCtrl) & 1U, 1U);
  EXPECT_EQ(bench.ReadReg(kRegHostGo), 0xCAFEU);
}

TEST(SequencerTest, HaltRaisesMaskableInterrupt) {
  NpuBench bench(62);
  bench.WriteReg(kRegIrqEnable, kIrqHalt);
  const std::vector<Instruction> program = {Nop(), Nop(), Halt()};
  ASSERT_TRUE(bench.Run(program));
  EXPECT_EQ(bench.ReadReg(kRegStatus), kStatusHalted | kStatusIbufEmpty);
  EXPECT_EQ(bench.ReadReg(kRegIrqStatus), kIrqHalt);
  EXPECT_EQ(bench.ReadReg(kRegIssued), 3U);
  EXPECT_GT(bench.ReadReg(kRegCycles), 0U);
  EXPECT_TRUE(bench.Irq());
  bench.WriteReg(kRegIrqEnable, 0);
  EXPECT_FALSE(bench.Irq());
  bench.WriteReg(kRegIrqEnable, kIrqHalt);
  EXPECT_TRUE(bench.Irq());
  bench.WriteReg(kRegIrqStatus, kIrqHalt);
  EXPECT_EQ(bench.ReadReg(kRegIrqStatus), 0U);
  EXPECT_FALSE(bench.Irq());
}

TEST(SequencerTest, SignalRecordsIdValueAndCount) {
  NpuBench bench(63);
  bench.WriteReg(kRegIrqEnable, kIrqSignal);
  const std::vector<Instruction> program = {Signal(5, 0x1234), Nop(),
                                            Signal(9, 0xDEADBEEFU), Halt()};
  ASSERT_TRUE(bench.Run(program));
  EXPECT_EQ(bench.ReadReg(kRegSignal), (2U << 16) | 9U);
  EXPECT_EQ(bench.ReadReg(kRegSignalValue), 0xDEADBEEFU);
  EXPECT_EQ(bench.ReadReg(kRegIrqStatus), kIrqSignal | kIrqHalt);
  EXPECT_TRUE(bench.Irq());
  bench.WriteReg(kRegIrqStatus, kIrqSignal);
  EXPECT_FALSE(bench.Irq());
}

TEST(SequencerTest, WaitHostHoldsUntilHostGoReachesValue) {
  NpuBench bench(64);
  const std::vector<Instruction> program = {Signal(1, 0), WaitHost(3),
                                            Signal(2, 0), Halt()};
  bench.LoadProgram(program, NpuBench::kProgramBase);
  bench.Start(NpuBench::kProgramBase, static_cast<uint32_t>(program.size()));
  ASSERT_TRUE(bench.WaitStatus(
      [](uint32_t status) { return (status & kStatusWaiting) != 0; }, 100000));
  EXPECT_EQ(bench.ReadReg(kRegSignal) & 0xFFU, 1U);
  bench.WriteReg(kRegHostGo, 2);
  for (int i = 0; i < 2000; ++i) {
    bench.Tick();
  }
  EXPECT_EQ(bench.ReadReg(kRegStatus) & (kStatusRunning | kStatusWaiting),
            kStatusRunning | kStatusWaiting);
  EXPECT_EQ(bench.ReadReg(kRegSignal) & 0xFFU, 1U);
  bench.WriteReg(kRegHostGo, 3);
  ASSERT_TRUE(bench.WaitStatus(
      [](uint32_t status) { return (status & kStatusHalted) != 0; }, 100000));
  EXPECT_EQ(bench.ReadReg(kRegSignal) & 0xFFU, 2U);

  bench.Start(NpuBench::kProgramBase, static_cast<uint32_t>(program.size()));
  ASSERT_TRUE(bench.WaitStatus(
      [](uint32_t status) { return (status & kStatusWaiting) != 0; }, 100000));
  EXPECT_EQ(bench.ReadReg(kRegHostGo), 0U);
  bench.WriteReg(kRegHostGo, 7);
  ASSERT_TRUE(bench.WaitStatus(
      [](uint32_t status) { return (status & kStatusHalted) != 0; }, 100000));
  EXPECT_TRUE(bench.Errors().empty());
}

class IllegalOpcodeTest : public testing::TestWithParam<uint32_t> {};

TEST_P(IllegalOpcodeTest, StopsWithError) {
  NpuBench bench(65);
  bench.WriteReg(kRegIrqEnable, kIrqIllegal);
  const std::vector<Instruction> program = {Signal(4, 1), Illegal(GetParam()),
                                            Signal(8, 2), Halt()};
  ASSERT_TRUE(bench.Run(program));
  EXPECT_NE(bench.ReadReg(kRegStatus) & kStatusError, 0U);
  EXPECT_EQ(bench.ReadReg(kRegStatus) & kStatusRunning, 0U);
  EXPECT_NE(bench.ReadReg(kRegIrqStatus) & kIrqIllegal, 0U);
  EXPECT_TRUE(bench.Irq());
  for (int i = 0; i < 1000; ++i) {
    bench.Tick();
  }
  EXPECT_EQ(bench.ReadReg(kRegSignal), (1U << 16) | 4U);
}

INSTANTIATE_TEST_SUITE_P(Opcodes, IllegalOpcodeTest,
                         testing::Values(3U, 9U, 0x40U, 0xFFU));

TEST(SequencerTest, StreamLongerThanPrefetchBuffer) {
  NpuBench bench(66);
  std::vector<Instruction> program(300, Nop());
  program.push_back(Signal(3, 300));
  program.push_back(Halt());
  ASSERT_TRUE(bench.Run(program));
  EXPECT_EQ(bench.ReadReg(kRegStatus), kStatusHalted | kStatusIbufEmpty);
  EXPECT_EQ(bench.ReadReg(kRegIssued), 302U);
  EXPECT_EQ(bench.ReadReg(kRegSignalValue), 300U);
}

TEST(SequencerTest, SoftResetStopsTheStream) {
  NpuBench bench(67);
  const std::vector<Instruction> first = {WaitHost(1), Signal(1, 0), Halt()};
  bench.LoadProgram(first, NpuBench::kProgramBase);
  bench.Start(NpuBench::kProgramBase, static_cast<uint32_t>(first.size()));
  ASSERT_TRUE(bench.WaitStatus(
      [](uint32_t status) { return (status & kStatusWaiting) != 0; }, 100000));
  bench.WriteReg(kRegCtrl, 2);
  for (int i = 0; i < 100; ++i) {
    bench.Tick();
  }
  EXPECT_EQ(bench.ReadReg(kRegStatus) & 0xFU, 0U);
  const std::vector<Instruction> second = {Signal(7, 70), Halt()};
  ASSERT_TRUE(bench.Run(second));
  EXPECT_EQ(bench.ReadReg(kRegSignal), (1U << 16) | 7U);
  EXPECT_EQ(bench.ReadReg(kRegStatus), kStatusHalted | kStatusIbufEmpty);
}

TEST(SequencerTest, BusyCountersFollowEngines) {
  NpuBench bench(68);
  const std::vector<Instruction> program = {
      DmaLoad({.ddr_addr = Ddr::kBase + 0x100000U,
               .mem = Mem::kFbuf,
               .onchip = 0,
               .rows = 4,
               .row_words = 8,
               .ddr_stride = 0x400}),
      Gemm({.a_base = 0,
            .a_stride = 1,
            .kb = 1,
            .m = 64,
            .w_base = 0,
            .w_stride = 1,
            .n8 = 2,
            .c_base = 100,
            .wmode = 0,
            .acc = false,
            .wait = kWaitDma}),
      Halt()};
  ASSERT_TRUE(bench.Run(program));
  const uint32_t cycles = bench.ReadReg(kRegCycles);
  const uint32_t gemm = bench.ReadReg(kRegGemmCycles);
  const uint32_t dma = bench.ReadReg(kRegDmaCycles);
  EXPECT_GT(gemm, 0U);
  EXPECT_GT(dma, 0U);
  EXPECT_EQ(bench.ReadReg(kRegVecCycles), 0U);
  EXPECT_LE(gemm + dma, cycles);
}

struct TraceRecord {
  uint32_t index = 0;
  uint32_t issue = 0;
  uint32_t completion = 0;
  uint32_t source = 0;
  uint32_t opcode = 0;
  uint32_t wait = 0;
  uint32_t flags = 0;
  uint32_t zero = 0;
};

TraceRecord ReadRecord(const Ddr& ddr, uint32_t address) {
  const uint32_t word3 = ddr.Read32(address + 12);
  return {.index = ddr.Read32(address),
          .issue = ddr.Read32(address + 4),
          .completion = ddr.Read32(address + 8),
          .source = word3 & 0xFU,
          .opcode = (word3 >> 4) & 0xFFU,
          .wait = (word3 >> 12) & 0xFU,
          .flags = (word3 >> 16) & 0xFU,
          .zero = word3 >> 20};
}

TEST(SequencerTest, TraceRecordsEveryInstruction) {
  NpuBench bench(69);
  bench.WriteReg(kRegTraceBase, kTraceArea);
  bench.WriteReg(kRegTraceCtrl, 1);
  const std::vector<Instruction> program = {
      DmaLoad({.ddr_addr = Ddr::kBase + 0x100000U,
               .mem = Mem::kFbuf,
               .onchip = 0,
               .rows = 1,
               .row_words = 4}),
      Gemm({.a_base = 0,
            .a_stride = 1,
            .kb = 1,
            .m = 32,
            .w_base = 0,
            .w_stride = 1,
            .n8 = 1,
            .c_base = 8,
            .wmode = 0,
            .acc = false,
            .wait = kWaitDma}),
      Vec({.op = 0,
           .src = 0,
           .src2 = 0,
           .dst = 16,
           .rows_q = 1,
           .n8 = 4,
           .p_base = 0,
           .aux = 0,
           .imm = 0,
           .q = false,
           .wait = kWaitGemm}),
      Signal(11, 22, kWaitVec),
      Nop(),
      Halt()};
  ASSERT_TRUE(bench.Run(program));
  ASSERT_TRUE(bench.WaitStatus(
      [&](uint32_t) { return (bench.ReadReg(kRegTraceCtrl) & 2U) != 0; },
      100000));
  ASSERT_EQ(bench.ReadReg(kRegTraceCount), 6U);
  const std::vector<uint32_t> sources = {0, 1, 2, 3, 3, 3};
  const std::vector<uint32_t> opcodes = {kOpDmaLoad, kOpGemm, kOpVec,
                                         kOpSignal,  kOpNop,  kOpHalt};
  const std::vector<uint32_t> waits = {0,        kWaitDma, kWaitGemm,
                                       kWaitVec, 0,        kWaitAll};
  uint32_t last_issue = 0;
  for (uint32_t i = 0; i < 6; ++i) {
    const TraceRecord record =
        ReadRecord(bench.Memory(), kTraceArea + (16 * i));
    EXPECT_EQ(record.index, i);
    EXPECT_EQ(record.source, sources[i]) << "record " << i;
    EXPECT_EQ(record.opcode, opcodes[i]) << "record " << i;
    EXPECT_EQ(record.wait, waits[i]) << "record " << i;
    EXPECT_EQ(record.flags, 0U) << "record " << i;
    EXPECT_EQ(record.zero, 0U) << "record " << i;
    EXPECT_LE(record.issue, record.completion) << "record " << i;
    EXPECT_GE(record.issue, last_issue) << "record " << i;
    last_issue = record.issue;
  }
  EXPECT_TRUE(bench.Errors().empty());
}

struct StatisticsRecord {
  uint32_t index = 0;
  uint32_t source = 0;
  uint32_t type = 0;
  uint32_t a = 0;
  uint32_t b = 0;
  uint32_t c = 0;
  uint32_t d = 0;
  uint32_t zero = 0;
};

StatisticsRecord ReadStatistics(const Ddr& ddr, uint32_t address) {
  const uint32_t word1 = ddr.Read32(address + 4);
  const uint32_t word2 = ddr.Read32(address + 8);
  const uint32_t word3 = ddr.Read32(address + 12);
  return {.index = ddr.Read32(address),
          .source = word1 & 0xFU,
          .type = (word3 >> 20) & 0xFU,
          .a = word1 >> 8,
          .b = word2 & 0xFFFFFFU,
          .c = (word2 >> 24) | ((word3 & 0xFFFFU) << 8),
          .d = word3 >> 24,
          .zero = ((word1 >> 4) & 0xFU) | (((word3 >> 16) & 0xFU) << 4)};
}

constexpr uint32_t kTraceStatistics = 0x5;

TEST(SequencerTest, StatisticsRecordsFollowEngineCompletions) {
  NpuBench bench(70);
  constexpr uint32_t kList = Ddr::kBase + 0x180000U;
  constexpr uint32_t kData = Ddr::kBase + 0x100000U;
  bench.Memory().Write32(kList, kData >> 4);
  bench.Memory().Write32(kList + 4, 0xFFFFFFFFU);
  bench.Memory().Write32(kList + 8, (kData + 0x80) >> 4);
  bench.Memory().Write32(kList + 12, 0xFFFFFFFFU);
  bench.WriteReg(kRegTraceBase, kTraceArea);
  bench.WriteReg(kRegTraceCtrl, kTraceStatistics);
  EXPECT_EQ(bench.ReadReg(kRegTraceCtrl) & 0x5U, 0x5U);
  const std::vector<Instruction> program = {
      DmaLoad({.ddr_addr = kData,
               .mem = Mem::kFbuf,
               .onchip = 0,
               .rows = 1,
               .row_words = 4}),
      Gemm({.a_base = 0,
            .a_stride = 1,
            .kb = 1,
            .m = 32,
            .w_base = 0,
            .w_stride = 1,
            .n8 = 1,
            .c_base = 8,
            .wmode = 0,
            .acc = false,
            .wait = kWaitDma}),
      Vec({.op = 0,
           .src = 0,
           .src2 = 0,
           .dst = 16,
           .rows_q = 1,
           .n8 = 4,
           .p_base = 0,
           .aux = 0,
           .imm = 0,
           .q = false,
           .wait = kWaitGemm}),
      DmaStore({.ddr_addr = kData + 0x1000,
                .mem = Mem::kFbuf,
                .onchip = 16,
                .rows = 1,
                .row_words = 2,
                .wait = kWaitVec}),
      DmaLoad({.ddr_addr = kList,
               .mem = Mem::kFbuf,
               .onchip = 32,
               .rows = 1,
               .row_words = 4,
               .tok = 1,
               .flags = kDmaFlagIndex | kDmaFlagRow}),
      Signal(11, 22, kWaitDma),
      Halt()};
  ASSERT_TRUE(bench.Run(program));
  ASSERT_TRUE(bench.WaitStatus(
      [&](uint32_t) { return (bench.ReadReg(kRegTraceCtrl) & 2U) != 0; },
      100000));
  ASSERT_EQ(bench.ReadReg(kRegTraceCount), 12U);
  const std::vector<uint32_t> indices = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6};
  const std::vector<uint32_t> types = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0};
  for (uint32_t i = 0; i < 12; ++i) {
    const uint32_t address = kTraceArea + (16 * i);
    if (types[i] == 0) {
      const TraceRecord record = ReadRecord(bench.Memory(), address);
      EXPECT_EQ(record.index, indices[i]) << "record " << i;
      EXPECT_EQ(record.zero, 0U) << "record " << i;
    } else {
      const StatisticsRecord record = ReadStatistics(bench.Memory(), address);
      EXPECT_EQ(record.index, indices[i]) << "record " << i;
      EXPECT_EQ(record.type, 1U) << "record " << i;
      EXPECT_EQ(record.zero, 0U) << "record " << i;
    }
  }
  const StatisticsRecord load = ReadStatistics(bench.Memory(), kTraceArea + 16);
  EXPECT_EQ(load.source, 0U);
  EXPECT_EQ(load.a, 32U);
  EXPECT_EQ(load.b, 0U);
  EXPECT_EQ(load.d, 0U);
  const StatisticsRecord gemm = ReadStatistics(bench.Memory(), kTraceArea + 48);
  EXPECT_EQ(gemm.source, 1U);
  EXPECT_EQ(gemm.a, 8U);
  EXPECT_EQ(gemm.b, 0U);
  EXPECT_EQ(gemm.c, 0U);
  EXPECT_EQ(gemm.d, 0U);
  const StatisticsRecord vec = ReadStatistics(bench.Memory(), kTraceArea + 80);
  EXPECT_EQ(vec.source, 2U);
  EXPECT_EQ(vec.d, 1U);
  EXPECT_EQ(vec.a, 14U);
  EXPECT_GT(vec.b, 0U);
  EXPECT_EQ(vec.c, 0U);
  const StatisticsRecord store =
      ReadStatistics(bench.Memory(), kTraceArea + 112);
  EXPECT_EQ(store.source, 0U);
  EXPECT_EQ(store.a, 0U);
  EXPECT_EQ(store.b, 16U);
  const StatisticsRecord indexed =
      ReadStatistics(bench.Memory(), kTraceArea + 144);
  EXPECT_EQ(indexed.a, 1U + 16U);
  EXPECT_EQ(indexed.d, 2U);
  EXPECT_TRUE(bench.Errors().empty());
}

TEST(SequencerTest, StatisticsRecordsSurviveTraceBackPressure) {
  NpuBench bench(71);
  bench.WriteReg(kRegTraceBase, kTraceArea);
  bench.WriteReg(kRegTraceCtrl, kTraceStatistics);
  std::vector<Instruction> program;
  for (uint32_t i = 0; i < 40; ++i) {
    program.push_back(Vec({.op = 5,
                           .src = 0,
                           .src2 = 0,
                           .dst = 64,
                           .rows_q = 1,
                           .n8 = 1,
                           .p_base = 0,
                           .aux = 0,
                           .imm = 0,
                           .q = false,
                           .wait = 0}));
    program.push_back(Gemm({.a_base = 0,
                            .a_stride = 1,
                            .kb = 1,
                            .m = 32,
                            .w_base = 0,
                            .w_stride = 1,
                            .n8 = 1,
                            .c_base = 128,
                            .wmode = 0,
                            .acc = false,
                            .wait = 0}));
    program.push_back(Nop());
  }
  program.push_back(Halt());
  ASSERT_TRUE(bench.Run(program));
  ASSERT_TRUE(bench.WaitStatus(
      [&](uint32_t) { return (bench.ReadReg(kRegTraceCtrl) & 2U) != 0; },
      400000));
  ASSERT_EQ(bench.ReadReg(kRegTraceCount), 121U + 80U);
  std::vector<bool> completed(program.size(), false);
  std::vector<bool> described(program.size(), false);
  uint32_t last_completion_index = 0;
  for (uint32_t i = 0; i < 201; ++i) {
    const uint32_t address = kTraceArea + (16 * i);
    const uint32_t type = (bench.Memory().Read32(address + 12) >> 20) & 0xFU;
    if (type == 0) {
      const TraceRecord record = ReadRecord(bench.Memory(), address);
      ASSERT_LT(record.index, program.size());
      EXPECT_FALSE(completed[record.index]);
      completed[record.index] = true;
      last_completion_index = record.index;
    } else {
      const StatisticsRecord record = ReadStatistics(bench.Memory(), address);
      EXPECT_EQ(type, 1U);
      EXPECT_EQ(record.index, last_completion_index) << "record " << i;
      EXPECT_FALSE(described[record.index]);
      described[record.index] = true;
    }
  }
  for (size_t i = 0; i < program.size(); ++i) {
    EXPECT_TRUE(completed[i]) << i;
    EXPECT_EQ(described[i], i % 3 != 2 && i + 1 != program.size()) << i;
  }
  EXPECT_TRUE(bench.Errors().empty());
}

}  // namespace
}  // namespace hotarubi::tb
