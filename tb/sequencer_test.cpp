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

}  // namespace
}  // namespace hotarubi::tb
