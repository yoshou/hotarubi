#include "hotarubi/driver.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "bench_bus.hpp"
#include "npu_bench.hpp"
#include "npu_isa.hpp"

namespace hotarubi {
namespace {

using std::chrono_literals::operator""ms;

class TempTree {
 public:
  TempTree()
      : root_(std::filesystem::path(testing::TempDir()) /
              (std::string(testing::UnitTest::GetInstance()
                               ->current_test_info()
                               ->name()) +
               "_" + std::to_string(getpid()))) {
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
  }
  TempTree(const TempTree&) = delete;
  TempTree& operator=(const TempTree&) = delete;
  TempTree(TempTree&&) = delete;
  TempTree& operator=(TempTree&&) = delete;
  ~TempTree() { std::filesystem::remove_all(root_); }

  [[nodiscard]] std::filesystem::path Path(std::string_view relative) const {
    return root_ / relative;
  }

  void WriteText(std::string_view relative, std::string_view text) const {
    const std::filesystem::path path = Path(relative);
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << text;
  }

  void WriteZeros(std::string_view relative, size_t size) const {
    std::ofstream(Path(relative), std::ios::binary) << std::string(size, '\0');
  }

  [[nodiscard]] std::vector<uint8_t> ReadBytes(
      std::string_view relative) const {
    std::ifstream file(Path(relative), std::ios::binary);
    return {std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()};
  }

 private:
  std::filesystem::path root_;
};

TEST(SharedBufferTest, MapsTheBufferAndReadsItsAddress) {
  const TempTree tree;
  tree.WriteText("class/phys_addr", "0x0000000078000000\n");
  tree.WriteText("class/size", "8192\n");
  tree.WriteZeros("udmabuf0", 8192);
  {
    auto buffer = SharedBuffer::Open(tree.Path("class"), tree.Path("udmabuf0"));
    ASSERT_TRUE(buffer.has_value()) << buffer.error();
    EXPECT_EQ(buffer->PhysicalAddress(), 0x78000000U);
    ASSERT_EQ(buffer->Bytes().size(), 8192U);
    buffer->Bytes()[100] = std::byte{0x5A};
  }
  EXPECT_EQ(tree.ReadBytes("udmabuf0").at(100), 0x5A);
}

TEST(SharedBufferTest, RejectsABufferOutsideThe32BitAddressSpace) {
  const TempTree tree;
  tree.WriteText("class/phys_addr", "0x00000000fffff000\n");
  tree.WriteText("class/size", "8192\n");
  tree.WriteZeros("udmabuf0", 8192);
  const auto buffer =
      SharedBuffer::Open(tree.Path("class"), tree.Path("udmabuf0"));
  ASSERT_FALSE(buffer.has_value());
  EXPECT_NE(buffer.error().find("32-bit"), std::string::npos);
}

TEST(SharedBufferTest, ReportsAMissingDevice) {
  const TempTree tree;
  tree.WriteText("class/phys_addr", "0x0000000078000000\n");
  tree.WriteText("class/size", "8192\n");
  const auto buffer =
      SharedBuffer::Open(tree.Path("class"), tree.Path("udmabuf0"));
  ASSERT_FALSE(buffer.has_value());
  EXPECT_NE(buffer.error().find("udmabuf0"), std::string::npos);
}

TEST(FindUioTest, FindsTheDeviceByName) {
  const TempTree tree;
  tree.WriteText("uio/uio0/name", "axi_dma\n");
  tree.WriteText("uio/uio3/name", "npu\n");
  const auto device = FindUio(tree.Path("uio"), "npu");
  ASSERT_TRUE(device.has_value()) << device.error();
  EXPECT_EQ(*device, std::filesystem::path("/dev/uio3"));
  EXPECT_FALSE(FindUio(tree.Path("uio"), "vcu").has_value());
}

TEST(UioBusTest, AccessesRegistersAs32BitWords) {
  const TempTree tree;
  tree.WriteZeros("uio0", UioBus::kMapSize);
  {
    auto bus = UioBus::Open(tree.Path("uio0"));
    ASSERT_TRUE(bus.has_value()) << bus.error();
    (*bus)->Write(Register::kInstrBase, 0x12345678U);
    EXPECT_EQ((*bus)->Read(Register::kInstrBase), 0x12345678U);
  }
  const std::vector<uint8_t> bytes = tree.ReadBytes("uio0");
  EXPECT_EQ(bytes.at(8), 0x78);
  EXPECT_EQ(bytes.at(9), 0x56);
  EXPECT_EQ(bytes.at(10), 0x34);
  EXPECT_EQ(bytes.at(11), 0x12);
}

constexpr uint32_t kInput = tb::Ddr::kBase + 0x1000U;
constexpr uint32_t kOutput = tb::Ddr::kBase + 0x2000U;
constexpr uint32_t kScale = 4;
constexpr uint32_t kTwo = 0x40000000U;

std::vector<tb::Instruction> DoubleWord() {
  return {tb::DmaLoad({.ddr_addr = kInput,
                       .mem = tb::Mem::kFbuf,
                       .onchip = 0,
                       .rows = 1,
                       .row_words = 1,
                       .ddr_stride = 128}),
          tb::Vec({.op = kScale,
                   .src = 0,
                   .dst = 1,
                   .rows_q = 1,
                   .n8 = 1,
                   .imm = kTwo,
                   .wait = tb::kWaitDma}),
          tb::DmaStore({.ddr_addr = kOutput,
                        .mem = tb::Mem::kFbuf,
                        .onchip = 1,
                        .rows = 1,
                        .row_words = 1,
                        .ddr_stride = 128,
                        .wait = tb::kWaitVec}),
          tb::Halt()};
}

TEST(DeviceTest, RunsAStreamToHalt) {
  tb::NpuBench bench;
  BenchBus bus(bench);
  Device device(bus);
  const std::vector<tb::Instruction> program = {tb::Nop(), tb::Nop(),
                                                tb::Halt()};
  bench.LoadProgram(program, tb::Ddr::kBase);
  const auto counters =
      device.Run(tb::Ddr::kBase, static_cast<uint32_t>(program.size()), 1000ms);
  ASSERT_TRUE(counters.has_value()) << counters.error();
  EXPECT_EQ(counters->issued, 3U);
  EXPECT_GT(counters->cycles, 0U);
}

TEST(DeviceTest, ResumesAfterTheHostHandlesASignal) {
  tb::NpuBench bench;
  BenchBus bus(bench);
  Device device(bus);
  std::vector<tb::Instruction> program = {tb::Signal(7, 1), tb::WaitHost(1)};
  for (const tb::Instruction& instruction : DoubleWord()) {
    program.push_back(instruction);
  }
  bench.LoadProgram(program, tb::Ddr::kBase);
  std::vector<Signal> signals;
  const auto counters =
      device.Run(tb::Ddr::kBase, static_cast<uint32_t>(program.size()), 1000ms,
                 [&](const Signal& signal) {
                   signals.push_back(signal);
                   for (uint32_t i = 0; i < 32; ++i) {
                     bench.Memory().Write32(
                         kInput + (4 * i),
                         std::bit_cast<uint32_t>(static_cast<float>(i) - 7.5F));
                   }
                 });
  ASSERT_TRUE(counters.has_value()) << counters.error();
  ASSERT_EQ(signals.size(), 1U);
  EXPECT_EQ(signals[0].id, 7U);
  EXPECT_EQ(signals[0].value, 1U);
  for (uint32_t i = 0; i < 32; ++i) {
    EXPECT_EQ(std::bit_cast<float>(bench.Memory().Read32(kOutput + (4 * i))),
              2.0F * (static_cast<float>(i) - 7.5F));
  }
}

TEST(DeviceTest, ReportsAnIllegalInstruction) {
  tb::NpuBench bench;
  BenchBus bus(bench);
  Device device(bus);
  const std::vector<tb::Instruction> program = {tb::Nop(), tb::Header(3, 0, 0),
                                                tb::Halt()};
  bench.LoadProgram(program, tb::Ddr::kBase);
  const auto counters =
      device.Run(tb::Ddr::kBase, static_cast<uint32_t>(program.size()), 1000ms);
  ASSERT_FALSE(counters.has_value());
  EXPECT_NE(counters.error().find("illegal"), std::string::npos);
}

TEST(DeviceTest, ReportsAStreamThatDoesNotHalt) {
  tb::NpuBench bench;
  BenchBus bus(bench);
  Device device(bus);
  const std::vector<tb::Instruction> program = {tb::WaitHost(1), tb::Halt()};
  bench.LoadProgram(program, tb::Ddr::kBase);
  const auto counters =
      device.Run(tb::Ddr::kBase, static_cast<uint32_t>(program.size()), 1ms);
  ASSERT_FALSE(counters.has_value());
  EXPECT_NE(counters.error().find("did not halt"), std::string::npos);
}

}  // namespace
}  // namespace hotarubi
