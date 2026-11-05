#include "machine.hpp"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "encoding.hpp"
#include "hotarubi/driver.hpp"
#include "iss_bus.hpp"
#include "memories.hpp"
#include "reference.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint64_t kBase = 0x30000000;
constexpr size_t kDdrBytes = size_t{1} << 16;
constexpr uint32_t kStream = 0x1000;
constexpr uint32_t kInput = 0x4000;
constexpr uint32_t kOutput = 0x8000;

constexpr uint32_t kCtrl = 0x00;
constexpr uint32_t kStatus = 0x04;
constexpr uint32_t kInstrBase = 0x08;
constexpr uint32_t kInstrCount = 0x0C;
constexpr uint32_t kIrqStatus = 0x10;
constexpr uint32_t kIrqEnable = 0x14;
constexpr uint32_t kSignal = 0x18;
constexpr uint32_t kSignalValue = 0x1C;
constexpr uint32_t kIssued = 0x20;
constexpr uint32_t kHostGo = 0x48;

uint32_t Address(uint32_t offset) {
  return static_cast<uint32_t>(kBase + offset);
}

Word128 Header(uint32_t opcode, uint32_t wait = 0, uint32_t flags = 0) {
  Word128 word;
  encoding::Set(word, 120, 8, opcode);
  encoding::Set(word, 116, 4, wait);
  return encoding::Set(word, 112, 4, flags);
}

Word128 Dma(uint32_t opcode, uint32_t ddr, uint32_t mem, uint32_t onchip,
            uint32_t rows, uint32_t row_words, uint32_t stride) {
  Word128 word = Header(opcode, 0x8);
  encoding::Set(word, 80, 32, Address(ddr));
  encoding::Set(word, 77, 3, mem);
  encoding::Set(word, 57, 20, onchip);
  encoding::Set(word, 45, 12, rows);
  encoding::Set(word, 33, 12, row_words);
  return encoding::Set(word, 9, 24, stride);
}

Word128 Vec(uint32_t op, uint32_t src, uint32_t src2, uint32_t dst,
            uint32_t rows_q, uint32_t n8) {
  Word128 word = Header(5, 0x1);
  encoding::Set(word, 107, 5, op);
  encoding::Set(word, 95, 12, src);
  encoding::Set(word, 83, 12, src2);
  encoding::Set(word, 70, 13, dst);
  encoding::Set(word, 61, 9, rows_q);
  return encoding::Set(word, 53, 8, n8);
}

Word128 SignalOp(uint32_t id, uint32_t value) {
  Word128 word = Header(6, 0xD);
  encoding::Set(word, 104, 8, id);
  return encoding::Set(word, 0, 32, value);
}

Word128 WaitHost(uint32_t value) {
  Word128 word = Header(8);
  return encoding::Set(word, 0, 32, value);
}

class MachineTest : public testing::Test {
 protected:
  void Load(const std::vector<Word128>& stream) {
    for (size_t i = 0; i < stream.size(); ++i) {
      Put(kStream + (16 * static_cast<uint32_t>(i)), stream.at(i).low, 8);
      Put(kStream + (16 * static_cast<uint32_t>(i)) + 8, stream.at(i).high, 8);
    }
    machine_.WriteRegister(kInstrBase, Address(kStream));
    machine_.WriteRegister(kInstrCount, static_cast<uint32_t>(stream.size()));
  }

  void Put(uint32_t offset, uint64_t value, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
      bytes_.at(offset + i) = static_cast<std::byte>((value >> (8 * i)) & 0xFF);
    }
  }

  [[nodiscard]] uint32_t Get32(uint32_t offset) const {
    uint32_t value = 0;
    for (uint32_t i = 4; i > 0; --i) {
      value =
          (value << 8) | std::to_integer<uint32_t>(bytes_.at(offset + i - 1));
    }
    return value;
  }

  void FillInput(uint32_t words) {
    std::mt19937 generator(5);
    std::uniform_int_distribution<uint32_t> fraction(0, 0x7FFFFF);
    for (uint32_t i = 0; i < words * 32; ++i) {
      Put(kInput + (4 * i), (uint32_t{125} << 23) | fraction(generator), 4);
    }
  }

  [[nodiscard]] static std::vector<Word128> AddStream() {
    return {Dma(1, kInput, 2, 0, 1, 4, 512),
            Dma(1, kInput + 512, 2, 4, 1, 4, 512), Vec(0, 0, 4, 8, 1, 4),
            Dma(2, kOutput, 2, 8, 1, 4, 512), Header(7, 0xD)};
  }

  void ExpectSum() const {
    for (uint32_t i = 0; i < 4 * 32; ++i) {
      ASSERT_EQ(Get32(kOutput + (4 * i)),
                reference::Add(Get32(kInput + (4 * i)),
                               Get32(kInput + 512 + (4 * i))));
    }
  }

  std::vector<std::byte> bytes_ = std::vector<std::byte>(kDdrBytes);
  Machine machine_{bytes_, kBase};
};

TEST_F(MachineTest, RunsAStreamToHalt) {
  FillInput(8);
  Load(AddStream());
  machine_.WriteRegister(kIrqEnable, 0x2);
  EXPECT_EQ(machine_.Run(), Stop::kIdle);
  machine_.WriteRegister(kCtrl, 1);
  EXPECT_EQ(machine_.ReadRegister(kStatus) & 0x7U, 0x1U);
  EXPECT_EQ(machine_.Run(), Stop::kHalt);
  ExpectSum();
  EXPECT_EQ(machine_.ReadRegister(kStatus) & 0x7U, 0x2U);
  EXPECT_EQ(machine_.ReadRegister(kIssued), 5U);
  EXPECT_EQ(machine_.ReadRegister(kIrqStatus), 0x2U);
  EXPECT_TRUE(machine_.Interrupt());
  machine_.WriteRegister(kIrqStatus, 0x2);
  EXPECT_EQ(machine_.ReadRegister(kIrqStatus), 0U);
  EXPECT_FALSE(machine_.Interrupt());
  EXPECT_EQ(machine_.Run(), Stop::kIdle);
}

TEST_F(MachineTest, SignalsAndWaitsForTheHost) {
  Load({SignalOp(7, 3), WaitHost(3), Header(0), SignalOp(9, 0x12345678),
        Header(7)});
  machine_.WriteRegister(kIrqEnable, 0x1);
  machine_.WriteRegister(kHostGo, 5);
  machine_.WriteRegister(kCtrl, 1);
  EXPECT_EQ(machine_.ReadRegister(kHostGo), 0U);
  EXPECT_EQ(machine_.Run(), Stop::kSignal);
  EXPECT_EQ(machine_.ReadRegister(kSignal), (1U << 16) | 7U);
  EXPECT_EQ(machine_.ReadRegister(kSignalValue), 3U);
  EXPECT_TRUE(machine_.Interrupt());
  EXPECT_EQ(machine_.Run(), Stop::kWaitHost);
  EXPECT_EQ(machine_.ReadRegister(kStatus) & 0x9U, 0x9U);
  EXPECT_EQ(machine_.ReadRegister(kIssued), 1U);
  machine_.WriteRegister(kHostGo, 2);
  EXPECT_EQ(machine_.Run(), Stop::kWaitHost);
  machine_.WriteRegister(kHostGo, 3);
  EXPECT_EQ(machine_.ReadRegister(kStatus) & 0x8U, 0U);
  EXPECT_EQ(machine_.Run(), Stop::kSignal);
  EXPECT_EQ(machine_.ReadRegister(kSignal), (2U << 16) | 9U);
  EXPECT_EQ(machine_.ReadRegister(kSignalValue), 0x12345678U);
  EXPECT_EQ(machine_.Run(), Stop::kHalt);
  EXPECT_EQ(machine_.ReadRegister(kIrqStatus), 0x3U);
  EXPECT_EQ(machine_.ReadRegister(kIssued), 5U);
}

TEST_F(MachineTest, StopsAtIllegalOpcodes) {
  for (const uint32_t opcode : {3U, 9U, 255U}) {
    Load({Header(0), Header(opcode), Header(7)});
    machine_.WriteRegister(kIrqStatus, 0xF);
    machine_.WriteRegister(kCtrl, 1);
    EXPECT_EQ(machine_.Run(), Stop::kIllegal);
    EXPECT_EQ(machine_.ReadRegister(kStatus) & 0x7U, 0x6U);
    EXPECT_EQ(machine_.ReadRegister(kIrqStatus), 0x4U);
    EXPECT_EQ(machine_.ReadRegister(kIssued), 2U);
    EXPECT_EQ(machine_.Run(), Stop::kIdle);
  }
}

TEST_F(MachineTest, StartRestartsAndSoftResetStops) {
  Load({SignalOp(1, 1), Header(0), Header(7)});
  machine_.WriteRegister(kCtrl, 1);
  EXPECT_EQ(machine_.Run(), Stop::kSignal);
  machine_.WriteRegister(kCtrl, 1);
  EXPECT_EQ(machine_.ReadRegister(kIssued), 0U);
  EXPECT_EQ(machine_.Run(), Stop::kSignal);
  EXPECT_EQ(machine_.ReadRegister(kSignal), (2U << 16) | 1U);
  machine_.WriteRegister(kCtrl, 2);
  EXPECT_EQ(machine_.ReadRegister(kStatus) & 0x7U, 0U);
  EXPECT_EQ(machine_.ReadRegister(kInstrCount), 3U);
  EXPECT_EQ(machine_.Run(), Stop::kIdle);
}

TEST_F(MachineTest, RunsOutOfInstructionsWithoutHalting) {
  Load({Header(0), Header(0)});
  machine_.WriteRegister(kCtrl, 1);
  EXPECT_EQ(machine_.Run(), Stop::kStarved);
  EXPECT_EQ(machine_.ReadRegister(kStatus) & 0x81U, 0x81U);
  EXPECT_EQ(machine_.ReadRegister(kIssued), 2U);
}

TEST_F(MachineTest, DriverRunsStreamsThroughTheBus) {
  FillInput(8);
  std::vector<Word128> stream = AddStream();
  stream.insert(stream.end() - 1, SignalOp(4, 2));
  stream.insert(stream.end() - 1, WaitHost(2));
  stream.insert(stream.end() - 1, Dma(1, kOutput, 2, 12, 1, 4, 512));
  stream.insert(stream.end() - 1, Dma(2, kOutput + 1024, 2, 12, 1, 4, 512));
  Load(stream);
  IssBus bus(machine_);
  Device device(bus);
  std::vector<Signal> signals;
  const auto counters =
      device.Run(Address(kStream), static_cast<uint32_t>(stream.size()),
                 std::chrono::milliseconds(1000), [&](const Signal& signal) {
                   signals.push_back(signal);
                   ExpectSum();
                   for (uint32_t i = 0; i < 4 * 32; ++i) {
                     Put(kOutput + (4 * i), i, 4);
                   }
                 });
  ASSERT_TRUE(counters.has_value()) << counters.error();
  EXPECT_EQ(counters->issued, static_cast<uint32_t>(stream.size()));
  ASSERT_EQ(signals.size(), 1U);
  EXPECT_EQ(signals.at(0).id, 4U);
  EXPECT_EQ(signals.at(0).value, 2U);
  for (uint32_t i = 0; i < 4 * 32; ++i) {
    EXPECT_EQ(Get32(kOutput + 1024 + (4 * i)), i);
  }
}

TEST_F(MachineTest, DriverReportsIllegalInstructionsAndHangs) {
  Load({Header(0), Header(3)});
  IssBus bus(machine_);
  Device device(bus);
  const auto illegal =
      device.Run(Address(kStream), 2, std::chrono::milliseconds(1000));
  ASSERT_FALSE(illegal.has_value());
  EXPECT_EQ(illegal.error(), "illegal instruction after 2 instructions");
  Load({WaitHost(1), Header(7)});
  const auto hang =
      device.Run(Address(kStream), 2, std::chrono::milliseconds(1000));
  ASSERT_FALSE(hang.has_value());
  EXPECT_NE(hang.error().find("did not halt"), std::string::npos);
}

}  // namespace
}  // namespace hotarubi::iss
