#include "dma_engine.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "ddr.hpp"
#include "hotarubi/exact.hpp"
#include "hotarubi/isa.hpp"
#include "memories.hpp"
#include "reference.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint64_t kBase = 0x20000000;
constexpr size_t kDdrBytes = size_t{1} << 20;
using isa::kNullEntry;
constexpr std::byte kUntouched{0xEE};

enum MemCode : uint8_t {
  kAbuf = 0,
  kWbuf = 1,
  kFbuf = 2,
  kPbuf = 3,
  kTbuf = 4,
  kCbuf = 5,
};

uint32_t Nibble(const WbufWord& word, uint32_t index) {
  return (word.at(index / 2) >> (4 * (index % 2))) & 0xFU;
}

class DmaEngineTest : public testing::Test {
 protected:
  static uint32_t Address(uint32_t offset) {
    return static_cast<uint32_t>(kBase + offset);
  }

  static uint32_t Entry(uint32_t offset) { return Address(offset) / 16; }

  [[nodiscard]] uint8_t Byte(uint32_t offset) const {
    return std::to_integer<uint8_t>(bytes_.at(offset));
  }

  [[nodiscard]] uint64_t Little(uint32_t offset, uint32_t size) const {
    uint64_t value = 0;
    for (uint32_t i = size; i > 0; --i) {
      value = (value << 8) | Byte(offset + i - 1);
    }
    return value;
  }

  void Put(uint32_t offset, uint64_t value, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
      bytes_.at(offset + i) = static_cast<std::byte>((value >> (8 * i)) & 0xFF);
    }
  }

  uint8_t RandomByte() {
    return static_cast<uint8_t>(
        std::uniform_int_distribution<uint32_t>(0, 255)(generator_));
  }

  void Randomize(uint32_t offset, uint32_t size) {
    for (uint32_t i = 0; i < size; ++i) {
      bytes_.at(offset + i) = std::byte{RandomByte()};
    }
  }

  void Execute(const DmaCommand& command) {
    engine_.Execute(command, ddr_, memories_);
  }

  static DmaCommand Load(uint32_t mem, uint32_t ddr, uint32_t onchip,
                         uint32_t rows, uint32_t row_words, uint32_t stride) {
    return {.store = false,
            .ddr_addr = Address(ddr) / 16,
            .mem = mem,
            .onchip = onchip,
            .rows = rows,
            .row_words = row_words,
            .ddr_stride = stride / 16,
            .kind = isa::dma::kKindRaw,
            .tok = 0,
            .acc = false,
            .flush = false};
  }

  void ExpectWord(uint32_t mem, uint32_t index, uint32_t offset) const {
    switch (mem) {
      case kAbuf:
        for (uint32_t i = 0; i < kAbufWordBytes; ++i) {
          ASSERT_EQ(memories_.abuf.at(index).at(i), Byte(offset + i));
        }
        break;
      case kWbuf:
        for (uint32_t i = 0; i < kWbufWordBytes; ++i) {
          ASSERT_EQ(memories_.wbuf.at(index).at(i), Byte(offset + i));
        }
        break;
      case kFbuf:
      case kPbuf: {
        const FbufWord& word =
            mem == kFbuf ? memories_.fbuf.at(index) : memories_.pbuf.at(index);
        for (uint32_t i = 0; i < kFbufElements; ++i) {
          ASSERT_EQ(word.at(i), Little(offset + (4 * i), 4));
        }
        break;
      }
      case kTbuf:
        ASSERT_EQ(memories_.tbuf.at(index), Little(offset, 8));
        break;
      default:
        ASSERT_EQ(memories_.cbuf.at(index).low, Little(offset, 8));
        ASSERT_EQ(memories_.cbuf.at(index).high, Little(offset + 8, 8));
        break;
    }
  }

  void RandomizeOnChip(uint32_t first, uint32_t count) {
    for (uint32_t w = first; w < first + count; ++w) {
      for (uint8_t& byte : memories_.abuf.at(w)) {
        byte = RandomByte();
      }
      for (uint32_t& element : memories_.fbuf.at(w)) {
        element = (static_cast<uint32_t>(RandomByte()) * 0x01010101U) ^
                  (w * 0x9E3779B9U);
      }
    }
  }

  std::vector<std::byte> bytes_ = std::vector<std::byte>(kDdrBytes);
  Ddr ddr_{bytes_, kBase};
  Memories memories_;
  DmaEngine engine_;
  std::mt19937 generator_{3};
};

TEST_F(DmaEngineTest, LoadsRawRowsIntoEveryMemory) {
  const std::array<std::array<uint32_t, 2>, 6> cases = {{
      {kAbuf, 144},
      {kWbuf, 144},
      {kFbuf, 128},
      {kPbuf, 128},
      {kTbuf, 16},
      {kCbuf, 16},
  }};
  for (const auto& [mem, bytes] : cases) {
    Randomize(0, 4096);
    const uint32_t stride = (2 * bytes) + 48;
    Execute(Load(mem, 1024, 10, 3, 2, stride));
    for (uint32_t r = 0; r < 3; ++r) {
      for (uint32_t w = 0; w < 2; ++w) {
        ExpectWord(mem, 10 + (r * 2) + w, 1024 + (r * stride) + (w * bytes));
      }
    }
  }
}

TEST_F(DmaEngineTest, StoresRawRowsFromAbufAndFbuf) {
  for (const uint32_t mem : {kAbuf, kFbuf}) {
    std::ranges::fill(bytes_, kUntouched);
    RandomizeOnChip(10, 6);
    const uint32_t bytes = mem == kAbuf ? 144 : 128;
    const uint32_t stride = (2 * bytes) + 32;
    DmaCommand command = Load(mem, 512, 10, 3, 2, stride);
    command.store = true;
    Execute(command);
    std::vector<std::byte> expected(kDdrBytes, kUntouched);
    for (uint32_t r = 0; r < 3; ++r) {
      for (uint32_t w = 0; w < 2; ++w) {
        const uint32_t offset = 512 + (r * stride) + (w * bytes);
        for (uint32_t i = 0; i < bytes; ++i) {
          uint32_t value = 0;
          if (mem == kAbuf) {
            value = i < kAbufWordBytes
                        ? memories_.abuf.at(10 + (r * 2) + w).at(i)
                        : 0U;
          } else {
            value = (memories_.fbuf.at(10 + (r * 2) + w).at(i / 4) >>
                     (8 * (i % 4))) &
                    0xFFU;
          }
          expected.at(offset + i) = static_cast<std::byte>(value);
        }
      }
    }
    ASSERT_TRUE(bytes_ == expected) << "mem " << mem;
  }
}

struct RowCase {
  uint32_t mem;
  uint32_t tok;
  bool indexed;
};

class RowFormatTest : public DmaEngineTest,
                      public testing::WithParamInterface<RowCase> {
 protected:
  static constexpr uint32_t kRows = 2;
  static constexpr uint32_t kRowWords = 4;
  static constexpr uint32_t kOnChip = 20;
  static constexpr uint32_t kList = 0xF000;
  static constexpr uint32_t kRowsStart = 0x100;

  void SetUp() override {
    const RowCase c = GetParam();
    mxint8_ = c.mem == kAbuf;
    blocks_ = c.tok;
    size_ =
        mxint8_ ? (32 * blocks_) + (16 * ((blocks_ + 15) / 16)) : 128 * blocks_;
    words_ = mxint8_ ? blocks_ : 4 * blocks_;
    stride_ = (kRowWords * size_) + 32;
    for (uint32_t u = 0; u < kRows * kRowWords; ++u) {
      tokens_.push_back(c.indexed ? 0x8000 + (u * (size_ + 16))
                                  : kRowsStart + ((u / kRowWords) * stride_) +
                                        ((u % kRowWords) * size_));
    }
    if (c.indexed) {
      tokens_.at(5) = kNullEntry;
    }
  }

  void WriteList() {
    if (!GetParam().indexed) {
      return;
    }
    for (uint32_t u = 0; u < tokens_.size(); ++u) {
      Put(kList + (4 * u),
          tokens_.at(u) == kNullEntry ? kNullEntry : Entry(tokens_.at(u)), 4);
    }
  }

  [[nodiscard]] DmaCommand Command() const {
    DmaCommand command =
        Load(GetParam().mem, GetParam().indexed ? kList : kRowsStart, kOnChip,
             kRows, kRowWords, stride_);
    command.kind =
        GetParam().indexed ? isa::dma::kKindRowIndexed : isa::dma::kKindRow;
    command.tok = GetParam().tok;
    return command;
  }

  void ExpectPacked(uint32_t u) const {
    const uint32_t token = tokens_.at(u);
    const uint32_t j = u % 4;
    for (uint32_t g = 0; g < words_; ++g) {
      const uint32_t index = kOnChip + ((u / 4) * words_) + g;
      for (uint32_t k = 0; k < (mxint8_ ? 32U : 8U); ++k) {
        if (mxint8_) {
          ASSERT_EQ(memories_.abuf.at(index).at((j * 32) + k),
                    token == kNullEntry ? 0 : Byte(token + (g * 32) + k));
        } else {
          ASSERT_EQ(
              memories_.fbuf.at(index).at((j * 8) + k),
              token == kNullEntry ? 0 : Little(token + (32 * g) + (4 * k), 4));
        }
      }
      if (mxint8_) {
        ASSERT_EQ(memories_.abuf.at(index).at(kAbufScaleOffset + j),
                  token == kNullEntry ? 127 : Byte(token + (32 * blocks_) + g));
      }
    }
  }

  [[nodiscard]] uint32_t Stored(uint32_t u, uint32_t i) const {
    const uint32_t base = kOnChip + ((u / 4) * words_);
    const uint32_t j = u % 4;
    if (!mxint8_) {
      const uint32_t element = i / 4;
      return (memories_.fbuf.at(base + (element / 8))
                  .at((j * 8) + (element % 8)) >>
              (8 * (i % 4))) &
             0xFFU;
    }
    if (i < 32 * blocks_) {
      return memories_.abuf.at(base + (i / 32)).at((j * 32) + (i % 32));
    }
    if (i < (32 * blocks_) + blocks_) {
      return memories_.abuf.at(base + (i - (32 * blocks_)))
          .at(kAbufScaleOffset + j);
    }
    return 0;
  }

  bool mxint8_ = false;
  uint32_t blocks_ = 0;
  uint32_t size_ = 0;
  uint32_t words_ = 0;
  uint32_t stride_ = 0;
  std::vector<uint32_t> tokens_;
};

TEST_P(RowFormatTest, PacksTokensIntoQuadsAndBack) {
  Randomize(0, 0x10000);
  WriteList();
  DmaCommand command = Command();
  Execute(command);
  for (uint32_t u = 0; u < tokens_.size(); ++u) {
    ExpectPacked(u);
  }
  if (GetParam().indexed) {
    return;
  }
  std::ranges::fill(bytes_, kUntouched);
  command.store = true;
  Execute(command);
  for (uint32_t u = 0; u < tokens_.size(); ++u) {
    if (tokens_.at(u) == kNullEntry) {
      continue;
    }
    for (uint32_t i = 0; i < size_; ++i) {
      ASSERT_EQ(Byte(tokens_.at(u) + i), Stored(u, i))
          << "token " << u << " byte " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    DmaEngine, RowFormatTest,
    testing::Values(RowCase{.mem = kAbuf, .tok = 3, .indexed = false},
                    RowCase{.mem = kAbuf, .tok = 17, .indexed = false},
                    RowCase{.mem = kAbuf, .tok = 1, .indexed = true},
                    RowCase{.mem = kAbuf, .tok = 32, .indexed = true},
                    RowCase{.mem = kFbuf, .tok = 2, .indexed = false},
                    RowCase{.mem = kFbuf, .tok = 1, .indexed = true}));

TEST_F(DmaEngineTest, Split8SeparatesNibblePlanes) {
  constexpr uint32_t kRows = 4;
  constexpr uint32_t kRowWords = 3;
  constexpr uint32_t kStride = (kRowWords * 144) + 16;
  Randomize(0, 0x4000);
  DmaCommand command = Load(kWbuf, 0x200, 100, kRows, kRowWords, kStride);
  command.kind = isa::dma::kKindSplit8;
  Execute(command);
  const uint32_t half = kRows * kRowWords / 2;
  for (uint32_t p = 0; p < kRows / 2; ++p) {
    for (uint32_t w = 0; w < kRowWords; ++w) {
      const WbufWord& high = memories_.wbuf.at(100 + (p * kRowWords) + w);
      const WbufWord& low = memories_.wbuf.at(100 + half + (p * kRowWords) + w);
      for (uint32_t c = 0; c < 8; ++c) {
        const uint32_t word =
            0x200 + (((2 * p) + (c / 4)) * kStride) + (w * 144);
        for (uint32_t k = 0; k < 32; ++k) {
          const uint32_t q = Byte(word + ((c % 4) * 32) + k);
          ASSERT_EQ(Nibble(high, (c * 32) + k), q >> 4);
          ASSERT_EQ(Nibble(low, (c * 32) + k), q & 0xFU);
        }
        ASSERT_EQ(high.at(kWbufScaleOffset + c), Byte(word + 128 + (c % 4)));
        ASSERT_EQ(low.at(kWbufScaleOffset + c), Byte(word + 128 + (c % 4)));
      }
    }
  }
}

class T32Split8Test : public DmaEngineTest {
 protected:
  static constexpr uint32_t kRows = 16;
  static constexpr uint32_t kRowWords = 2;
  static constexpr uint32_t kStride = (kRowWords * 144) + 32;
  static constexpr uint32_t kGroups = kRows / 8;
  static constexpr uint32_t kOnChip = 200;

  static uint32_t WordOffset(uint32_t row, uint32_t w) {
    return 0x400 + (row * kStride) + (w * 144);
  }

  [[nodiscard]] uint32_t Element(uint32_t r8, uint32_t w, uint32_t t,
                                 uint32_t d) const {
    return Byte(WordOffset((r8 * 8) + (t / 4), w) + ((t % 4) * 32) + d);
  }

  [[nodiscard]] uint32_t Scale(uint32_t r8, uint32_t w, uint32_t t) const {
    return Byte(WordOffset((r8 * 8) + (t / 4), w) + 128 + (t % 4));
  }

  [[nodiscard]] bool Nonzero(uint32_t r8, uint32_t w, uint32_t t) const {
    for (uint32_t d = 0; d < 32; ++d) {
      if (Element(r8, w, t, d) != 0) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] uint32_t Emax(uint32_t r8, uint32_t w) const {
    uint32_t emax = 0;
    bool any = false;
    for (uint32_t t = 0; t < 32; ++t) {
      if (Nonzero(r8, w, t)) {
        any = true;
        emax = std::max(emax, Scale(r8, w, t));
      }
    }
    return any ? emax : 127;
  }

  [[nodiscard]] uint32_t Aligned(uint32_t r8, uint32_t w, uint32_t t,
                                 uint32_t d, uint32_t emax) const {
    const uint32_t shift = emax - Scale(r8, w, t);
    if (!Nonzero(r8, w, t) || shift >= 9) {
      return 0;
    }
    const int64_t r = reference::NearestEven(
        exact::Real::Integer(SignedByte(Element(r8, w, t, d)))
            .Scaled(-static_cast<int64_t>(shift)));
    return static_cast<uint32_t>(r & 0xFF);
  }

  void ExpectGroup(uint32_t r8, uint32_t w) const {
    const uint32_t emax = Emax(r8, w);
    const uint32_t half = kRows * kRowWords / 2;
    for (uint32_t cg = 0; cg < 4; ++cg) {
      const uint32_t index = kOnChip + (((4 * w) + cg) * kGroups) + r8;
      const WbufWord& high = memories_.wbuf.at(index);
      const WbufWord& low = memories_.wbuf.at(index + half);
      for (uint32_t c = 0; c < 8; ++c) {
        for (uint32_t t = 0; t < 32; ++t) {
          const uint32_t bits = Aligned(r8, w, t, (cg * 8) + c, emax);
          ASSERT_EQ(Nibble(high, (c * 32) + t), bits >> 4)
              << "group " << r8 << " w " << w << " cg " << cg;
          ASSERT_EQ(Nibble(low, (c * 32) + t), bits & 0xFU);
        }
        ASSERT_EQ(high.at(kWbufScaleOffset + c), emax);
        ASSERT_EQ(low.at(kWbufScaleOffset + c), emax);
      }
    }
  }
};

TEST_F(T32Split8Test, TransposesAlignedBlocks) {
  Randomize(0, 0x4000);
  for (uint32_t row = 0; row < kRows; ++row) {
    for (uint32_t w = 0; w < kRowWords; ++w) {
      for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t t = ((row % 8) * 4) + j;
        Put(WordOffset(row, w) + 128 + j, 120 + (((t * 5) + row + w) % 12), 1);
      }
    }
  }
  for (uint32_t j = 0; j < 4; ++j) {
    for (uint32_t k = 0; k < 32; ++k) {
      Put(WordOffset(2, 0) + (j * 32) + k, 0, 1);
    }
    Put(WordOffset(2, 0) + 128 + j, 200, 1);
  }
  for (uint32_t row = 8; row < 16; ++row) {
    for (uint32_t i = 0; i < 128; ++i) {
      Put(WordOffset(row, 1) + i, 0, 1);
    }
  }
  DmaCommand command =
      Load(kWbuf, WordOffset(0, 0), kOnChip, kRows, kRowWords, kStride);
  command.kind = isa::dma::kKindT32Split8;
  Execute(command);
  for (uint32_t r8 = 0; r8 < kGroups; ++r8) {
    for (uint32_t w = 0; w < kRowWords; ++w) {
      ExpectGroup(r8, w);
    }
  }
}

class GatherTest : public DmaEngineTest {
 protected:
  static constexpr uint32_t kBlocks = 3;
  static constexpr uint32_t kRowBytes = (32 * kBlocks) + 16;
  static constexpr uint32_t kFeatureRows = 10;
  static constexpr uint32_t kFeatures = 0x10000;
  static constexpr uint32_t kDst = 300;

  struct List {
    uint32_t offset = 0;
    uint32_t slots = 0;
    uint32_t w_area = 0;
    uint32_t stride = 0;
  };

  uint32_t RandomFloat(uint32_t low) {
    std::uniform_int_distribution<uint32_t> fraction(0, 0x7FFFFF);
    const uint32_t sign = fraction(generator_) & 1U;
    const uint32_t exponent = low + (fraction(generator_) % 8);
    return (sign << 31) | (exponent << 23) | fraction(generator_);
  }

  void WriteFeatures() {
    Randomize(kFeatures, kFeatureRows * kRowBytes);
    for (uint32_t i = 0; i < kFeatureRows; ++i) {
      for (uint32_t b = 0; b < kBlocks; ++b) {
        Put(kFeatures + (i * kRowBytes) + (32 * kBlocks) + b,
            120 + (((i * 3) + b) % 17), 1);
      }
    }
  }

  [[nodiscard]] uint32_t Coefficient(const List& list, uint32_t s, uint32_t t,
                                     uint32_t b, uint32_t bw,
                                     uint32_t row) const {
    const uint32_t w_word = list.w_area + (((b * list.stride) + (s / 8)) * 128);
    const uint32_t c = reference::Mul(
        static_cast<uint32_t>(Little(w_word + (4 * ((t * 8) + (s % 8))), 4)),
        bw);
    const exact::Real scaled = reference::Value(c).Scaled(
        static_cast<int64_t>(Byte(row + (32 * kBlocks) + b)) - 133);
    if (scaled.IsZero() || scaled.TopBit() < -126) {
      return 0;
    }
    if (scaled.TopBit() > 127) {
      return (c & 0x80000000U) | 0x7F7FFFFFU;
    }
    return reference::Round(scaled);
  }

  void WriteList(const List& list) {
    Put(list.offset, Entry(list.w_area), 8);
    for (uint32_t i = 0; i < 0x2000; i += 4) {
      Put(list.w_area + i, RandomFloat(118), 4);
    }
    std::uniform_int_distribution<uint32_t> pick(0, kFeatureRows + 2);
    for (uint32_t s = 0; s < list.slots; ++s) {
      for (uint32_t t = 0; t < kTokens; ++t) {
        for (uint32_t k = 0; k < 4; ++k) {
          const uint32_t index = pick(generator_);
          const uint32_t row = kFeatures + (index * kRowBytes);
          const uint32_t bw = RandomFloat(122);
          const uint32_t entry = index < kFeatureRows ? Entry(row) : kNullEntry;
          Put(list.offset + 16 + (((s * 16) + (t * 4) + k) * 8),
              entry | (uint64_t{bw} << 32), 8);
          if (entry != kNullEntry) {
            Accumulate(list, s, t, bw, row);
          }
        }
      }
    }
  }

  void Accumulate(const List& list, uint32_t s, uint32_t t, uint32_t bw,
                  uint32_t row) {
    for (uint32_t b = 0; b < kBlocks; ++b) {
      const uint32_t coefficient = Coefficient(list, s, t, b, bw, row);
      for (uint32_t v = 0; v < 32; ++v) {
        uint32_t& acc = expected_.at(t).at((b * 32) + v);
        acc = reference::Add(
            acc, reference::Mul(
                     reference::FromInt(SignedByte(Byte(row + (b * 32) + v))),
                     coefficient));
      }
    }
  }

  std::array<std::array<uint32_t, size_t{32} * kBlocks>, kTokens> expected_{};
};

TEST_F(GatherTest, AccumulatesWeightedRows) {
  WriteFeatures();
  const std::array<List, 2> lists = {
      {{.offset = 0x30000, .slots = 12, .w_area = 0x40000, .stride = 4},
       {.offset = 0x34000, .slots = 3, .w_area = 0x50000, .stride = 1}}};
  for (const List& list : lists) {
    WriteList(list);
  }
  const std::vector<FbufWord> before = memories_.fbuf;
  DmaCommand first =
      Load(kFbuf, lists.at(0).offset, kDst, lists.at(0).slots, kBlocks, 0);
  first.kind = isa::dma::kKindGacc;
  first.ddr_stride = lists.at(0).stride * 8;
  Execute(first);
  ASSERT_EQ(memories_.fbuf, before);
  DmaCommand second =
      Load(kFbuf, lists.at(1).offset, kDst, lists.at(1).slots, kBlocks, 0);
  second.kind = isa::dma::kKindGacc;
  second.ddr_stride = lists.at(1).stride * 8;
  second.acc = true;
  second.flush = true;
  Execute(second);
  for (uint32_t g = 0; g < 4 * kBlocks; ++g) {
    for (uint32_t t = 0; t < kTokens; ++t) {
      for (uint32_t c = 0; c < 8; ++c) {
        ASSERT_EQ(memories_.fbuf.at(kDst + g).at((t * 8) + c),
                  expected_.at(t).at((g * 8) + c))
            << "word " << g << " token " << t << " column " << c;
      }
    }
  }
}

TEST_F(GatherTest, ReadsCoefficientsOnlyForSlotGroupsWithRows) {
  WriteFeatures();
  const List list{
      .offset = 0x30000, .slots = 24, .w_area = 0x40000, .stride = 4};
  WriteList(list);
  for (uint32_t s = 0; s < list.slots; ++s) {
    for (uint32_t e = 0; e < 16; ++e) {
      const uint32_t entry =
          s / 8 == 1 ? Entry(kFeatures + ((e % kFeatureRows) * kRowBytes))
                     : kNullEntry;
      Put(list.offset + 16 + (((s * 16) + e) * 8), entry, 4);
    }
  }
  DmaCommand command = Load(kFbuf, list.offset, kDst, list.slots, kBlocks, 0);
  command.kind = isa::dma::kKindGacc;
  command.ddr_stride = list.stride * 8;
  Execute(command);
  const DmaActivity& activity = engine_.Activity();
  EXPECT_EQ(activity.gather_entries, 24U * 16);
  EXPECT_EQ(activity.gather_rows, 8U * 16);
  EXPECT_EQ(activity.null_entries, 16U * 16);
  EXPECT_EQ(activity.gather_row_beats, 8U * 16 * ((2 * kBlocks) + 1));
  EXPECT_EQ(activity.gather_fetches, 1U);
  EXPECT_EQ(activity.read_beats, 1U + (24 * 8) + (2 * kBlocks * 8));
}

using DmaEngineDeathTest = DmaEngineTest;

TEST_F(DmaEngineDeathTest, RejectsATokenSizeOutsideTheFormat) {
  DmaCommand command = Load(kAbuf, 0, 0, 1, 4, 0);
  command.kind = isa::dma::kKindRow;
  command.tok = 0;
  EXPECT_DEATH(Execute(command), "row format with tok = 0");
  command.tok = 33;
  EXPECT_DEATH(Execute(command), "row format with tok = 33");
  command.mem = kFbuf;
  command.tok = 9;
  EXPECT_DEATH(Execute(command), "row format with tok = 9");
}

TEST_F(DmaEngineDeathTest, RejectsAGatherIntoAnotherMemory) {
  DmaCommand command = Load(kAbuf, 0, 0, 1, 1, 0);
  command.kind = isa::dma::kKindGacc;
  EXPECT_DEATH(Execute(command), "GACC writes FBUF");
}

TEST_F(DmaEngineDeathTest, RejectsAnEmptyTransfer) {
  EXPECT_DEATH(Execute(Load(kFbuf, 0, 0, 0, 5, 128)),
               "a DMA transfer with 0 rows of 5 words");
  EXPECT_DEATH(Execute(Load(kAbuf, 0, 0, 5, 0, 144)),
               "a DMA transfer with 5 rows of 0 words");
}

TEST_F(DmaEngineDeathTest, RejectsAStoreOfAnotherKind) {
  DmaCommand command = Load(kFbuf, 0, 0, 1, 4, 0);
  command.store = true;
  command.kind = isa::dma::kKindRowIndexed;
  EXPECT_DEATH(Execute(command), "DMA_ST of kind 2");
}

TEST_F(DmaEngineDeathTest, RejectsAccumulationOutsideAGather) {
  DmaCommand command = Load(kFbuf, 0, 0, 1, 4, 0);
  command.flush = true;
  EXPECT_DEATH(Execute(command), "acc and flush are used only by GACC");
}

}  // namespace
}  // namespace hotarubi::iss
