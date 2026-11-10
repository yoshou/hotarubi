#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "fp32_ref.hpp"
#include "npu_bench.hpp"
#include "npu_data.hpp"
#include "npu_isa.hpp"

namespace hotarubi::tb {
namespace {

constexpr uint32_t kRowArea = Ddr::kBase + 0x100000U;
constexpr uint32_t kWArea = Ddr::kBase + 0x400000U;
constexpr uint32_t kListArea = Ddr::kBase + 0x600000U;
constexpr uint32_t kOutArea = Ddr::kBase + 0x800000U;
constexpr uint32_t kInitArea = Ddr::kBase + 0x900000U;
constexpr uint32_t kNull = 0xFFFFFFFFU;

struct GaccList {
  uint32_t list_address = 0;
  uint32_t w_address = 0;
  uint32_t w_stride = 0;
  uint32_t quad_row_words = 0;
  uint32_t slots = 0;
  std::vector<uint32_t> rows;
  std::vector<uint32_t> bw;
};

using Accumulators = std::array<std::vector<uint32_t>, 4>;

uint32_t TokenBytes(uint32_t row_words) {
  return (32 * row_words) + (16 * ((row_words + 15) / 16));
}

uint32_t ScaleByPowerOfTwo(uint32_t value, int exponent) {
  const uint32_t field = (value >> 23) & 0xFFU;
  if (field == 0) {
    return kFp32Zero;
  }
  return Fp32Round((value & kFp32SignBit) != 0, (value & 0x7FFFFFU) | 0x800000U,
                   static_cast<int>(field) - 150 + exponent);
}

GaccList MakeList(std::mt19937& rng, Ddr& ddr, uint32_t list_address,
                  uint32_t slots, uint32_t row_words, uint32_t quad,
                  uint32_t row_pool, uint32_t w_address,
                  uint32_t null_in_nine = 1,
                  const std::vector<bool>& null_slots = {}) {
  GaccList list{.list_address = list_address,
                .w_address = w_address,
                .w_stride = 0,
                .quad_row_words = (slots + 7) / 8,
                .slots = slots,
                .rows = {},
                .bw = {}};
  list.w_stride = (quad + 2) * list.quad_row_words;
  const uint32_t w_words = row_words * list.w_stride;
  for (uint32_t i = 0; i < w_words; ++i) {
    FbufWord word{};
    for (uint32_t& value : word) {
      value = RandomFp32(rng, -3, 3);
    }
    WriteFbuf(ddr, w_address + (i * kFbufWordBytes), word);
  }
  ddr.Write32(list_address,
              (w_address + (quad * list.quad_row_words * kFbufWordBytes)) / 16);
  ddr.Write32(list_address + 4, 0);
  ddr.Write32(list_address + 8, 0);
  ddr.Write32(list_address + 12, 0);
  for (uint32_t e = 0; e < slots * 16; ++e) {
    const bool null_slot = !null_slots.empty() && null_slots.at(e / 16);
    const bool null_entry = rng() % 9 < null_in_nine || null_slot;
    const uint32_t row =
        null_entry
            ? kNull
            : (kRowArea + ((rng() % row_pool) * TokenBytes(row_words))) / 16;
    list.rows.push_back(row);
    list.bw.push_back(RandomFp32(rng, -4, 2));
    ddr.Write32(list_address + 16 + (8 * e), row);
    ddr.Write32(list_address + 20 + (8 * e), list.bw.back());
  }
  return list;
}

void WriteRowPool(std::mt19937& rng, Ddr& ddr, uint32_t row_pool,
                  uint32_t row_words) {
  for (uint32_t r = 0; r < row_pool; ++r) {
    const uint32_t base = kRowArea + (r * TokenBytes(row_words));
    for (uint32_t i = 0; i < 32 * row_words; ++i) {
      ddr.Write8(base + i, static_cast<uint8_t>(rng()));
    }
    for (uint32_t i = 32 * row_words; i < TokenBytes(row_words); ++i) {
      const uint32_t b = i - (32 * row_words);
      ddr.Write8(base + i,
                 b < row_words ? static_cast<uint8_t>(118 + (rng() % 16)) : 0);
    }
  }
}

void ReferenceGacc(const Ddr& ddr, const GaccList& list, uint32_t row_words,
                   uint32_t quad, Accumulators& acc) {
  for (uint32_t s = 0; s < list.slots; ++s) {
    for (uint32_t t = 0; t < 4; ++t) {
      for (uint32_t k = 0; k < 4; ++k) {
        const uint32_t e = (((s * 4) + t) * 4) + k;
        if (list.rows[e] == kNull) {
          continue;
        }
        const uint32_t row = list.rows[e] * 16;
        for (uint32_t ch = 0; ch < 32 * row_words; ++ch) {
          const uint32_t kb = ch / 32;
          const uint32_t w_word =
              (kb * list.w_stride) + (quad * list.quad_row_words) + (s / 8);
          const uint32_t w =
              ddr.Read32(list.w_address + (w_word * kFbufWordBytes) +
                         (4 * ((t * 8) + (s % 8))));
          const uint32_t c = Fp32Mul(w, list.bw[e]);
          const int e_row =
              static_cast<int>(ddr.Read8(row + (32 * row_words) + kb)) - 127;
          const uint32_t scaled = ScaleByPowerOfTwo(c, e_row - 6);
          const auto q = static_cast<int8_t>(ddr.Read8(row + ch));
          acc.at(t)[ch] =
              Fp32Add(acc.at(t)[ch], Fp32Mul(Fp32FromInt(q, 0), scaled));
        }
      }
    }
  }
}

std::vector<FbufWord> ToWords(const Accumulators& acc, uint32_t row_words) {
  std::vector<FbufWord> words(size_t{4} * row_words);
  for (uint32_t g = 0; g < words.size(); ++g) {
    for (uint32_t t = 0; t < 4; ++t) {
      for (uint32_t c = 0; c < 8; ++c) {
        words[g].at((t * 8) + c) = acc.at(t)[(8 * g) + c];
      }
    }
  }
  return words;
}

constexpr uint32_t kGaccAcc = 1;
constexpr uint32_t kGaccFlush = 2;

Instruction Gacc(const GaccList& list, uint32_t row_words, uint32_t dst,
                 uint32_t flags) {
  return DmaLoad({.ddr_addr = list.list_address,
                  .mem = Mem::kFbuf,
                  .onchip = dst,
                  .rows = list.slots,
                  .row_words = row_words,
                  .ddr_stride = list.w_stride * kFbufWordBytes,
                  .kind = DmaKind::kGacc,
                  .acc = (flags & kGaccAcc) != 0,
                  .flush = (flags & kGaccFlush) != 0});
}

void RunAndCompare(NpuBench& bench, std::vector<Instruction> program,
                   uint32_t dst, const std::vector<FbufWord>& expected) {
  program.push_back(
      DmaStore({.ddr_addr = kOutArea,
                .mem = Mem::kFbuf,
                .onchip = dst,
                .rows = 1,
                .row_words = static_cast<uint32_t>(expected.size())}));
  program.push_back(Halt());
  ASSERT_TRUE(bench.Run(program));
  EXPECT_EQ(bench.ReadReg(0x04) & 0x6U, 0x2U);
  EXPECT_TRUE(bench.Errors().empty());
  for (uint32_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(ReadFbuf(bench.Memory(), kOutArea + (i * kFbufWordBytes)),
              expected[i])
        << "word " << i;
  }
}

std::vector<FbufWord> Preload(std::mt19937& rng, Ddr& ddr, uint32_t words) {
  std::vector<FbufWord> init(words);
  for (uint32_t i = 0; i < words; ++i) {
    for (uint32_t& value : init[i]) {
      value = RandomFp32(rng, -2, 2);
    }
    WriteFbuf(ddr, kInitArea + (i * kFbufWordBytes), init[i]);
  }
  return init;
}

TEST(GaccTest, SingleListFromZero) {
  std::mt19937 rng(51);
  NpuBench bench(51);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kRowWords = 2;
  constexpr uint32_t kSlots = 16;
  constexpr uint32_t kQuad = 3;
  constexpr uint32_t kDst = 40;
  WriteRowPool(rng, ddr, 64, kRowWords);
  const GaccList list =
      MakeList(rng, ddr, kListArea, kSlots, kRowWords, kQuad, 64, kWArea);
  Accumulators acc;
  for (std::vector<uint32_t>& token : acc) {
    token.assign(size_t{32} * kRowWords, kFp32Zero);
  }
  ReferenceGacc(ddr, list, kRowWords, kQuad, acc);
  Preload(rng, ddr, 4 * kRowWords);
  RunAndCompare(bench,
                {DmaLoad({.ddr_addr = kInitArea,
                          .mem = Mem::kFbuf,
                          .onchip = kDst,
                          .rows = 1,
                          .row_words = 4 * kRowWords}),
                 Gacc(list, kRowWords, kDst, kGaccFlush)},
                kDst, ToWords(acc, kRowWords));
}

TEST(GaccTest, ChainedListsAccumulate) {
  std::mt19937 rng(52);
  NpuBench bench(52);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kRowWords = 3;
  constexpr uint32_t kQuad = 1;
  constexpr uint32_t kDst = 7;
  WriteRowPool(rng, ddr, 48, kRowWords);
  const GaccList first =
      MakeList(rng, ddr, kListArea, 8, kRowWords, kQuad, 48, kWArea);
  const GaccList second = MakeList(rng, ddr, kListArea + 0x10000U, 24,
                                   kRowWords, kQuad, 48, kWArea + 0x80000U);
  Accumulators acc;
  for (std::vector<uint32_t>& token : acc) {
    token.assign(size_t{32} * kRowWords, kFp32Zero);
  }
  ReferenceGacc(ddr, first, kRowWords, kQuad, acc);
  ReferenceGacc(ddr, second, kRowWords, kQuad, acc);
  RunAndCompare(bench,
                {Gacc(first, kRowWords, kDst, 0),
                 Gacc(second, kRowWords, kDst, kGaccAcc | kGaccFlush)},
                kDst, ToWords(acc, kRowWords));
}

TEST(GaccTest, WithoutFlushLeavesFbufUnchanged) {
  std::mt19937 rng(53);
  NpuBench bench(53);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kRowWords = 1;
  constexpr uint32_t kDst = 100;
  WriteRowPool(rng, ddr, 16, kRowWords);
  const GaccList list =
      MakeList(rng, ddr, kListArea, 8, kRowWords, 0, 16, kWArea);
  const std::vector<FbufWord> init = Preload(rng, ddr, 4 * kRowWords);
  RunAndCompare(bench,
                {DmaLoad({.ddr_addr = kInitArea,
                          .mem = Mem::kFbuf,
                          .onchip = kDst,
                          .rows = 1,
                          .row_words = 4 * kRowWords}),
                 Gacc(list, kRowWords, kDst, 0)},
                kDst, init);
}

TEST(GaccTest, MaximumBlocksPerToken) {
  std::mt19937 rng(54);
  NpuBench bench(54);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kRowWords = 8;
  constexpr uint32_t kQuad = 5;
  constexpr uint32_t kDst = 200;
  WriteRowPool(rng, ddr, 32, kRowWords);
  const GaccList list =
      MakeList(rng, ddr, kListArea, 8, kRowWords, kQuad, 32, kWArea);
  Accumulators acc;
  for (std::vector<uint32_t>& token : acc) {
    token.assign(size_t{32} * kRowWords, kFp32Zero);
  }
  ReferenceGacc(ddr, list, kRowWords, kQuad, acc);
  RunAndCompare(bench, {Gacc(list, kRowWords, kDst, kGaccFlush)}, kDst,
                ToWords(acc, kRowWords));
}

TEST(GaccTest, PartialLastSlotGroup) {
  std::mt19937 rng(55);
  NpuBench bench(55);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kRowWords = 5;
  constexpr uint32_t kSlots = 13;
  constexpr uint32_t kQuad = 2;
  constexpr uint32_t kDst = 300;
  WriteRowPool(rng, ddr, 40, kRowWords);
  const GaccList list =
      MakeList(rng, ddr, kListArea, kSlots, kRowWords, kQuad, 40, kWArea);
  Accumulators acc;
  for (std::vector<uint32_t>& token : acc) {
    token.assign(size_t{32} * kRowWords, kFp32Zero);
  }
  ReferenceGacc(ddr, list, kRowWords, kQuad, acc);
  RunAndCompare(bench, {Gacc(list, kRowWords, kDst, kGaccFlush)}, kDst,
                ToWords(acc, kRowWords));
}

void RunFromZero(uint32_t seed, uint32_t row_words, uint32_t slots,
                 uint32_t null_in_nine, const std::vector<bool>& null_slots) {
  std::mt19937 rng(seed);
  NpuBench bench(seed);
  Ddr& ddr = bench.Memory();
  constexpr uint32_t kQuad = 1;
  constexpr uint32_t kDst = 64;
  WriteRowPool(rng, ddr, 64, row_words);
  const GaccList list = MakeList(rng, ddr, kListArea, slots, row_words, kQuad,
                                 64, kWArea, null_in_nine, null_slots);
  Accumulators acc;
  for (std::vector<uint32_t>& token : acc) {
    token.assign(size_t{32} * row_words, kFp32Zero);
  }
  ReferenceGacc(ddr, list, row_words, kQuad, acc);
  RunAndCompare(bench, {Gacc(list, row_words, kDst, kGaccFlush)}, kDst,
                ToWords(acc, row_words));
}

TEST(GaccTest, ManySlotGroupsWithMostEntriesNull) {
  RunFromZero(58, 8, 200, 7, {});
}

TEST(GaccTest, NullSlotGroupsBetweenDenseGroups) {
  std::vector<bool> null_slots(72, false);
  for (const uint32_t group : {0U, 2U, 3U, 6U, 8U}) {
    for (uint32_t s = 0; s < 8; ++s) {
      null_slots.at((group * 8) + s) = true;
    }
  }
  RunFromZero(59, 3, 72, 0, null_slots);
}

TEST(GaccTest, ListWithOnlyNullEntries) {
  RunFromZero(60, 8, 40, 0, std::vector<bool>(40, true));
}

}  // namespace
}  // namespace hotarubi::tb
