#include "dma_engine.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <vector>

#include "ddr.hpp"
#include "fp32.hpp"
#include "hotarubi/require.hpp"
#include "memories.hpp"
#include "word.hpp"

namespace hotarubi::iss {
namespace {

constexpr uint32_t kBeatBytes = 16;
constexpr uint32_t kWordBufferBytes = 144;
constexpr uint32_t kMxint8WordBytes = 144;
constexpr uint32_t kNullEntry = 0xFFFFFFFF;
constexpr uint32_t kModeRaw = 0;
constexpr uint32_t kModeSplit8 = 1;
constexpr uint32_t kModeGather = 3;
constexpr uint32_t kFlagIndex = 1;
constexpr uint32_t kFlagRows = 2;
constexpr uint32_t kFlagAccumulate = 1;
constexpr uint32_t kFlagFlush = 2;
constexpr std::byte kNullScale{0x7F};
constexpr uint32_t kT32Rows = 8;
constexpr uint32_t kT32Tokens = 32;
constexpr uint32_t kGatherSlotGroup = 8;
constexpr uint32_t kGatherEntriesPerSlot = 16;
constexpr uint32_t kGatherMaxBlocks = 8;
constexpr uint32_t kGatherMaxSlots = 2047;
constexpr uint32_t kMaxRowBeats = 65536;
constexpr uint32_t kBurstBeats = 16;

using WordBytes = std::array<std::byte, kWordBufferBytes>;
using Mxint8Word = std::array<std::byte, kMxint8WordBytes>;
using Unit = std::vector<std::byte>;
using Quad = std::array<Unit, kTokens>;

uint32_t BeatsPerWord(uint32_t mem) {
  switch (mem) {
    case 0:
    case 1:
      return 9;
    case 4:
    case 5:
      return 1;
    default:
      return 8;
  }
}

uint32_t RowAddress(const DmaCommand& command, uint32_t row, uint64_t offset) {
  return static_cast<uint32_t>(uint64_t{command.ddr_addr} +
                               (uint64_t{row} * command.ddr_stride) + offset);
}

uint32_t UnitAddress(uint32_t entry) { return (entry & 0x0FFFFFFFU) << 4; }

uint8_t Byte(std::byte value) { return std::to_integer<uint8_t>(value); }

uint64_t Bursts(uint64_t beats) {
  return (beats + kBurstBeats - 1) / kBurstBeats;
}

void CountRead(DmaActivity& activity, uint64_t beats) {
  activity.read_beats += beats;
  activity.read_bursts += Bursts(beats);
}

void CountWrite(DmaActivity& activity, uint64_t beats) {
  activity.write_beats += beats;
  activity.write_bursts += Bursts(beats);
}

std::vector<uint32_t> ReadList(const Ddr& ddr, uint32_t address, uint32_t count,
                               DmaActivity& activity) {
  CountRead(activity, (uint64_t{count} + 3) / 4);
  std::vector<uint32_t> entries(count);
  std::vector<std::byte> bytes(size_t{count} * 4);
  ddr.Read(address, bytes);
  for (size_t i = 0; i < entries.size(); ++i) {
    entries[i] =
        static_cast<uint32_t>(LoadLittle(std::span(bytes).subspan(i * 4, 4)));
  }
  return entries;
}

FbufWord ToFbuf(std::span<const std::byte, kWordBufferBytes> word) {
  FbufWord out{};
  for (size_t i = 0; i < out.size(); ++i) {
    out.at(i) = static_cast<uint32_t>(LoadLittle(word.subspan(i * 4, 4)));
  }
  return out;
}

void WriteWord(Memories& memories, uint32_t mem, uint32_t address,
               std::span<const std::byte, kWordBufferBytes> word) {
  switch (mem) {
    case 0:
      std::ranges::transform(word.first<kAbufWordBytes>(),
                             memories.abuf.at(address % kAbufDepth).begin(),
                             Byte);
      break;
    case 1:
      std::ranges::transform(
          word, memories.wbuf.at(address % kWbufDepth).begin(), Byte);
      break;
    case 2:
      memories.fbuf.at(address % kFbufDepth) = ToFbuf(word);
      break;
    case 3:
      memories.pbuf.at(address % kPbufDepth) = ToFbuf(word);
      break;
    case 4:
      memories.tbuf.at(address % kTbufDepth) = LoadLittle(word.first<8>());
      break;
    case 5:
      memories.cbuf.at(address % kCbufDepth) = LoadWord128(word.first<16>());
      break;
    default:
      break;
  }
}

WordBytes ReadWord(const Memories& memories, uint32_t mem, uint32_t address) {
  WordBytes word{};
  if (mem == 0) {
    std::ranges::transform(memories.abuf.at(address % kAbufDepth), word.begin(),
                           [](uint8_t value) { return std::byte{value}; });
    return word;
  }
  const FbufWord& in = memories.fbuf.at(address % kFbufDepth);
  for (size_t i = 0; i < in.size(); ++i) {
    StoreLittle(in.at(i), std::span(word).subspan(i * 4, 4));
  }
  return word;
}

struct RowFormat {
  bool mxint8 = false;
  uint32_t tok = 0;
  uint32_t words = 0;
  uint32_t data_bytes = 0;
  uint32_t bytes = 0;
};

RowFormat MakeRowFormat(const DmaCommand& command, bool indexed) {
  const uint32_t tok = command.tok == 0 ? 1 : std::min(command.tok, 32U);
  const bool mxint8 = command.mem == 0;
  Require(mxint8 || tok <= 8,
          std::format("FP32 row format with tok = {} above 8", tok));
  const uint32_t words = mxint8 ? tok : 4 * tok;
  uint32_t scale_beats = 0;
  if (mxint8) {
    scale_beats = tok > 16 ? 2 : 1;
  }
  const RowFormat format{.mxint8 = mxint8,
                         .tok = tok,
                         .words = words,
                         .data_bytes = words * 32,
                         .bytes = ((2 * words) + scale_beats) * kBeatBytes};
  const uint32_t units = command.rows * command.row_words;
  Require(units % kTokens == 0,
          std::format("row format with {} tokens, not a multiple of 4", units));
  Require(indexed || uint64_t{command.row_words} * (format.bytes / kBeatBytes) <
                         kMaxRowBeats,
          std::format("row format with {} tokens of {} bytes per row",
                      command.row_words, format.bytes));
  return format;
}

void PackQuad(const RowFormat& format, const Quad& quad, Memories& memories,
              uint32_t mem, uint32_t base) {
  for (uint32_t w = 0; w < format.words; ++w) {
    WordBytes word{};
    for (uint32_t j = 0; j < kTokens; ++j) {
      std::ranges::copy(
          std::span(quad.at(j)).subspan(size_t{w} * kBlock, kBlock),
          std::span(word).subspan(size_t{j} * kBlock).begin());
      if (format.mxint8) {
        word.at(kAbufScaleOffset + j) = quad.at(j).at(format.data_bytes + w);
      }
    }
    WriteWord(memories, mem, base + w, word);
  }
}

void UnpackQuad(const RowFormat& format, const Memories& memories, uint32_t mem,
                uint32_t base, Quad& quad) {
  for (Unit& unit : quad) {
    unit.assign(format.bytes, std::byte{0});
  }
  for (uint32_t w = 0; w < format.words; ++w) {
    const WordBytes word = ReadWord(memories, mem, base + w);
    for (uint32_t j = 0; j < kTokens; ++j) {
      std::ranges::copy(
          std::span(word).subspan(size_t{j} * kBlock, kBlock),
          std::span(quad.at(j)).subspan(size_t{w} * kBlock).begin());
      if (format.mxint8) {
        quad.at(j).at(format.data_bytes + w) = word.at(kAbufScaleOffset + j);
      }
    }
  }
}

void LoadWords(const DmaCommand& command, const Ddr& ddr, Memories& memories,
               DmaActivity& activity) {
  const uint32_t beats = BeatsPerWord(command.mem);
  const uint32_t bytes = beats * kBeatBytes;
  const uint32_t count = command.rows * command.row_words;
  if ((command.flags & kFlagIndex) != 0) {
    const std::vector<uint32_t> entries =
        ReadList(ddr, command.ddr_addr, count, activity);
    for (uint32_t e = 0; e < count; ++e) {
      WordBytes word{};
      if (entries[e] != kNullEntry) {
        CountRead(activity, beats);
        ddr.Read(UnitAddress(entries[e]), std::span(word).first(bytes));
      } else {
        ++activity.null_entries;
        if (command.mem == 0) {
          std::ranges::fill(std::span(word).subspan(kAbufScaleOffset, kTokens),
                            kNullScale);
        }
      }
      WriteWord(memories, command.mem, command.onchip + e, word);
    }
    return;
  }
  for (uint32_t r = 0; r < command.rows; ++r) {
    CountRead(activity, uint64_t{command.row_words} * beats);
    for (uint32_t w = 0; w < command.row_words; ++w) {
      WordBytes word{};
      ddr.Read(RowAddress(command, r, uint64_t{w} * bytes),
               std::span(word).first(bytes));
      WriteWord(memories, command.mem,
                command.onchip + (r * command.row_words) + w, word);
    }
  }
}

void LoadRows(const DmaCommand& command, const Ddr& ddr, Memories& memories,
              DmaActivity& activity) {
  const bool indexed = (command.flags & kFlagIndex) != 0;
  const RowFormat format = MakeRowFormat(command, indexed);
  const uint32_t count = command.rows * command.row_words;
  const std::vector<uint32_t> entries =
      indexed ? ReadList(ddr, command.ddr_addr, count, activity)
              : std::vector<uint32_t>{};
  Quad quad;
  for (uint32_t u = 0; u < count; ++u) {
    Unit& unit = quad.at(u % kTokens);
    unit.assign(format.bytes, std::byte{0});
    if (!indexed) {
      CountRead(activity, format.bytes / kBeatBytes);
      ddr.Read(RowAddress(command, u / command.row_words,
                          uint64_t{u % command.row_words} * format.bytes),
               unit);
    } else if (entries[u] != kNullEntry) {
      CountRead(activity, format.bytes / kBeatBytes);
      ddr.Read(UnitAddress(entries[u]), unit);
    } else {
      ++activity.null_entries;
      if (format.mxint8) {
        std::ranges::fill(
            std::span(unit).subspan(format.data_bytes, format.tok), kNullScale);
      }
    }
    if (u % kTokens == kTokens - 1) {
      PackQuad(format, quad, memories, command.mem,
               command.onchip + ((u / kTokens) * format.words));
    }
  }
}

void StoreWords(const DmaCommand& command, Ddr& ddr, const Memories& memories,
                DmaActivity& activity) {
  const uint32_t beats = BeatsPerWord(command.mem);
  const uint32_t bytes = beats * kBeatBytes;
  const uint32_t count = command.rows * command.row_words;
  if ((command.flags & kFlagIndex) != 0) {
    const std::vector<uint32_t> entries =
        ReadList(ddr, command.ddr_addr, count, activity);
    for (uint32_t e = 0; e < count; ++e) {
      if (entries[e] != kNullEntry) {
        CountWrite(activity, beats);
        const WordBytes word =
            ReadWord(memories, command.mem, command.onchip + e);
        ddr.Write(UnitAddress(entries[e]), std::span(word).first(bytes));
      } else {
        ++activity.null_entries;
      }
    }
    return;
  }
  for (uint32_t r = 0; r < command.rows; ++r) {
    CountWrite(activity, uint64_t{command.row_words} * beats);
    for (uint32_t w = 0; w < command.row_words; ++w) {
      const WordBytes word = ReadWord(
          memories, command.mem, command.onchip + (r * command.row_words) + w);
      ddr.Write(RowAddress(command, r, uint64_t{w} * bytes),
                std::span(word).first(bytes));
    }
  }
}

void StoreRows(const DmaCommand& command, Ddr& ddr, const Memories& memories,
               DmaActivity& activity) {
  const bool indexed = (command.flags & kFlagIndex) != 0;
  const RowFormat format = MakeRowFormat(command, indexed);
  const uint32_t count = command.rows * command.row_words;
  const std::vector<uint32_t> entries =
      indexed ? ReadList(ddr, command.ddr_addr, count, activity)
              : std::vector<uint32_t>{};
  Quad quad;
  for (uint32_t q = 0; q < count / kTokens; ++q) {
    UnpackQuad(format, memories, command.mem,
               command.onchip + (q * format.words), quad);
    for (uint32_t j = 0; j < kTokens; ++j) {
      const uint32_t u = (q * kTokens) + j;
      if (!indexed) {
        CountWrite(activity, format.bytes / kBeatBytes);
        ddr.Write(RowAddress(command, u / command.row_words,
                             uint64_t{u % command.row_words} * format.bytes),
                  quad.at(j));
      } else if (entries[u] != kNullEntry) {
        CountWrite(activity, format.bytes / kBeatBytes);
        ddr.Write(UnitAddress(entries[u]), quad.at(j));
      } else {
        ++activity.null_entries;
      }
    }
  }
}

void SetNibble(WbufWord& word, uint32_t index, uint32_t value) {
  uint8_t& byte = word.at(index / 2);
  byte = index % 2 == 0 ? static_cast<uint8_t>((byte & 0xF0U) | (value & 0xFU))
                        : static_cast<uint8_t>((byte & 0x0FU) | (value << 4));
}

void LoadSplit8(const DmaCommand& command, const Ddr& ddr, Memories& memories,
                DmaActivity& activity) {
  Require(command.rows % 2 == 0,
          std::format("SPLIT8 with an odd number of rows {}", command.rows));
  const uint32_t half = (command.rows * command.row_words) / 2;
  for (uint32_t p = 0; p < command.rows / 2; ++p) {
    for (uint32_t w = 0; w < command.row_words; ++w) {
      WbufWord high{};
      WbufWord low{};
      for (uint32_t r = 0; r < 2; ++r) {
        Mxint8Word in{};
        CountRead(activity, in.size() / kBeatBytes);
        ddr.Read(RowAddress(command, (2 * p) + r, uint64_t{w} * in.size()), in);
        for (uint32_t k = 0; k < kTokens * kBlock; ++k) {
          const uint32_t value = Byte(in.at(k));
          SetNibble(high, (r * kTokens * kBlock) + k, value >> 4);
          SetNibble(low, (r * kTokens * kBlock) + k, value);
        }
        for (uint32_t s = 0; s < kTokens; ++s) {
          const uint8_t scale = Byte(in.at(kAbufScaleOffset + s));
          high.at(kWbufScaleOffset + (r * kTokens) + s) = scale;
          low.at(kWbufScaleOffset + (r * kTokens) + s) = scale;
        }
      }
      const uint32_t index = (p * command.row_words) + w;
      memories.wbuf.at((command.onchip + index) % kWbufDepth) = high;
      memories.wbuf.at((command.onchip + half + index) % kWbufDepth) = low;
    }
  }
}

std::array<WbufWord, 8> TransposeGroup(
    const std::array<Mxint8Word, kT32Rows>& in) {
  std::array<bool, kT32Tokens> nonzero{};
  std::array<uint32_t, kT32Tokens> scales{};
  uint32_t emax = 0;
  bool any = false;
  for (uint32_t t = 0; t < kT32Tokens; ++t) {
    const Mxint8Word& word = in.at(t / kTokens);
    const auto block =
        std::span(word).subspan(size_t{t % kTokens} * kBlock, kBlock);
    nonzero.at(t) = std::ranges::any_of(
        block, [](std::byte b) { return b != std::byte{0}; });
    scales.at(t) = Byte(word.at(kAbufScaleOffset + (t % kTokens)));
    if (nonzero.at(t)) {
      any = true;
      emax = std::max(emax, scales.at(t));
    }
  }
  if (!any) {
    emax = 127;
  }
  std::array<WbufWord, 8> out{};
  for (uint32_t t = 0; t < kT32Tokens; ++t) {
    const Mxint8Word& word = in.at(t / kTokens);
    for (uint32_t d = 0; d < kBlock; ++d) {
      const uint8_t value =
          nonzero.at(t)
              ? AlignElement(Byte(word.at(((t % kTokens) * kBlock) + d)),
                             emax - scales.at(t))
              : 0;
      const uint32_t nibble = ((d % kLanes) * kBlock) + t;
      SetNibble(out.at(d / kLanes), nibble, static_cast<uint32_t>(value) >> 4);
      SetNibble(out.at(4 + (d / kLanes)), nibble, value);
    }
  }
  for (WbufWord& word : out) {
    std::ranges::fill(std::span(word).subspan(kWbufScaleOffset, kLanes),
                      static_cast<uint8_t>(emax));
  }
  return out;
}

void LoadT32(const DmaCommand& command, const Ddr& ddr, Memories& memories,
             DmaActivity& activity) {
  Require(command.rows % kT32Rows == 0,
          std::format("T32_SPLIT8 with {} rows, not a multiple of 8",
                      command.rows));
  const uint32_t groups = command.rows / kT32Rows;
  const uint32_t half = (command.rows * command.row_words) / 2;
  for (uint32_t g = 0; g < groups; ++g) {
    for (uint32_t w = 0; w < command.row_words; ++w) {
      std::array<Mxint8Word, kT32Rows> in{};
      for (uint32_t f = 0; f < kT32Rows; ++f) {
        CountRead(activity, kMxint8WordBytes / kBeatBytes);
        ddr.Read(RowAddress(command, (g * kT32Rows) + f,
                            uint64_t{w} * kMxint8WordBytes),
                 in.at(f));
      }
      const std::array<WbufWord, 8> out = TransposeGroup(in);
      const uint32_t base = command.onchip + g + (w * 4 * groups);
      for (uint32_t cg = 0; cg < 4; ++cg) {
        const uint32_t address = base + (cg * groups);
        memories.wbuf.at(address % kWbufDepth) = out.at(cg);
        memories.wbuf.at((address + half) % kWbufDepth) = out.at(4 + cg);
      }
    }
  }
}

struct GatherWeights {
  uint32_t base = 0;
  uint32_t stride = 0;
  uint32_t blocks = 0;
};

using GatherCoefficients = std::array<FbufWord, kGatherMaxBlocks>;

GatherCoefficients LoadGatherCoefficients(const Ddr& ddr,
                                          const GatherWeights& weights,
                                          uint32_t group,
                                          DmaActivity& activity) {
  GatherCoefficients w{};
  for (uint32_t i = 0; i < weights.blocks; ++i) {
    std::array<std::byte, size_t{kFbufElements} * 4> chunk{};
    CountRead(activity, chunk.size() / kBeatBytes);
    ddr.Read(weights.base + (group * 128) + (i * weights.stride), chunk);
    for (uint32_t e = 0; e < kFbufElements; ++e) {
      w.at(i).at(e) = static_cast<uint32_t>(
          LoadLittle(std::span(chunk).subspan(size_t{e} * 4, 4)));
    }
  }
  return w;
}

void AccumulateGatherBlock(uint32_t w, uint32_t bw,
                           std::span<const std::byte> row, uint32_t block,
                           uint32_t blocks, std::span<uint32_t> acc) {
  const uint32_t coefficient = ScaleCoefficient(
      Fp32Mul(w, bw), Byte(row[(2 * blocks * kBeatBytes) + block]));
  for (uint32_t v = 0; v < kBlock; ++v) {
    acc[v] = Fp32Add(
        acc[v],
        Fp32Mul(Int8ToFp32(Byte(row[(block * kBlock) + v])), coefficient));
  }
}

void FlushGather(
    const std::array<std::array<uint32_t, kGatherChannels>, kTokens>& acc,
    uint32_t blocks, uint32_t onchip, Memories& memories) {
  for (uint32_t g = 0; g < blocks * 4; ++g) {
    FbufWord word{};
    for (uint32_t t = 0; t < kTokens; ++t) {
      for (uint32_t c = 0; c < kLanes; ++c) {
        word.at((t * kLanes) + c) = acc.at(t).at((g * kLanes) + c);
      }
    }
    memories.fbuf.at(((onchip & 0x1FFFU) + g) % kFbufDepth) = word;
  }
}

}  // namespace

uint32_t Int8ToFp32(uint8_t value) {
  if (value == 0) {
    return 0;
  }
  const bool negative = value >= 128;
  const uint32_t magnitude = negative ? 256U - value : value;
  const int32_t top = std::bit_width(magnitude) - 1;
  return (negative ? kFp32SignBit : 0U) |
         (static_cast<uint32_t>(127 + top) << 23) |
         ((magnitude << (23 - top)) & 0x7FFFFFU);
}

uint32_t ScaleCoefficient(uint32_t coefficient, uint32_t scale) {
  const int32_t exponent =
      static_cast<int32_t>(Fp32Exponent(coefficient) + scale) - 133;
  if (Fp32Exponent(coefficient) == 0 || exponent <= 0) {
    return 0;
  }
  const uint32_t sign = coefficient & kFp32SignBit;
  if (exponent >= 255) {
    return sign | 0x7F7FFFFFU;
  }
  return sign | (static_cast<uint32_t>(exponent) << 23) |
         (coefficient & 0x7FFFFFU);
}

uint8_t AlignElement(uint8_t value, uint32_t shift) {
  if (shift >= 9) {
    return 0;
  }
  const int32_t shifted = (SignedByte(value) * 256) >> shift;
  const bool guard = ((shifted >> 7) & 1) != 0;
  const bool sticky = (shifted & 0x7F) != 0;
  const bool odd = ((shifted >> 8) & 1) != 0;
  return static_cast<uint8_t>(
      static_cast<uint32_t>((shifted >> 8) +
                            (guard && (sticky || odd) ? 1 : 0)) &
      0xFFU);
}

void DmaEngine::Execute(const DmaCommand& command, Ddr& ddr,
                        Memories& memories) {
  activity_ = {};
  const bool rows = (command.flags & kFlagRows) != 0;
  if (command.store) {
    if (command.rows * command.row_words == 0) {
      return;
    }
    if (rows) {
      StoreRows(command, ddr, memories, activity_);
    } else {
      StoreWords(command, ddr, memories, activity_);
    }
    return;
  }
  if (command.mode == kModeGather) {
    Gather(command, ddr, memories);
    return;
  }
  if (command.rows * command.row_words == 0) {
    return;
  }
  if (command.mode == kModeRaw) {
    if (rows) {
      LoadRows(command, ddr, memories, activity_);
    } else {
      LoadWords(command, ddr, memories, activity_);
    }
    return;
  }
  Require(command.flags == 0, "SPLIT8 and T32_SPLIT8 do not take flags");
  Require(command.mem == static_cast<uint32_t>(Mem::kWbuf),
          "SPLIT8 and T32_SPLIT8 write WBUF");
  if (command.mode == kModeSplit8) {
    LoadSplit8(command, ddr, memories, activity_);
  } else {
    LoadT32(command, ddr, memories, activity_);
  }
}

void DmaEngine::Gather(const DmaCommand& command, const Ddr& ddr,
                       Memories& memories) {
  const uint32_t blocks = command.row_words;
  Require(blocks >= 1 && blocks <= kGatherMaxBlocks,
          std::format("GACC with {} blocks per token", blocks));
  Require(command.rows <= kGatherMaxSlots,
          std::format("GACC with {} slots", command.rows));
  if ((command.flags & kFlagAccumulate) == 0) {
    for (auto& token : accumulators_) {
      token.fill(0);
    }
  }
  const uint64_t entries = uint64_t{command.rows} * kGatherEntriesPerSlot;
  activity_.gather_entries = entries;
  CountRead(activity_, 1 + (entries / 2));
  const uint64_t header = ddr.Read64(command.ddr_addr);
  const GatherWeights weights{
      .base = UnitAddress(static_cast<uint32_t>(header)) +
              (((command.ddr_stride & 0xFFFFU) *
                static_cast<uint32_t>((header >> 48) & 0xFFFFU))
               << 7),
      .stride = static_cast<uint32_t>((header >> 32) & 0xFFFFU) << 7,
      .blocks = blocks,
  };
  GatherCoefficients w{};
  std::vector<std::byte> row(((size_t{2} * blocks) + 1) * kBeatBytes);
  for (uint32_t s = 0; s < command.rows; ++s) {
    if (s % kGatherSlotGroup == 0) {
      w = LoadGatherCoefficients(ddr, weights, s / kGatherSlotGroup, activity_);
    }
    for (uint32_t t = 0; t < kTokens; ++t) {
      for (uint32_t k = 0; k < kTokens; ++k) {
        const uint64_t entry =
            ddr.Read64(command.ddr_addr + kBeatBytes +
                       (((s * kGatherEntriesPerSlot) + (t * kTokens) + k) * 8));
        if (static_cast<uint32_t>(entry) == kNullEntry) {
          ++activity_.null_entries;
          continue;
        }
        CountRead(activity_, row.size() / kBeatBytes);
        ++activity_.gather_rows;
        ddr.Read(UnitAddress(static_cast<uint32_t>(entry)), row);
        for (uint32_t i = 0; i < blocks; ++i) {
          AccumulateGatherBlock(
              w.at(i).at((t * kLanes) + (s % kGatherSlotGroup)),
              static_cast<uint32_t>(entry >> 32), row, i, blocks,
              std::span(accumulators_.at(t))
                  .subspan(size_t{i} * kBlock, kBlock));
        }
      }
    }
  }
  if ((command.flags & kFlagFlush) != 0) {
    FlushGather(accumulators_, blocks, command.onchip, memories);
  }
}

}  // namespace hotarubi::iss
