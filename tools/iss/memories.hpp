#ifndef HOTARUBI_TOOLS_ISS_MEMORIES_HPP_
#define HOTARUBI_TOOLS_ISS_MEMORIES_HPP_

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "word.hpp"

namespace hotarubi::iss {

inline constexpr uint32_t kTokens = 4;
inline constexpr uint32_t kLanes = 8;
inline constexpr uint32_t kBlock = 32;
inline constexpr uint32_t kFbufElements = kTokens * kLanes;
inline constexpr uint32_t kAbufWordBytes = 132;
inline constexpr uint32_t kWbufWordBytes = 144;
inline constexpr uint32_t kAbufScaleOffset = 128;
inline constexpr uint32_t kWbufScaleOffset = 128;

inline constexpr uint32_t kAbufDepth = 8192;
inline constexpr uint32_t kWbufDepth = 2048;
inline constexpr uint32_t kFbufDepth = 4096;
inline constexpr uint32_t kPbufDepth = 512;
inline constexpr uint32_t kTbufDepth = 1024;
inline constexpr uint32_t kCbufDepth = 1024;

constexpr int32_t SignedByte(uint32_t byte) {
  return byte >= 128 ? static_cast<int32_t>(byte) - 256
                     : static_cast<int32_t>(byte);
}

using AbufWord = std::array<uint8_t, kAbufWordBytes>;
using WbufWord = std::array<uint8_t, kWbufWordBytes>;
using FbufWord = std::array<uint32_t, kFbufElements>;

enum class Mem : uint8_t {
  kAbuf = 0,
  kWbuf = 1,
  kFbuf = 2,
  kPbuf = 3,
  kTbuf = 4,
  kCbuf = 5,
};

struct AccessSet {
  std::bitset<kAbufDepth> abuf;
  std::bitset<kWbufDepth> wbuf;
  std::bitset<kFbufDepth> fbuf;
  std::bitset<kPbufDepth> pbuf;
  std::bitset<kTbufDepth> tbuf;
  std::bitset<kCbufDepth> cbuf;

  void Add(Mem mem, uint32_t address);
  void AddRange(Mem mem, uint32_t lo, uint32_t hi);
  void AddTable(Mem mem);
};

struct Conflict {
  Mem mem = Mem::kFbuf;
  uint32_t address = 0;
};

struct Extent {
  Mem mem = Mem::kFbuf;
  uint32_t lo = 0;
  uint32_t hi = 0;

  friend constexpr bool operator==(const Extent&, const Extent&) = default;
};

std::vector<Extent> Extents(const AccessSet& set);
std::optional<Conflict> FirstOutside(const AccessSet& inner,
                                     const AccessSet& outer);

struct AccessLog {
  AccessSet reads;
  AccessSet writes;

  void Read(Mem mem, uint32_t address) { reads.Add(mem, address); }
  void ReadTable(Mem mem) { reads.AddTable(mem); }
  void Write(Mem mem, uint32_t address) { writes.Add(mem, address); }
  void Clear();
};

std::optional<Conflict> FindConflict(const AccessLog& earlier,
                                     const AccessLog& later);

struct Memories {
  Memories();

  std::vector<AbufWord> abuf;
  std::vector<WbufWord> wbuf;
  std::vector<FbufWord> fbuf;
  std::vector<FbufWord> pbuf;
  std::vector<uint64_t> tbuf;
  std::vector<Word128> cbuf;
  AccessLog log;
};

}  // namespace hotarubi::iss

#endif  // HOTARUBI_TOOLS_ISS_MEMORIES_HPP_
