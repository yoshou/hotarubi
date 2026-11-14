#ifndef HOTARUBI_TOOLS_COMPILER_SCHEDULE_HPP_
#define HOTARUBI_TOOLS_COMPILER_SCHEDULE_HPP_

#include <cstdint>
#include <limits>
#include <vector>

#include "isa.hpp"
#include "timing.hpp"

namespace hotarubi::compiler {

inline constexpr uint32_t kNoTile = std::numeric_limits<uint32_t>::max();

struct ScheduleOptions {
  uint32_t window = 64;
  iss::TimingParameters timing = {};
};

struct Span {
  uint32_t tile = kNoTile;
  Mem mem = Mem::kFbuf;
  uint32_t lo = 0;
  uint32_t hi = 0;
  bool write = false;
};

struct Step {
  int engine = 0;
  bool dma = false;
  bool store = false;
  bool loaded = false;
  iss::DdrExtent extent{};
  uint64_t cost = 0;
  std::vector<Span> spans;
};

struct TileSize {
  Mem mem = Mem::kFbuf;
  uint32_t words = 0;
};

struct Plan {
  std::vector<uint32_t> order;
  std::vector<uint32_t> base;
};

[[nodiscard]] Plan ScheduleStream(const std::vector<Step>& steps,
                                  const std::vector<TileSize>& tiles,
                                  const ScheduleOptions& options);

}  // namespace hotarubi::compiler

#endif  // HOTARUBI_TOOLS_COMPILER_SCHEDULE_HPP_
