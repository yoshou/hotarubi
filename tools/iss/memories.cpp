#include "memories.hpp"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <optional>

#include "rom.hpp"

namespace hotarubi::iss {
namespace {

template <size_t N>
std::optional<uint32_t> FirstCommon(const std::bitset<N>& a,
                                    const std::bitset<N>& b) {
  const std::bitset<N> common = a & b;
  if (common.none()) {
    return std::nullopt;
  }
  for (size_t i = 0; i < N; ++i) {
    if (common.test(i)) {
      return static_cast<uint32_t>(i);
    }
  }
  return std::nullopt;
}

std::optional<Conflict> Overlap(const AccessSet& a, const AccessSet& b) {
  if (const auto word = FirstCommon(a.fbuf, b.fbuf)) {
    return Conflict{.mem = Mem::kFbuf, .address = *word};
  }
  if (const auto word = FirstCommon(a.abuf, b.abuf)) {
    return Conflict{.mem = Mem::kAbuf, .address = *word};
  }
  if (const auto word = FirstCommon(a.wbuf, b.wbuf)) {
    return Conflict{.mem = Mem::kWbuf, .address = *word};
  }
  if (const auto word = FirstCommon(a.pbuf, b.pbuf)) {
    return Conflict{.mem = Mem::kPbuf, .address = *word};
  }
  if (const auto word = FirstCommon(a.tbuf, b.tbuf)) {
    return Conflict{.mem = Mem::kTbuf, .address = *word};
  }
  if (const auto word = FirstCommon(a.cbuf, b.cbuf)) {
    return Conflict{.mem = Mem::kCbuf, .address = *word};
  }
  return std::nullopt;
}

}  // namespace

void AccessSet::Add(Mem mem, uint32_t address) {
  switch (mem) {
    case Mem::kAbuf:
      abuf.set(address % kAbufDepth);
      break;
    case Mem::kWbuf:
      wbuf.set(address % kWbufDepth);
      break;
    case Mem::kFbuf:
      fbuf.set(address % kFbufDepth);
      break;
    case Mem::kPbuf:
      pbuf.set(address % kPbufDepth);
      break;
    case Mem::kTbuf:
      tbuf.set(address % kTbufDepth);
      break;
    case Mem::kCbuf:
      cbuf.set(address % kCbufDepth);
      break;
  }
}

void AccessSet::AddRange(Mem mem, uint32_t lo, uint32_t hi) {
  for (uint32_t address = lo; address < hi; ++address) {
    Add(mem, address);
  }
}

void AccessSet::AddTable(Mem mem) {
  if (mem == Mem::kTbuf) {
    tbuf.set();
  } else if (mem == Mem::kCbuf) {
    cbuf.set();
  }
}

void AccessSet::ClearTables() {
  tbuf.reset();
  cbuf.reset();
}

void AccessLog::Clear() {
  reads = {};
  writes = {};
}

std::vector<Extent> Extents(const AccessSet& set) {
  std::vector<Extent> extents;
  const auto hull = [&](Mem mem, const auto& bits) {
    if (bits.none()) {
      return;
    }
    uint32_t lo = 0;
    uint32_t hi = 0;
    bool first = true;
    for (size_t i = 0; i < bits.size(); ++i) {
      if (bits.test(i)) {
        if (first) {
          lo = static_cast<uint32_t>(i);
          first = false;
        }
        hi = static_cast<uint32_t>(i) + 1;
      }
    }
    extents.push_back({.mem = mem, .lo = lo, .hi = hi});
  };
  hull(Mem::kAbuf, set.abuf);
  hull(Mem::kWbuf, set.wbuf);
  hull(Mem::kFbuf, set.fbuf);
  hull(Mem::kPbuf, set.pbuf);
  hull(Mem::kTbuf, set.tbuf);
  hull(Mem::kCbuf, set.cbuf);
  return extents;
}

std::optional<Conflict> FirstOutside(const AccessSet& inner,
                                     const AccessSet& outer) {
  AccessSet complement;
  complement.abuf = ~outer.abuf;
  complement.wbuf = ~outer.wbuf;
  complement.fbuf = ~outer.fbuf;
  complement.pbuf = ~outer.pbuf;
  complement.tbuf = ~outer.tbuf;
  complement.cbuf = ~outer.cbuf;
  return Overlap(inner, complement);
}

std::optional<Conflict> FindConflict(const AccessLog& earlier,
                                     const AccessLog& later) {
  if (const auto conflict = Overlap(earlier.writes, later.reads)) {
    return conflict;
  }
  if (const auto conflict = Overlap(earlier.reads, later.writes)) {
    return conflict;
  }
  return Overlap(earlier.writes, later.writes);
}

Memories::Memories()
    : abuf(kAbufDepth),
      wbuf(kWbufDepth),
      fbuf(kFbufDepth),
      pbuf(kPbufDepth),
      tbuf(kTbufDepth),
      cbuf(kCbufDepth) {
  std::ranges::copy(kRom, cbuf.begin());
}

}  // namespace hotarubi::iss
