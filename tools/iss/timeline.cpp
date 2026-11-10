#include "timeline.hpp"

#include <algorithm>
#include <cstdint>

#include "trace_record.hpp"

namespace hotarubi::iss {
namespace {

constexpr std::array<uint32_t, Timeline::kEngines> kWaitBits = {0x1, 0x4, 0x8};

}  // namespace

void Timeline::Start() {
  last_issue_ = 0;
  issued_ = 0;
  free_.fill(0);
  busy_.fill(0);
}

Span Timeline::Issue(Source source, uint32_t wait, uint64_t cost) {
  uint64_t issue = issued_ == 0 ? start_cycles_ : last_issue_ + 1;
  for (int e = 0; e < kEngines; ++e) {
    if ((wait & kWaitBits.at(e)) != 0) {
      issue = std::max(issue, free_.at(e) + 1);
    }
  }
  const auto engine = static_cast<size_t>(source);
  if (engine < kEngines) {
    issue = std::max(issue, free_.at(engine) + 1);
  }
  Span span{.issue = issue, .start = issue, .completion = issue};
  if (engine < kEngines) {
    span.completion = span.start + std::max<uint64_t>(cost, 1);
    free_.at(engine) = span.completion;
    busy_.at(engine) += span.completion - issue;
  }
  last_issue_ = issue;
  ++issued_;
  return span;
}

uint64_t Timeline::Cycles() const { return issued_ == 0 ? 0 : last_issue_ + 1; }

uint64_t Timeline::Stall() const {
  const uint64_t cycles = Cycles();
  const uint64_t busy = start_cycles_ + issued_;
  return cycles > busy ? cycles - busy : 0;
}

}  // namespace hotarubi::iss
